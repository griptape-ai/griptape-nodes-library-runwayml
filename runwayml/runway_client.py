"""Async HTTP client for Runway's task API.

Every generation endpoint follows the same shape: POST a payload, get a task id, poll
`GET /v1/tasks/{id}` until it reaches a terminal state, then read output URLs off the
response. That loop, its error mapping, and the output-URL extraction were duplicated
across five nodes, so they live here once.

Raw HTTP rather than the official `runwayml` SDK: the API adds models faster than the
SDK tracks them, and this library's own package is itself named `runwayml`, so
depending on the SDK would collide on import.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from typing import Any

import httpx
from api_surface import MAX_DATA_URI_BYTES, RUNWAY_API_BASE, RUNWAY_API_VERSION
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger

API_KEY_ENV_VAR = "RUNWAYML_API_SECRET"
SERVICE = "RunwayML"

SUBMIT_TIMEOUT_SECONDS = 60.0
STATUS_TIMEOUT_SECONDS = 30.0
POLL_INTERVAL_SECONDS = 10
# 20 minutes of polling. Runway's own ceiling is lower for every model this library
# exposes, so hitting this means something is wrong rather than merely slow.
MAX_POLL_ATTEMPTS = 120
# Consecutive status-request failures tolerated before giving up. One 502 or 429 in a 20-minute
# poll must not destroy a job that is already paid for and still running.
MAX_POLL_ERRORS = 3

STATUS_SUCCEEDED = "SUCCEEDED"
STATUS_FAILED = "FAILED"
STATUS_CANCELLED = "CANCELLED"
TERMINAL_STATUSES = frozenset({STATUS_SUCCEEDED, STATUS_FAILED, STATUS_CANCELLED})

# Runway rejects an oversized inline data URI with this status. The message it returns is
# about request size and does not mention that a shorter path exists, so nodes translate it.
HTTP_PAYLOAD_TOO_LARGE = 413


class RunwayError(Exception):
    """Base class for every Runway failure this library reports to the user."""


class RunwayAuthError(RunwayError):
    pass


class RunwayRequestError(RunwayError):
    """Runway rejected the request. Carries the status code so callers can special-case it."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class RunwayTaskFailedError(RunwayError):
    pass


class RunwayTaskTimeoutError(RunwayError):
    pass


def get_api_key() -> str:
    """Read the Runway API secret.

    Raises:
        RunwayAuthError: If no secret is configured.
    """
    api_key = GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR)
    if not api_key:
        msg = (
            f"Attempted to call RunwayML. Failed because no API key is configured. "
            f"Set {API_KEY_ENV_VAR} in the Settings > API Keys & Secrets panel, or in your environment."
        )
        raise RunwayAuthError(msg)
    return api_key


def extract_output_urls(task: dict[str, Any]) -> list[str]:
    """Pull output URLs off a completed task.

    Runway has returned outputs as a bare string, a dict with a `url` key, and a list of
    either, depending on endpoint and vintage. All four shapes are accepted so a node does
    not fail on a successful generation just because the envelope changed.
    """
    output = task.get("output")
    if not output:
        return []

    candidates = output if isinstance(output, list) else [output]
    urls = []
    for item in candidates:
        if isinstance(item, dict) and "url" in item:
            urls.append(item["url"])
        elif isinstance(item, str) and item.startswith(("http://", "https://")):
            urls.append(item)
    return urls


class RunwayClient:
    """Submits Runway tasks and waits for them.

    One instance per node run. Used as an async context manager so the underlying
    connection pool is always closed:

        async with RunwayClient(node_name="RunwayML I2V") as client:
            task_id = await client.submit("image_to_video", payload)
            urls = await client.await_output(task_id)
    """

    def __init__(
        self,
        node_name: str,
        api_key: str | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._node_name = node_name
        self._api_key = api_key or get_api_key()
        # `transport` is httpx's documented injection point for tests; leaving it None uses
        # the real network stack.
        self._client = httpx.AsyncClient(
            base_url=RUNWAY_API_BASE,
            transport=transport,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "X-Runway-Version": RUNWAY_API_VERSION,
            },
        )

    async def __aenter__(self) -> RunwayClient:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self._client.aclose()

    async def submit(self, endpoint: str, payload: dict[str, Any]) -> str:
        """Create a task and return its id.

        Raises:
            RunwayRequestError: If Runway rejects the request.
        """
        # Payloads carry base64 media, so log the keys rather than the body.
        logger.info("%s: submitting %s task with fields %s", self._node_name, endpoint, sorted(payload))

        self._reject_oversized_media(endpoint, payload)

        try:
            response = await self._client.post(f"/{endpoint}", json=payload, timeout=SUBMIT_TIMEOUT_SECONDS)
        except httpx.RequestError as e:
            msg = f"Attempted to reach RunwayML to start a {endpoint} job. Failed because the request could not be sent: {e}"
            raise RunwayRequestError(msg, status_code=0) from e

        # Any 2xx means the task exists and is billable, so treating a 201 as a failure would
        # discard an id the job can no longer be cancelled or cross-referenced by.
        if not response.is_success:
            raise RunwayRequestError(self._describe_rejection(endpoint, response), status_code=response.status_code)

        task_id = response.json().get("id")
        if not task_id:
            msg = f"Attempted to start a RunwayML {endpoint} job. Failed because the response contained no task id."
            raise RunwayRequestError(msg, status_code=response.status_code)

        logger.info("%s: task %s created", self._node_name, task_id)
        return task_id

    def _reject_oversized_media(self, endpoint: str, payload: dict[str, Any]) -> None:
        """Fail before uploading media RunwayML will refuse.

        Without this the node base64-encodes the file, hands httpx tens of megabytes, waits out
        the transfer, and only then gets a 413. The size is knowable up front.

        Raises:
            RunwayRequestError: If any inlined data URI exceeds RunwayML's cap.
        """
        for field, value in self._iter_inlined_media(payload):
            if len(value) <= MAX_DATA_URI_BYTES:
                continue

            cap_mb = MAX_DATA_URI_BYTES // (1024 * 1024)
            msg = (
                f"Attempted to send media to RunwayML for a {endpoint} job. Failed because "
                f"'{field}' is {len(value) / (1024 * 1024):.1f}MB once encoded and RunwayML "
                f"allows {cap_mb}MB inline, about {cap_mb * 0.66:.1f}MB of source file. "
                "Use a smaller or shorter file, or a public HTTPS URL."
            )
            raise RunwayRequestError(msg, status_code=HTTP_PAYLOAD_TOO_LARGE)

    @classmethod
    def _iter_inlined_media(cls, node: Any, path: str = "") -> Iterator[tuple[str, str]]:
        """Yield every inlined data URI in a payload, with the field path that holds it.

        Walked recursively rather than over the top level: `character_performance` nests its two
        videos under `character.uri`/`reference.uri`, and `video_to_video` puts a reference image
        inside a list, so a flat scan would skip exactly the node that inlines the most.
        """
        if isinstance(node, str):
            if node.startswith("data:"):
                yield path or "request", node
        elif isinstance(node, dict):
            for key, value in node.items():
                yield from cls._iter_inlined_media(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                yield from cls._iter_inlined_media(value, f"{path}[{index}]")

    async def await_output(
        self,
        task_id: str,
        on_status: Callable[[str], None] | None = None,
    ) -> list[str]:
        """Poll until the task finishes and return its output URLs.

        `on_status` is called with each observed status so a node can stream progress.

        Raises:
            RunwayTaskFailedError: If Runway reports the task failed or was cancelled.
            RunwayTaskTimeoutError: If the task is still running at the polling ceiling.
        """
        consecutive_errors = 0
        try:
            for attempt in range(MAX_POLL_ATTEMPTS):
                # Poll before sleeping: a cached or fast result should not be held back by a
                # fixed delay, which previously cost every job a full interval.
                try:
                    task = await self._get_task(task_id)
                except RunwayRequestError as e:
                    # The status request failed, not the job. Retry before giving up: the job is
                    # already billed and still running, so abandoning it over one blip is worse
                    # than waiting another interval.
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_POLL_ERRORS:
                        raise
                    logger.warning(
                        "%s: status check for task %s failed (%d/%d), retrying: %s",
                        self._node_name,
                        task_id,
                        consecutive_errors,
                        MAX_POLL_ERRORS,
                        e,
                    )
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue

                consecutive_errors = 0
                status = task.get("status", "UNKNOWN")

                if on_status is not None:
                    on_status(status)
                logger.info(
                    "%s: task %s status %s (attempt %d/%d)",
                    self._node_name,
                    task_id,
                    status,
                    attempt + 1,
                    MAX_POLL_ATTEMPTS,
                )

                if status in TERMINAL_STATUSES:
                    return self._read_terminal_task(task_id, task, status)

                await asyncio.sleep(POLL_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            # Stopping the flow cancels this task, and the cancellation can land in the status
            # request as readily as in the sleep -- the request window is the longer of the two.
            # Runway keeps generating, and billing, unless it is told to stop.
            await self.cancel(task_id)
            raise
        except RunwayRequestError:
            # Reached only after MAX_POLL_ERRORS consecutive failures, so the job is genuinely
            # unreachable rather than momentarily flaky. Cancel so it stops billing, matching the
            # timeout and cancellation exits.
            await self.cancel(task_id)
            raise

        timeout_seconds = MAX_POLL_ATTEMPTS * POLL_INTERVAL_SECONDS
        await self.cancel(task_id)
        msg = (
            f"Attempted to generate with RunwayML (task {task_id}). Gave up after "
            f"{timeout_seconds // 60} minutes because the job was still running. The job has been cancelled."
        )
        raise RunwayTaskTimeoutError(msg)

    async def cancel(self, task_id: str) -> None:
        """Best-effort cancel so an abandoned task stops consuming credits.

        Never raises: this runs while another failure is already being reported, and
        replacing that failure with a cancellation error would hide the real cause.
        """
        try:
            await self._client.delete(f"/tasks/{task_id}", timeout=STATUS_TIMEOUT_SECONDS)
        except httpx.HTTPError as e:
            logger.warning("%s: could not cancel task %s: %s", self._node_name, task_id, e)
        else:
            logger.info("%s: cancelled task %s", self._node_name, task_id)

    async def _get_task(self, task_id: str) -> dict[str, Any]:
        try:
            response = await self._client.get(f"/tasks/{task_id}", timeout=STATUS_TIMEOUT_SECONDS)
        except httpx.RequestError as e:
            msg = (
                f"Attempted to check RunwayML task {task_id}. Failed because the status request could not be sent: {e}"
            )
            raise RunwayRequestError(msg, status_code=0) from e

        if not response.is_success:
            msg = (
                f"Attempted to check RunwayML task {task_id}. Failed with status "
                f"{response.status_code}: {response.text}"
            )
            raise RunwayRequestError(msg, status_code=response.status_code)

        return response.json()

    def _read_terminal_task(self, task_id: str, task: dict[str, Any], status: str) -> list[str]:
        if status == STATUS_CANCELLED:
            msg = f"RunwayML task {task_id} was cancelled before it finished."
            raise RunwayTaskFailedError(msg)

        if status == STATUS_FAILED:
            reason = task.get("failure") or task.get("error") or "RunwayML did not give a reason."
            msg = f"Attempted to generate with RunwayML (task {task_id}). Failed because: {reason}"
            raise RunwayTaskFailedError(msg)

        urls = extract_output_urls(task)
        if not urls:
            msg = (
                f"RunwayML reported task {task_id} succeeded but returned no downloadable output. "
                f"Output field was: {task.get('output')!r}"
            )
            raise RunwayTaskFailedError(msg)
        return urls

    def _describe_rejection(self, endpoint: str, response: httpx.Response) -> str:
        """Turn a rejection into something an artist can act on."""
        if response.status_code == HTTP_PAYLOAD_TOO_LARGE:
            return (
                f"Attempted to send media to RunwayML for a {endpoint} job. Failed because the request "
                "was too large. RunwayML caps inlined media at 5MB (about 3.3MB before encoding); "
                "use a smaller file or a public HTTPS URL."
            )
        detail = self._read_error_detail(response)
        return f"Attempted to start a RunwayML {endpoint} job. Failed because: {detail}"

    @staticmethod
    def _read_error_detail(response: httpx.Response) -> str:
        """Extract the useful part of a RunwayML error body.

        RunwayML pairs a generic `error` ("Validation of body failed") with an `issues` array
        naming the offending field and its allowed values. Only the array says what to change,
        so it is pulled out rather than leaving raw JSON for the user to read.
        """
        try:
            body = response.json()
        except ValueError:
            return f"status {response.status_code}: {response.text}"

        if not isinstance(body, dict):
            return f"status {response.status_code}: {response.text}"

        message = str(body.get("error") or f"status {response.status_code}")

        issues = body.get("issues")
        if not isinstance(issues, list):
            return message

        described = []
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            path = issue.get("path")
            field = ".".join(str(p) for p in path) if isinstance(path, list) else str(path or "")
            described.append(f"{field or 'request'}: {issue.get('message', 'invalid')}")

        if not described:
            return message
        return f"{message} ({'; '.join(described)})"
