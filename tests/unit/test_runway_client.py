"""Tests for the shared Runway task client.

This is the one place the submit/poll/download round trip lives, so every node's failure
behaviour depends on it. Requests are served by an httpx MockTransport rather than the
network, and `asyncio.sleep` is stubbed so polling tests do not actually wait.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from runway_client import (
    MAX_POLL_ATTEMPTS,
    RunwayClient,
    RunwayRequestError,
    RunwayTaskFailedError,
    RunwayTaskTimeoutError,
    extract_output_urls,
)

TASK_ID = "task-123"
OUTPUT_URL = "https://runway.example/out.mp4"


class FakeRunway:
    """A scripted Runway: one submit response, then a queue of task statuses."""

    def __init__(self, statuses: list[dict[str, Any]], submit_status: int = 200) -> None:
        self._statuses = statuses
        self._submit_status = submit_status
        self.submitted_payloads: list[dict[str, Any]] = []
        self.status_calls = 0
        self.deleted: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            self.submitted_payloads.append(json.loads(request.content))
            if self._submit_status != httpx.codes.OK:
                return httpx.Response(self._submit_status, text="rejected")
            return httpx.Response(200, json={"id": TASK_ID})

        if request.method == "DELETE":
            self.deleted.append(str(request.url))
            return httpx.Response(204)

        index = min(self.status_calls, len(self._statuses) - 1)
        self.status_calls += 1
        return httpx.Response(200, json=self._statuses[index])

    def client(self) -> RunwayClient:
        return RunwayClient(node_name="test", api_key="k", transport=httpx.MockTransport(self.handler))


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record sleeps instead of performing them, so poll timing stays assertable."""
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("runway_client.asyncio.sleep", fake_sleep)
    return slept


class TestExtractOutputUrls:
    @pytest.mark.parametrize(
        "output",
        [
            [{"url": OUTPUT_URL}],
            [OUTPUT_URL],
            {"url": OUTPUT_URL},
            OUTPUT_URL,
        ],
    )
    def test_every_envelope_shape_runway_has_used(self, output: Any) -> None:
        assert extract_output_urls({"output": output}) == [OUTPUT_URL]

    @pytest.mark.parametrize("output", [None, [], {}, "", ["not-a-url"], [{"no_url": 1}]])
    def test_absent_or_unusable_output_yields_nothing(self, output: Any) -> None:
        assert extract_output_urls({"output": output}) == []


class TestSubmit:
    @pytest.mark.asyncio
    async def test_returns_task_id_and_sends_payload(self) -> None:
        runway = FakeRunway([{"status": "SUCCEEDED", "output": [OUTPUT_URL]}])
        async with runway.client() as client:
            assert await client.submit("image_to_video", {"model": "gen4_turbo"}) == TASK_ID
        assert runway.submitted_payloads == [{"model": "gen4_turbo"}]

    @pytest.mark.asyncio
    async def test_rejection_reports_status_code(self) -> None:
        runway = FakeRunway([], submit_status=400)
        async with runway.client() as client:
            with pytest.raises(RunwayRequestError) as excinfo:
                await client.submit("image_to_video", {})
        assert excinfo.value.status_code == httpx.codes.BAD_REQUEST

    @pytest.mark.asyncio
    async def test_payload_too_large_explains_the_size_limit(self) -> None:
        """Runway's own 413 body does not mention that an HTTPS URL avoids the cap."""
        runway = FakeRunway([], submit_status=413)
        async with runway.client() as client:
            with pytest.raises(RunwayRequestError, match="HTTPS URL"):
                await client.submit("image_to_video", {})

    @pytest.mark.asyncio
    async def test_missing_task_id_is_an_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
            return httpx.Response(200, json={})

        async with RunwayClient("test", api_key="k", transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(RunwayRequestError, match="no task id"):
                await client.submit("image_to_video", {})


class TestAwaitOutput:
    @pytest.mark.asyncio
    async def test_returns_urls_on_success(self) -> None:
        runway = FakeRunway([{"status": "SUCCEEDED", "output": [{"url": OUTPUT_URL}]}])
        async with runway.client() as client:
            assert await client.await_output(TASK_ID) == [OUTPUT_URL]

    @pytest.mark.asyncio
    async def test_an_already_finished_task_is_not_delayed(self, _no_sleeping: list[float]) -> None:
        """Polling must happen before sleeping, or every job pays a full interval up front."""
        runway = FakeRunway([{"status": "SUCCEEDED", "output": [OUTPUT_URL]}])
        async with runway.client() as client:
            await client.await_output(TASK_ID)
        assert runway.status_calls == 1
        assert _no_sleeping == []

    @pytest.mark.asyncio
    async def test_polls_until_terminal(self, _no_sleeping: list[float]) -> None:
        runway = FakeRunway(
            [
                {"status": "PENDING"},
                {"status": "RUNNING"},
                {"status": "SUCCEEDED", "output": [OUTPUT_URL]},
            ]
        )
        async with runway.client() as client:
            assert await client.await_output(TASK_ID) == [OUTPUT_URL]
        assert runway.status_calls == 3
        assert len(_no_sleeping) == 2

    @pytest.mark.asyncio
    async def test_reports_each_status_to_the_callback(self) -> None:
        runway = FakeRunway([{"status": "PENDING"}, {"status": "SUCCEEDED", "output": [OUTPUT_URL]}])
        seen: list[str] = []
        async with runway.client() as client:
            await client.await_output(TASK_ID, on_status=seen.append)
        assert seen == ["PENDING", "SUCCEEDED"]

    @pytest.mark.asyncio
    async def test_failure_surfaces_runways_reason(self) -> None:
        runway = FakeRunway([{"status": "FAILED", "failure": "content moderation"}])
        async with runway.client() as client:
            with pytest.raises(RunwayTaskFailedError, match="content moderation"):
                await client.await_output(TASK_ID)

    @pytest.mark.asyncio
    async def test_failure_without_a_reason_still_reports(self) -> None:
        runway = FakeRunway([{"status": "FAILED"}])
        async with runway.client() as client:
            with pytest.raises(RunwayTaskFailedError, match="did not give a reason"):
                await client.await_output(TASK_ID)

    @pytest.mark.asyncio
    async def test_cancelled_task_is_a_failure(self) -> None:
        runway = FakeRunway([{"status": "CANCELLED"}])
        async with runway.client() as client:
            with pytest.raises(RunwayTaskFailedError, match="cancelled"):
                await client.await_output(TASK_ID)

    @pytest.mark.asyncio
    async def test_success_with_no_output_is_a_failure(self) -> None:
        """A node must not report success when there is nothing to download."""
        runway = FakeRunway([{"status": "SUCCEEDED", "output": []}])
        async with runway.client() as client:
            with pytest.raises(RunwayTaskFailedError, match="no downloadable output"):
                await client.await_output(TASK_ID)

    @pytest.mark.asyncio
    async def test_timeout_cancels_the_task_so_it_stops_billing(self) -> None:
        runway = FakeRunway([{"status": "RUNNING"}])
        async with runway.client() as client:
            with pytest.raises(RunwayTaskTimeoutError):
                await client.await_output(TASK_ID)

        assert runway.status_calls == MAX_POLL_ATTEMPTS
        assert any(TASK_ID in url for url in runway.deleted)

    @pytest.mark.asyncio
    async def test_a_status_request_failure_cancels_the_paid_task(self) -> None:
        """Giving up on a running, already-billed job without stopping it leaks credits."""
        deleted: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "DELETE":
                deleted.append(str(request.url))
                return httpx.Response(204)
            if request.method == "GET":
                return httpx.Response(500, text="boom")
            return httpx.Response(200, json={"id": TASK_ID})

        async with RunwayClient("test", api_key="k", transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(RunwayRequestError):
                await client.await_output(TASK_ID)

        assert any(TASK_ID in url for url in deleted)

    @pytest.mark.asyncio
    async def test_a_status_request_failure_is_reported(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(500, text="boom")
            return httpx.Response(200, json={"id": TASK_ID})

        async with RunwayClient("test", api_key="k", transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(RunwayRequestError, match="500"):
                await client.await_output(TASK_ID)


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_never_raises(self) -> None:
        """Cancel runs while another failure is being reported; it must not mask it."""

        def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
            msg = "network down"
            raise httpx.ConnectError(msg)

        async with RunwayClient("test", api_key="k", transport=httpx.MockTransport(handler)) as client:
            await client.cancel(TASK_ID)


class TestOversizedMediaPreflight:
    """The cap must be found wherever a node puts media, not just at the payload root."""

    OVERSIZED = "data:video/mp4;base64," + "A" * (6 * 1024 * 1024)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("label", "payload"),
        [
            ("top level", {"promptImage": OVERSIZED}),
            ("nested dict (act_two)", {"character": {"type": "video", "uri": OVERSIZED}}),
            ("inside a list (v2v references)", {"references": [{"type": "image", "uri": OVERSIZED}]}),
        ],
    )
    async def test_oversized_media_is_rejected_before_upload(self, label: str, payload: dict[str, Any]) -> None:
        runway = FakeRunway([])
        async with runway.client() as client:
            with pytest.raises(RunwayRequestError, match="allows 5MB inline"):
                await client.submit("image_to_video", payload)

        assert runway.submitted_payloads == [], f"{label}: should not have been uploaded"

    @pytest.mark.asyncio
    async def test_media_within_the_cap_is_sent(self) -> None:
        runway = FakeRunway([{"status": "SUCCEEDED", "output": [OUTPUT_URL]}])
        async with runway.client() as client:
            await client.submit("image_to_video", {"promptImage": "data:image/png;base64,AAAA"})
        assert len(runway.submitted_payloads) == 1


class TestErrorDetail:
    """RunwayML's generic `error` says nothing actionable; the `issues` array does."""

    @staticmethod
    def rejection(body: Any, status: int = 400) -> str:
        captured: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
            if isinstance(body, str):
                return httpx.Response(status, text=body)
            return httpx.Response(status, json=body)

        client = RunwayClient("test", api_key="k", transport=httpx.MockTransport(handler))
        try:
            captured["message"] = ""
            import asyncio

            asyncio.run(client.submit("text_to_image", {}))
        except RunwayRequestError as e:
            return str(e)
        raise AssertionError("expected a rejection")

    def test_field_issues_are_surfaced(self) -> None:
        message = self.rejection(
            {
                "error": "Validation of body failed",
                "issues": [{"path": ["ratio"], "message": 'Invalid option: expected one of "1024:1024"'}],
            }
        )
        assert "Validation of body failed" in message
        assert "ratio:" in message
        assert "1024:1024" in message

    def test_multiple_issues_are_all_reported(self) -> None:
        message = self.rejection(
            {
                "error": "Validation of body failed",
                "issues": [
                    {"path": ["duration"], "message": "Invalid input"},
                    {"path": ["ratio"], "message": "Invalid option"},
                ],
            }
        )
        assert "duration:" in message
        assert "ratio:" in message

    def test_top_level_issue_without_a_path_is_labelled(self) -> None:
        """RunwayML reports unrecognized keys with an empty path."""
        message = self.rejection(
            {"error": "Validation of body failed", "issues": [{"path": [], "message": 'Unrecognized keys: "flavor"'}]}
        )
        assert "request:" in message
        assert "flavor" in message

    def test_error_without_issues_is_passed_through(self) -> None:
        """A sunset notice arrives as a bare `error` and is already actionable."""
        message = self.rejection({"error": 'gen4_aleph reached its sunset date. Use "aleph2".'})
        assert "sunset" in message
        assert "aleph2" in message

    def test_non_json_body_falls_back_to_status_and_text(self) -> None:
        message = self.rejection("upstream exploded", status=502)
        assert "502" in message
        assert "upstream exploded" in message


class TestApiKey:
    def test_a_missing_key_names_where_to_set_it(self) -> None:
        from runway_client import RunwayAuthError, get_api_key

        with patch("runway_client.GriptapeNodes") as gn:
            gn.SecretsManager.return_value.get_secret.return_value = None
            with pytest.raises(RunwayAuthError, match="API Keys & Secrets"):
                get_api_key()

    def test_a_configured_key_is_returned(self) -> None:
        from runway_client import get_api_key

        with patch("runway_client.GriptapeNodes") as gn:
            gn.SecretsManager.return_value.get_secret.return_value = "k"
            assert get_api_key() == "k"


class TestTransportFailures:
    @pytest.mark.asyncio
    async def test_an_unsendable_submit_is_reported(self) -> None:
        """A connection error carries no status code, so it must not be mistaken for a rejection."""

        def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
            msg = "network down"
            raise httpx.ConnectError(msg)

        async with RunwayClient("test", api_key="k", transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(RunwayRequestError, match="could not be sent") as excinfo:
                await client.submit("image_to_video", {})

        assert excinfo.value.status_code == 0

    @pytest.mark.asyncio
    async def test_transient_poll_failures_are_retried_before_giving_up(self) -> None:
        """One blip must not destroy a job that is already billed and still running."""
        from runway_client import MAX_POLL_ERRORS

        attempts = {"get": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                attempts["get"] += 1
                if attempts["get"] <= MAX_POLL_ERRORS - 1:
                    return httpx.Response(502, text="bad gateway")
                return httpx.Response(200, json={"status": "SUCCEEDED", "output": [OUTPUT_URL]})
            return httpx.Response(200, json={"id": TASK_ID})

        async with RunwayClient("test", api_key="k", transport=httpx.MockTransport(handler)) as client:
            assert await client.await_output(TASK_ID) == [OUTPUT_URL]

        assert attempts["get"] == MAX_POLL_ERRORS


class TestHeaders:
    @pytest.mark.asyncio
    async def test_auth_and_version_headers_are_sent(self) -> None:
        captured: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(request.headers)
            return httpx.Response(200, json={"id": TASK_ID})

        async with RunwayClient("test", api_key="secret", transport=httpx.MockTransport(handler)) as client:
            await client.submit("image_to_video", {})

        assert captured["authorization"] == "Bearer secret"
        assert captured["x-runway-version"]
