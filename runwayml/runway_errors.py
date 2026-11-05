"""Errors the RunwayML nodes raise beyond what `NodeError` builds itself.

A missing or refused API key and a failed HTTP call come from `NodeError.from_missing_secret`,
`NodeError.from_http_response` and `NodeError.from_request_exception`. This module adds what those
leave out: RunwayML's `issues` list on a rejected request, and a task that ends without a video or image.

Node files import it by bare name. The engine loads each node file on its own with the manifest's
folder on `sys.path`, so neither the package name nor a relative import resolves.
"""

from __future__ import annotations

from typing import Any

import httpx
from griptape_nodes.exe_types.core_types import NodeError

API_KEY_ENV_VAR = "RUNWAYML_API_SECRET"
API_KEY_URL = "https://dev.runwayml.com"
PROVIDER = "RunwayML"
_PAYLOAD_TOO_LARGE_STATUS = 413


def missing_key_error() -> NodeError:
    return NodeError.from_missing_secret(API_KEY_ENV_VAR, key_url=API_KEY_URL)


def request_exception_error(error: httpx.RequestError, *, task_id: str | None = None) -> NodeError:
    """The error for a request that never got an HTTP response."""
    return NodeError.from_request_exception(
        error, provider=PROVIDER, secret_name=API_KEY_ENV_VAR, fields={"task_id": task_id} if task_id else None
    )


def response_error(response: httpx.Response, *, task_id: str | None = None) -> NodeError:
    """The error for a non-2xx response, with the field problems from RunwayML's `issues` list."""
    fields: dict[str, str | int | float | bool] = {}
    if request_id := response.headers.get("apigw-requestid"):
        fields["request_id"] = request_id
    if task_id:
        fields["task_id"] = task_id
    error = NodeError.from_http_response(response, provider=PROVIDER, secret_name=API_KEY_ENV_VAR, fields=fields)

    if response.status_code == _PAYLOAD_TOO_LARGE_STATUS:
        message = (
            "RunwayML rejected the request as too large. It caps inlined media at 5MB "
            "(about 3.3MB before encoding). Use a smaller file or a public HTTPS URL."
        )
        return NodeError(message, fields=error.fields, response=error.response, links=error.links)

    issues = (error.response or {}).get("issues")
    problems = (
        [_describe_issue(issue) for issue in issues if isinstance(issue, dict)] if isinstance(issues, list) else []
    )
    if not problems:
        return error
    return NodeError(
        f"{error} ({'; '.join(problems)})", fields=error.fields, response=error.response, links=error.links
    )


def task_error(task_id: str, task: dict[str, Any], status: str) -> NodeError:
    """The error for a task that ended FAILED or CANCELLED."""
    fields: dict[str, str | int | float | bool] = {"task_id": task_id, "status": status}
    if failure_code := task.get("failureCode"):
        fields["error_code"] = str(failure_code)
    if status == "CANCELLED":
        return NodeError(f"{PROVIDER} cancelled the task before it finished. Run the node again.", fields=fields)

    reason = str(task.get("failure") or task.get("error") or "").strip()
    response = {key: task[key] for key in ("failure", "failureCode", "error") if task.get(key)}
    return NodeError(
        reason or f"{PROVIDER} could not complete the generation and gave no reason.",
        fields=fields,
        response=response or None,
    )


def _describe_issue(issue: dict[str, Any]) -> str:
    path = issue.get("path")
    field = ".".join(str(p) for p in path) if isinstance(path, list) else str(path or "")
    return f"{field or 'request'}: {issue.get('message', 'invalid')}"
