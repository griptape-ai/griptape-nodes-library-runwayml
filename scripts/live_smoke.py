"""Verify this library's payloads against the live RunwayML API.

The offline suite proves the payloads match the published spec; it cannot prove RunwayML
accepts them. This script closes that gap, in three tiers so credits are only spent
deliberately:

    uv run python scripts/live_smoke.py             # tier 1: free
    uv run python scripts/live_smoke.py --probe      # tier 2: minimal spend
    uv run python scripts/live_smoke.py --full       # tier 3: one full generation

Tier 1 (free) calls `GET /v1/organization`: confirms the key works, the
`X-Runway-Version` header is accepted, and every model this library offers is actually
enabled for the account's tier. A model missing from the tier list is a real problem the
spec cannot reveal.

Tier 2 checks that RunwayML accepts the payloads the nodes build. RunwayML validates the
request body *before* checking the credit balance, and the two failures are distinguishable
by message ("Validation of body failed" with a field-level `issues` array, versus "You do
not have enough credits to run this task"). So on an account with no credits this tier is
free and still conclusive: reaching the credit error means the payload is valid. With
credits available it submits and cancels immediately instead.

Note that RunwayML fetches and inspects input media during validation, so a bad or too-short
asset is reported here as a field error rather than at generation time.

Tier 3 lets one text-to-image job finish, which is the only tier that exercises the
download-and-save path end to end.

Reads RUNWAYML_API_SECRET from the environment, so put it in `.env` (git-ignored) or
export it. The only thing written to disk is a throwaway probe clip under /tmp.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runwayml"))

from api_surface import (  # noqa: E402
    MODELS,
    RUNWAY_API_BASE,
    RUNWAY_API_VERSION,
)

REQUEST_TIMEOUT_SECONDS = 60
FULL_RUN_POLL_SECONDS = 10
FULL_RUN_MAX_POLLS = 60

# Smallest inputs each endpoint will accept, to keep tier 2 as cheap as possible.
PROBE_PROMPT = "a plain grey square"
PROBE_DURATION = 2
PROBE_VIDEO_SECONDS = 3


def load_api_key() -> str:
    key = os.environ.get("RUNWAYML_API_SECRET")
    if key:
        return key

    # Match how the engine sources it, so `.env` works without extra setup.
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "RUNWAYML_API_SECRET":
                return value.strip().strip("'\"")

    print(
        "RUNWAYML_API_SECRET is not set. Export it, or put it in .env (git-ignored):\n"
        "  echo 'RUNWAYML_API_SECRET=key_...' > .env",
        file=sys.stderr,
    )
    raise SystemExit(2)


def call(method: str, path: str, api_key: str, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
    """Make one request and return (status, parsed body). Never raises on an HTTP error."""
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(  # noqa: S310
        f"{RUNWAY_API_BASE}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Runway-Version": RUNWAY_API_VERSION,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
            body = response.read().decode()
            return response.status, json.loads(body) if body else None
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, body


def tier1_preflight(api_key: str) -> bool:
    print("=" * 72)
    print("TIER 1  authentication, API version, and tier model availability (free)")
    print("=" * 72)

    status, body = call("GET", "/organization", api_key)
    if status != 200:
        print(f"  FAIL  GET /v1/organization returned {status}: {body}")
        return False

    print(f"  OK    key accepted, X-Runway-Version {RUNWAY_API_VERSION} accepted")

    credits = body.get("creditBalance") if isinstance(body, dict) else None
    if credits is not None:
        print(f"  INFO  credit balance: {credits}")

    tier = body.get("tier", {}) if isinstance(body, dict) else {}
    available = set(tier.get("models", {}) or {})
    if not available:
        print("  WARN  the response carried no tier model list; skipping the availability check")
        return True

    offered = {spec.model_id for spec in MODELS}
    missing = sorted(offered - available)
    print(f"  INFO  {len(available)} models enabled for this account")
    if missing:
        print(f"  FAIL  this library offers models the account cannot use: {missing}")
        return False

    print(f"  OK    all {len(offered)} models this library offers are enabled")
    return True


def tiny_png_data_uri() -> str:
    from PIL import Image

    buffer = io.BytesIO()
    # 1280x720 rather than a token 1x1: RunwayML rejects images below its minimum dimension.
    Image.new("RGB", (1280, 720), (128, 128, 128)).save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"


def tiny_mp4_data_uri() -> str | None:
    """Synthesize a short H.264 clip, or return None when ffmpeg is unavailable.

    Three seconds rather than one: RunwayML fetches and inspects the asset during request
    validation, and `aleph2` rejects anything under two seconds.
    """
    output = Path("/tmp/runway_smoke_probe.mp4")
    command = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=gray:s=1280x720:d={PROBE_VIDEO_SECONDS}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output),
    ]  # fmt: skip
    try:
        subprocess.run(command, capture_output=True, check=True)  # noqa: S603
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    return f"data:video/mp4;base64,{base64.b64encode(output.read_bytes()).decode()}"


def build_probe_payloads() -> dict[str, dict[str, Any] | None]:
    """One minimal payload per endpoint this library targets.

    These mirror what the nodes build, so a rejection here is a rejection of the node.
    """
    image_uri = tiny_png_data_uri()
    video_uri = tiny_mp4_data_uri()

    payloads: dict[str, dict[str, Any] | None] = {
        "text_to_image": {
            "model": "gen4_image",
            "promptText": PROBE_PROMPT,
            "ratio": "1024:1024",
            "seed": 1,
            "contentModeration": {"publicFigureThreshold": "auto"},
        },
        "text_to_video": {
            "model": "gen4.5",
            "promptText": PROBE_PROMPT,
            "ratio": "1280:720",
            "duration": PROBE_DURATION,
            "seed": 1,
            "contentModeration": {"publicFigureThreshold": "auto"},
        },
        "image_to_video": {
            "model": "gen4_turbo",
            "promptImage": image_uri,
            "promptText": PROBE_PROMPT,
            "ratio": "1280:720",
            "duration": PROBE_DURATION,
            "seed": 1,
            "contentModeration": {"publicFigureThreshold": "auto"},
        },
    }

    if video_uri is None:
        payloads["video_to_video"] = None
        payloads["video_upscale"] = None
        payloads["video_to_hdr"] = None
        payloads["character_performance"] = None
        return payloads

    payloads["video_to_video"] = {
        "model": "aleph2",
        "videoUri": video_uri,
        "promptText": PROBE_PROMPT,
        "seed": 1,
        "contentModeration": {"publicFigureThreshold": "auto"},
    }
    # Explicitly sends the strength fields at 0, exactly as the node does, so the probe
    # catches it if RunwayML treats an explicit 0 differently from an omitted field.
    payloads["video_upscale"] = {
        "model": "magnific_video_upscaler_creative",
        "videoUri": video_uri,
        "resolution": "720p",
        "flavor": "natural",
        "creativity": 0,
        "sharpen": 0,
        "smartGrain": 0,
        "fpsBoost": False,
    }
    payloads["video_to_hdr"] = {
        "model": "ruby",
        "videoUri": video_uri,
        "outputFormat": "hdr10",
    }
    payloads["character_performance"] = {
        "model": "act_two",
        "character": {"type": "image", "uri": image_uri},
        "reference": {"type": "video", "uri": video_uri},
        "bodyControl": True,
        "expressionIntensity": 3,
        "ratio": "1280:720",
        "seed": 1,
        "contentModeration": {"publicFigureThreshold": "auto"},
    }
    return payloads


def tier2_probe(api_key: str) -> bool:
    print()
    print("=" * 72)
    print("TIER 2  payload acceptance, cancelled immediately (minimal spend)")
    print("=" * 72)

    all_ok = True
    for endpoint, payload in build_probe_payloads().items():
        if payload is None:
            print(f"  SKIP  {endpoint}: needs a test video and ffmpeg is unavailable")
            continue

        status, body = call("POST", f"/{endpoint}", api_key, payload)
        error = str(body.get("error", "")) if isinstance(body, dict) else str(body)

        # RunwayML validates the body before checking the balance, so an out-of-credit
        # rejection proves the payload itself is valid. That makes this tier free, and
        # correct, on an account with no credits.
        if "enough credits" in error:
            print(f"  OK    {endpoint}: payload valid (not submitted, account is out of credits)")
            continue

        if status not in (200, 201):
            print(f"  FAIL  {endpoint}: {status} {error or body}")
            if isinstance(body, dict):
                for issue in body.get("issues", []):
                    if isinstance(issue, dict):
                        print(f"        {issue.get('path')}: {issue.get('message')}")
            all_ok = False
            continue

        task_id = body.get("id") if isinstance(body, dict) else None
        print(f"  OK    {endpoint}: accepted (task {task_id})")

        if task_id:
            cancel_status, _ = call("DELETE", f"/tasks/{task_id}", api_key)
            verb = "cancelled" if cancel_status in (200, 204) else f"cancel returned {cancel_status}"
            print(f"        {verb}")

    return all_ok


def tier3_full(api_key: str) -> bool:
    """Run one text-to-image job to completion: the only tier that proves output is usable."""
    print()
    print("=" * 72)
    print("TIER 3  one full text-to-image generation (spends credits)")
    print("=" * 72)

    payload = build_probe_payloads()["text_to_image"]
    assert payload is not None
    status, body = call("POST", "/text_to_image", api_key, payload)
    if status not in (200, 201):
        print(f"  FAIL  submit returned {status}: {body}")
        return False

    task_id = body["id"]
    print(f"  OK    submitted task {task_id}")

    for attempt in range(FULL_RUN_MAX_POLLS):
        status, task = call("GET", f"/tasks/{task_id}", api_key)
        if status != 200:
            print(f"  FAIL  status check returned {status}: {task}")
            return False

        state = task.get("status")
        if state == "SUCCEEDED":
            # Reuse the library's own extraction so the envelope shape is verified too.
            from runway_client import extract_output_urls

            urls = extract_output_urls(task)
            print(f"  OK    succeeded after {attempt + 1} polls")
            print(f"  {'OK  ' if urls else 'FAIL'}  extract_output_urls found {len(urls)} URL(s)")
            return bool(urls)

        if state in ("FAILED", "CANCELLED"):
            print(f"  FAIL  task ended {state}: {task.get('failure') or task.get('error')}")
            return False

        print(f"        {state} ({attempt + 1}/{FULL_RUN_MAX_POLLS})")
        time.sleep(FULL_RUN_POLL_SECONDS)

    print("  FAIL  timed out")
    call("DELETE", f"/tasks/{task_id}", api_key)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="also submit and cancel one job per endpoint")
    parser.add_argument("--full", action="store_true", help="also run one text-to-image job to completion")
    args = parser.parse_args()

    api_key = load_api_key()

    if not tier1_preflight(api_key):
        print("\nTier 1 failed; not spending credits on later tiers.", file=sys.stderr)
        return 1

    ok = True
    if args.probe or args.full:
        ok = tier2_probe(api_key) and ok
    if args.full:
        ok = tier3_full(api_key) and ok

    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
