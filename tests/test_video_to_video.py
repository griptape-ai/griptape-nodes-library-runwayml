"""Node-level checks for ``RunwayML_VideoToVideo``.

Coercion of arbitrary input shapes is exercised in
``tests/unit/media/test_coercion.py``; this module covers V2V-specific
wiring: ``PublicArtifactUrlParameter`` upload of the input video, and
reference-image format validation.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _stub_public_artifact_dependencies() -> Iterator[None]:
    with (
        patch(
            "griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter.PublicArtifactUrlParameter._get_secret_value",
            return_value="fake-key",
        ),
        patch(
            "griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter.PublicArtifactUrlParameter._get_bucket_id",
            return_value="fake-bucket",
        ),
        patch(
            "griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter.GriptapeCloudStorageDriver"
        ),
    ):
        yield


@pytest.fixture
def node():
    from video_to_video import RunwayML_VideoToVideo

    return RunwayML_VideoToVideo(name="test")


@pytest.fixture(autouse=True)
def _stub_secret() -> Iterator[None]:
    with patch("video_to_video.GriptapeNodes") as gn:
        gn.SecretsManager.return_value.get_secret.return_value = "fake-key"
        yield


def _set_minimal_required(node) -> None:
    node.set_parameter_value("prompt", "hello")
    node.set_parameter_value("model", "gen4_aleph")


def test_video_input_is_wrapped_with_public_artifact_url_parameter(node) -> None:
    assert node._public_video is not None
    assert node.get_parameter_by_name("video") is node._public_video._parameter


def test_validation_passes_when_required_inputs_are_present(node) -> None:
    _set_minimal_required(node)
    node.set_parameter_value("video", "{inputs}/clip.mp4")

    with patch.object(node._public_video, "get_public_url_for_parameter") as mock_upload:
        assert node.validate_node() is None
        mock_upload.assert_not_called()


def test_missing_video_yields_required_error(node) -> None:
    _set_minimal_required(node)
    node.set_parameter_value("video", None)

    errors = node.validate_node()
    assert errors is not None
    assert any("required" in str(e) for e in errors)


def test_missing_prompt_yields_required_error(node) -> None:
    _set_minimal_required(node)
    node.set_parameter_value("video", "https://example.com/v.mp4")
    node.set_parameter_value("prompt", "")

    errors = node.validate_node()
    assert errors is not None
    assert any("prompt" in str(e).lower() for e in errors)


def test_process_uploads_video_then_cleans_up(node) -> None:
    """Macro paths get uploaded to Griptape Cloud; the API receives an HTTPS URL."""
    _set_minimal_required(node)
    node.set_parameter_value("video", "{inputs}/clip.mp4")

    public_url = "https://cloud.griptape.ai/storage/abc/clip.mp4"

    with (
        patch.object(node._public_video, "get_public_url_for_parameter", return_value=public_url) as upload,
        patch.object(node._public_video, "delete_uploaded_artifact") as cleanup,
        patch("video_to_video.requests") as mock_requests,
        patch("video_to_video.time.sleep"),
    ):
        post = MagicMock(status_code=200)
        post.json.return_value = {"id": "task-1"}
        get_resp = MagicMock(status_code=200)
        get_resp.json.return_value = {"status": "SUCCEEDED", "output": [{"url": "https://out/v.mp4"}]}
        mock_requests.post.return_value = post
        mock_requests.get.return_value = get_resp

        result_gen = node.process()
        async_fn = next(result_gen)
        async_fn()

    upload.assert_called_once()
    payload = mock_requests.post.call_args.kwargs["json"]
    assert payload["videoUri"] == public_url
    assert not payload["videoUri"].startswith("data:")
    cleanup.assert_called_once()


def test_process_cleans_up_when_api_call_fails(node) -> None:
    _set_minimal_required(node)
    node.set_parameter_value("video", "{inputs}/clip.mp4")

    with (
        patch.object(node._public_video, "get_public_url_for_parameter", return_value="https://x/v.mp4"),
        patch.object(node._public_video, "delete_uploaded_artifact") as cleanup,
        patch("video_to_video.requests") as mock_requests,
        patch("video_to_video.time.sleep"),
    ):
        post = MagicMock(status_code=500, text="server boom")
        mock_requests.post.return_value = post

        result_gen = node.process()
        async_fn = next(result_gen)
        async_fn()

    cleanup.assert_called_once()


# ---------------------------------------------------------------------------
# Reference image format validation (kept inline; not migrated to PublicArtifactUrlParameter)
# ---------------------------------------------------------------------------


def test_supported_reference_image_data_uri_passes(node) -> None:
    node.set_parameter_value("reference_image", "data:image/jpeg;base64,AAAA")
    assert node._get_image_data_uri("reference_image") == "data:image/jpeg;base64,AAAA"


def test_unsupported_reference_image_data_uri_raises(node) -> None:
    node.set_parameter_value("reference_image", "data:image/gif;base64,AAAA")
    with pytest.raises(ValueError, match="Unsupported reference image format"):
        node._get_image_data_uri("reference_image")
