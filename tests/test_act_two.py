"""Node-level checks for ``RunwayML_ActTwo``.

Coercion of arbitrary input shapes is exercised in
``tests/unit/media/test_coercion.py``; this module covers the wiring between
Act Two's parameters, ``PublicArtifactUrlParameter`` uploads, and the API
payload.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _stub_public_artifact_dependencies() -> Iterator[None]:
    """Patch out Griptape Cloud calls so ``PublicArtifactUrlParameter`` constructs offline.

    The component normally requires ``GT_CLOUD_API_KEY`` and a live bucket lookup at
    construction time; we don't need either to exercise node-level wiring.
    """
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
    from act_two import RunwayML_ActTwo

    return RunwayML_ActTwo(name="test")


@pytest.fixture(autouse=True)
def _stub_secret() -> Iterator[None]:
    with patch("act_two.GriptapeNodes") as gn:
        gn.SecretsManager.return_value.get_secret.return_value = "fake-key"
        yield


def _set_minimal_required(node) -> None:
    node.set_parameter_value("character_type", "image")
    node.set_parameter_value("ratio", "1280:720")
    node.set_parameter_value("expression_intensity", 3)
    node.set_parameter_value("model", "act_two")


def test_public_url_wrappers_registered_for_each_media_input(node) -> None:
    """Each media input must be wrapped so RunwayML receives an HTTPS URL, not a data URI."""
    assert node._public_character_image is not None
    assert node._public_character_video is not None
    assert node._public_reference_video is not None
    # The wrapped ``Parameter`` instances were registered on the node, with their original names.
    assert node.get_parameter_by_name("character_image") is node._public_character_image._parameter
    assert node.get_parameter_by_name("character_video") is node._public_character_video._parameter
    assert node.get_parameter_by_name("reference_video") is node._public_reference_video._parameter


def test_validation_passes_when_required_inputs_are_present(node) -> None:
    """``validate_node`` only checks presence; it must not trigger uploads."""
    _set_minimal_required(node)
    node.set_parameter_value("character_image", "{inputs}/image_196.png")
    node.set_parameter_value("reference_video", "{inputs}/clip.mp4")

    with (
        patch.object(node._public_character_image, "get_public_url_for_parameter") as mock_image,
        patch.object(node._public_reference_video, "get_public_url_for_parameter") as mock_video,
    ):
        assert node.validate_node() is None
        mock_image.assert_not_called()
        mock_video.assert_not_called()


def test_missing_character_image_yields_required_error(node) -> None:
    _set_minimal_required(node)
    node.set_parameter_value("character_image", None)
    node.set_parameter_value("reference_video", "https://example.com/v.mp4")

    errors = node.validate_node()
    assert errors is not None
    assert any("Character image is required" in str(e) for e in errors)


def test_missing_reference_video_yields_required_error(node) -> None:
    _set_minimal_required(node)
    node.set_parameter_value("character_image", "https://example.com/i.png")
    node.set_parameter_value("reference_video", None)

    errors = node.validate_node()
    assert errors is not None
    assert any("Reference video" in str(e) for e in errors)


def test_process_uploads_image_and_video_then_cleans_up(node) -> None:
    """Regression: with a macro-path image and video, both wrappers upload, the API gets
    HTTPS URLs (no data URIs), and uploaded artifacts are deleted regardless of API outcome.
    """
    _set_minimal_required(node)
    node.set_parameter_value("character_image", "{inputs}/image_196.png")
    node.set_parameter_value("reference_video", "{inputs}/clip.mp4")

    image_url = "https://cloud.griptape.ai/storage/abc/img.png"
    video_url = "https://cloud.griptape.ai/storage/abc/clip.mp4"

    with (
        patch.object(node._public_character_image, "get_public_url_for_parameter", return_value=image_url) as up_image,
        patch.object(node._public_reference_video, "get_public_url_for_parameter", return_value=video_url) as up_video,
        patch.object(node._public_character_image, "delete_uploaded_artifact") as cleanup_image,
        patch.object(node._public_character_video, "delete_uploaded_artifact") as cleanup_char_video,
        patch.object(node._public_reference_video, "delete_uploaded_artifact") as cleanup_ref_video,
        patch("act_two.requests") as mock_requests,
    ):
        post_response = MagicMock(status_code=200)
        post_response.json.return_value = {"id": "task-1"}
        get_response = MagicMock(status_code=200)
        get_response.json.return_value = {"status": "SUCCEEDED", "output": [{"url": "https://out/v.mp4"}]}
        mock_requests.post.return_value = post_response
        mock_requests.get.return_value = get_response

        # Run the yielded async work synchronously.
        result_gen = node.process()
        async_fn = next(result_gen)
        with patch("act_two.time.sleep"):
            with patch.object(node, "_download_and_store_video", create=True):
                # Defensive: if the node tries to download, just no-op.
                async_fn()

    up_image.assert_called_once()
    up_video.assert_called_once()

    sent_payload = mock_requests.post.call_args.kwargs["json"]
    assert sent_payload["character"]["uri"] == image_url
    assert sent_payload["reference"]["uri"] == video_url
    assert not sent_payload["character"]["uri"].startswith("data:")
    assert not sent_payload["reference"]["uri"].startswith("data:")

    cleanup_image.assert_called_once()
    cleanup_char_video.assert_called_once()
    cleanup_ref_video.assert_called_once()


def test_process_cleans_up_when_api_call_fails(node) -> None:
    _set_minimal_required(node)
    node.set_parameter_value("character_image", "{inputs}/image_196.png")
    node.set_parameter_value("reference_video", "{inputs}/clip.mp4")

    with (
        patch.object(node._public_character_image, "get_public_url_for_parameter", return_value="https://x/i.png"),
        patch.object(node._public_reference_video, "get_public_url_for_parameter", return_value="https://x/v.mp4"),
        patch.object(node._public_character_image, "delete_uploaded_artifact") as cleanup_image,
        patch.object(node._public_character_video, "delete_uploaded_artifact") as cleanup_char_video,
        patch.object(node._public_reference_video, "delete_uploaded_artifact") as cleanup_ref_video,
        patch("act_two.requests") as mock_requests,
    ):
        post_response = MagicMock(status_code=500, text="server boom")
        mock_requests.post.return_value = post_response

        result_gen = node.process()
        async_fn = next(result_gen)
        with patch("act_two.time.sleep"):
            async_fn()

    cleanup_image.assert_called_once()
    cleanup_char_video.assert_called_once()
    cleanup_ref_video.assert_called_once()
