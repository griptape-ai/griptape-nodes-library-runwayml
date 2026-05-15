"""Node-level checks for ``RunwayML_VideoToVideo``.

The general coercion behaviour is covered in
``tests/unit/media/test_coercion.py``; this module only covers V2V-specific
wiring: format validation on reference images, and the transcoding-aware video
path.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest
from griptape.artifacts import ImageUrlArtifact
from video_to_video import RunwayML_VideoToVideo, VideoUrlArtifact


@pytest.fixture
def node() -> RunwayML_VideoToVideo:
    return RunwayML_VideoToVideo(name="test")


@pytest.fixture(autouse=True)
def _stub_secret() -> Iterator[None]:
    with patch("video_to_video.GriptapeNodes") as gn:
        gn.SecretsManager.return_value.get_secret.return_value = "fake-key"
        yield


def _set_minimal_required(node: RunwayML_VideoToVideo) -> None:
    node.set_parameter_value("prompt", "hello")
    node.set_parameter_value("model", "gen4_aleph")


# ---------------------------------------------------------------------------
# Video pass-through and reads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "video_uri",
    [
        "https://example.com/v.mp4",
        "runway://dataset/abc",
        "data:video/mp4;base64,AAAA",
    ],
)
def test_pass_through_video_uris_skip_transcoding(node: RunwayML_VideoToVideo, video_uri: str) -> None:
    """Schemes the API accepts directly must not be downloaded or transcoded."""
    _set_minimal_required(node)
    node.set_parameter_value("video", video_uri)

    with (
        patch.object(RunwayML_VideoToVideo, "_read_to_data_uri") as mock_read,
        patch.object(RunwayML_VideoToVideo, "_transcode_video_file") as mock_transcode,
    ):
        assert node._get_video_data_uri("video") == video_uri
        mock_read.assert_not_called()
        mock_transcode.assert_not_called()


def test_local_video_path_is_read_and_transcoded(node: RunwayML_VideoToVideo) -> None:
    _set_minimal_required(node)
    node.set_parameter_value("video", "/Volumes/inputs/v.mp4")

    with patch.object(
        RunwayML_VideoToVideo,
        "_read_to_data_uri",
        return_value="data:video/mp4;base64,READ",
    ) as mock_read:
        assert node._get_video_data_uri("video") == "data:video/mp4;base64,READ"
        mock_read.assert_called_once_with("/Volumes/inputs/v.mp4")


def test_macro_path_video_url_artifact_is_read(node: RunwayML_VideoToVideo) -> None:
    """Regression: ``LoadVideo`` outputs use ``{inputs}/...`` macro paths."""
    _set_minimal_required(node)
    node.set_parameter_value("video", VideoUrlArtifact("{inputs}/clip.mp4"))

    with patch.object(
        RunwayML_VideoToVideo,
        "_read_to_data_uri",
        return_value="data:video/mp4;base64,READ",
    ) as mock_read:
        assert node._get_video_data_uri("video") == "data:video/mp4;base64,READ"
        mock_read.assert_called_once_with("{inputs}/clip.mp4")


# ---------------------------------------------------------------------------
# Reference image format validation
# ---------------------------------------------------------------------------


def test_supported_image_data_uri_passes(node: RunwayML_VideoToVideo) -> None:
    node.set_parameter_value("reference_image", "data:image/jpeg;base64,AAAA")
    assert node._get_image_data_uri("reference_image") == "data:image/jpeg;base64,AAAA"


def test_unsupported_image_data_uri_raises(node: RunwayML_VideoToVideo) -> None:
    node.set_parameter_value("reference_image", "data:image/gif;base64,AAAA")
    with pytest.raises(ValueError, match="Unsupported reference image format"):
        node._get_image_data_uri("reference_image")


def test_https_image_url_is_downloaded_for_format_check(node: RunwayML_VideoToVideo) -> None:
    """Force-fetch HTTPS so we can inspect the content type before sending to RunwayML."""
    node.set_parameter_value("reference_image", ImageUrlArtifact("https://example.com/i.png"))

    with patch("media.coercion.File") as MockFile:
        MockFile.return_value.read_data_uri.return_value = "data:image/png;base64,READ"
        result = node._get_image_data_uri("reference_image")

    assert result == "data:image/png;base64,READ"
    MockFile.assert_called_with("https://example.com/i.png")


# ---------------------------------------------------------------------------
# validate_node
# ---------------------------------------------------------------------------


def test_valid_https_video_validates(node: RunwayML_VideoToVideo) -> None:
    _set_minimal_required(node)
    node.set_parameter_value("video", "https://example.com/v.mp4")

    assert node.validate_node() is None


def test_missing_video_yields_required_error(node: RunwayML_VideoToVideo) -> None:
    _set_minimal_required(node)
    node.set_parameter_value("video", None)

    errors = node.validate_node()
    assert errors is not None
    assert any("required" in str(e) for e in errors)


def test_local_video_path_validates_when_read_succeeds(node: RunwayML_VideoToVideo) -> None:
    """Regression for #28: macro paths and absolute paths must validate, not be rejected."""
    _set_minimal_required(node)
    node.set_parameter_value("video", "{inputs}/clip.mp4")

    with patch.object(
        RunwayML_VideoToVideo,
        "_read_to_data_uri",
        return_value="data:video/mp4;base64,READ",
    ):
        assert node.validate_node() is None
