"""Node-level checks for ``RunwayML_VideoToVideo``.

The general coercion behaviour is covered in
``tests/unit/media/test_coercion.py``; this module only covers V2V-specific
wiring: format validation on reference images, and the transcoding-aware video
path.
"""

from __future__ import annotations

import base64
import subprocess
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from griptape.artifacts import ImageUrlArtifact, VideoUrlArtifact
from video_to_video import RunwayML_VideoToVideo


@pytest.fixture
def node() -> RunwayML_VideoToVideo:
    return RunwayML_VideoToVideo(name="test")


@pytest.fixture(autouse=True)
def _stub_secret() -> Iterator[None]:
    with patch("runway_node.get_api_key", return_value="fake-key"):
        yield


def _set_minimal_required(node: RunwayML_VideoToVideo) -> None:
    node.set_parameter_value("prompt", "hello")
    node.set_parameter_value("model", "aleph2")


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


@pytest.mark.parametrize("source", ["/Volumes/inputs/v.mp4", "/Volumes/inputs/v.mov", "/Volumes/inputs/v.webm"])
def test_containers_runway_accepts_are_not_recompressed(node: RunwayML_VideoToVideo, source: str) -> None:
    """This node can be asked for ProRes 4444, so re-encoding the source would defeat it."""
    _set_minimal_required(node)

    with (
        patch("video_to_video.File") as MockFile,
        patch.object(RunwayML_VideoToVideo, "_transcode_video_file") as mock_transcode,
    ):
        MockFile.return_value.read_bytes.return_value = b"RAW"
        result = node._read_to_data_uri(source)

    mock_transcode.assert_not_called()
    assert result.startswith("data:video/")
    assert base64.b64encode(b"RAW").decode() in result


@pytest.mark.parametrize("source", ["/Volumes/inputs/take_01", "http://host/download?id=5"])
def test_an_unidentifiable_container_is_normalized_not_assumed_to_be_mp4(
    node: RunwayML_VideoToVideo, source: str
) -> None:
    """`guess_type` returns None here; defaulting that to mp4 would skip the transcode."""
    _set_minimal_required(node)

    with (
        patch("video_to_video.File") as MockFile,
        patch.object(RunwayML_VideoToVideo, "_transcode_video_file", return_value=None) as mock_transcode,
    ):
        MockFile.return_value.read_bytes.return_value = b"RAW"
        node._read_to_data_uri(source)

    mock_transcode.assert_called_once()


def test_a_container_runway_rejects_is_normalized(node: RunwayML_VideoToVideo) -> None:
    _set_minimal_required(node)

    with (
        patch("video_to_video.File") as MockFile,
        patch.object(RunwayML_VideoToVideo, "_transcode_video_file", return_value=None) as mock_transcode,
    ):
        MockFile.return_value.read_bytes.return_value = b"RAW"
        result = node._read_to_data_uri("/Volumes/inputs/v.avi")

    mock_transcode.assert_called_once()
    assert result.startswith("data:video/mp4;base64,")


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


class TestTranscodeFallback:
    """The ffmpeg path, which only runs for containers RunwayML will not take."""

    def test_a_successful_transcode_is_used_and_temp_files_are_removed(
        self, node: RunwayML_VideoToVideo, tmp_path: Path
    ) -> None:
        transcoded = tmp_path / "out.transcoded.mp4"

        def fake_run(cmd, **_kwargs):
            Path(cmd[-1]).write_bytes(b"TRANSCODED")
            return subprocess.CompletedProcess(cmd, 0)

        with (
            patch("video_to_video.File") as MockFile,
            patch("video_to_video.subprocess.run", side_effect=fake_run),
            patch("video_to_video.tempfile.NamedTemporaryFile") as mock_temp,
        ):
            MockFile.return_value.read_bytes.return_value = b"RAW"
            source = tmp_path / "src.mp4"
            mock_temp.return_value.__enter__.return_value.name = str(source)
            result = node._read_to_data_uri("/Volumes/inputs/v.avi")

        assert base64.b64encode(b"TRANSCODED").decode() in result
        assert not source.exists()
        assert not transcoded.exists()

    def test_ffmpeg_missing_falls_back_to_the_original_bytes(self, node: RunwayML_VideoToVideo) -> None:
        """No ffmpeg is not a hard failure: RunwayML gets the bytes and rules on them."""
        with (
            patch("video_to_video.File") as MockFile,
            patch("video_to_video.subprocess.run", side_effect=FileNotFoundError),
        ):
            MockFile.return_value.read_bytes.return_value = b"RAW"
            result = node._read_to_data_uri("/Volumes/inputs/v.avi")

        assert base64.b64encode(b"RAW").decode() in result

    def test_a_failed_transcode_falls_back_to_the_original_bytes(self, node: RunwayML_VideoToVideo) -> None:
        with (
            patch("video_to_video.File") as MockFile,
            patch("video_to_video.subprocess.run", side_effect=subprocess.SubprocessError("boom")),
        ):
            MockFile.return_value.read_bytes.return_value = b"RAW"
            result = node._read_to_data_uri("/Volumes/inputs/v.avi")

        assert base64.b64encode(b"RAW").decode() in result

    def test_an_unreadable_source_is_reported(self, node: RunwayML_VideoToVideo) -> None:
        from griptape_nodes.files.file import FileLoadError
        from griptape_nodes.retained_mode.events.os_events import FileIOFailureReason

        with patch("video_to_video.File") as MockFile:
            MockFile.return_value.read_bytes.side_effect = FileLoadError(FileIOFailureReason.FILE_NOT_FOUND, "gone")
            with pytest.raises(ValueError, match="failed to read video"):
                node._read_to_data_uri("/Volumes/inputs/gone.avi")


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


def test_an_unreadable_reference_image_fails_rather_than_being_dropped(node: RunwayML_VideoToVideo) -> None:
    """Dropped silently, RunwayML is paid for an edit that ignored the reference."""
    _set_minimal_required(node)
    node.set_parameter_value("video", "https://example.com/v.mp4")
    node.set_parameter_value("reference_image", "/Volumes/gone/missing.png")

    with patch("media.coercion.File") as MockFile:
        MockFile.return_value.read_data_uri.return_value = None
        with pytest.raises(ValueError, match="reference image could not be read"):
            node.build_payload()


def test_no_reference_image_is_simply_omitted(node: RunwayML_VideoToVideo) -> None:
    _set_minimal_required(node)
    node.set_parameter_value("video", "https://example.com/v.mp4")

    assert "references" not in node.build_payload()


# ---------------------------------------------------------------------------
# validate_before_node_run
# ---------------------------------------------------------------------------


def test_valid_https_video_validates(node: RunwayML_VideoToVideo) -> None:
    _set_minimal_required(node)
    node.set_parameter_value("video", "https://example.com/v.mp4")

    assert node.validate_before_node_run() is None


def test_missing_video_yields_required_error(node: RunwayML_VideoToVideo) -> None:
    _set_minimal_required(node)
    node.set_parameter_value("video", None)

    errors = node.validate_before_node_run()
    assert errors is not None
    assert any("no input video is set" in str(e) for e in errors)


def test_local_video_path_validates_when_read_succeeds(node: RunwayML_VideoToVideo) -> None:
    """Regression for #28: macro paths and absolute paths must validate, not be rejected."""
    _set_minimal_required(node)
    node.set_parameter_value("video", "{inputs}/clip.mp4")

    with patch.object(
        RunwayML_VideoToVideo,
        "_read_to_data_uri",
        return_value="data:video/mp4;base64,READ",
    ):
        assert node.validate_before_node_run() is None
