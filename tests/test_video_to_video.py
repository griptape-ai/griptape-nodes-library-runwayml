import base64
from unittest.mock import patch

import pytest
from griptape.artifacts import ImageUrlArtifact, UrlArtifact

from runwayml.video_to_video import RunwayML_VideoToVideo, VideoUrlArtifact


@pytest.fixture
def node():
    return RunwayML_VideoToVideo(name="test")


class TestCoerceVideoUri:
    """Covers `_coerce_video_uri` — the core scheme-validation helper added in #28."""

    @pytest.mark.parametrize(
        "uri",
        [
            "https://example.com/video.mp4",
            "https://cdn.example.com/path/to/video.mp4?token=abc",
            "runway://dataset/abc-123",
            "data:video/mp4;base64,AAAA",
            "data:video/webm;base64,AAAA",
        ],
    )
    def test_passes_accepted_schemes_through_unchanged(self, node, uri):
        assert node._coerce_video_uri(uri) == uri

    def test_strips_surrounding_whitespace(self, node):
        assert node._coerce_video_uri("  https://example.com/v.mp4  ") == "https://example.com/v.mp4"

    @pytest.mark.parametrize(
        "uri",
        [
            "ftp://example.com/v.mp4",
            "s3://bucket/v.mp4",
            "gs://bucket/v.mp4",
        ],
    )
    def test_rejects_unsupported_schemes(self, node, uri):
        with pytest.raises(ValueError, match="must be an https://"):
            node._coerce_video_uri(uri)

    def test_error_message_includes_offending_value(self, node):
        with pytest.raises(ValueError, match="s3://bad/v.mp4"):
            node._coerce_video_uri("s3://bad/v.mp4")

    @pytest.mark.parametrize(
        "path",
        [
            "/absolute/path/v.mp4",
            "relative/path/v.mp4",
            "v.mp4",
            "/Volumes/griptape/oncall/BUG/inputs/LTX Video Retake_ltx_video_retake_1.mp4",
            # Macro-template paths (resolved against the project by `File`):
            "{inputs}/FHE_Arri-test2_for_face_replace-Brayden_Rec709-720P-reference.mp4",
            "{outputs}/sub/v.mp4",
        ],
    )
    def test_local_file_paths_are_read_into_data_uri(self, node, path):
        """Regression for #28 over-rejecting: local paths must be read via `File`, not rejected."""
        payload = b"local-bytes"
        with (
            patch("runwayml.video_to_video.File") as MockFile,
            patch.object(RunwayML_VideoToVideo, "_transcode_video_file", return_value=None),
        ):
            MockFile.return_value.read_bytes.return_value = payload
            result = node._coerce_video_uri(path)
            MockFile.assert_called_with(path)

        expected = "data:video/mp4;base64," + base64.b64encode(payload).decode()
        assert result == expected

    def test_file_uri_is_read_locally(self, node):
        payload = b"file-uri-bytes"
        with (
            patch("runwayml.video_to_video.File") as MockFile,
            patch.object(RunwayML_VideoToVideo, "_transcode_video_file", return_value=None),
        ):
            MockFile.return_value.read_bytes.return_value = payload
            result = node._coerce_video_uri("file:///tmp/v.mp4")
            # `file://` prefix must be stripped so `File` gets a plain path.
            MockFile.assert_called_with("/tmp/v.mp4")

        expected = "data:video/mp4;base64," + base64.b64encode(payload).decode()
        assert result == expected

    def test_http_url_is_downloaded_and_returned_as_data_uri(self, node):
        payload = b"\x00\x01\x02fake-video-bytes"
        with (
            patch("runwayml.video_to_video.File") as MockFile,
            patch.object(RunwayML_VideoToVideo, "_transcode_video_file", return_value=None),
        ):
            MockFile.return_value.read_bytes.return_value = payload
            result = node._coerce_video_uri("http://example.com/v.mp4")

        expected = "data:video/mp4;base64," + base64.b64encode(payload).decode()
        assert result == expected

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:8080/v.mp4",
            "http://127.0.0.1:8080/v.mp4",
            "http://some-internal-host/v.mp4",
        ],
    )
    def test_http_path_works_for_any_host_not_just_localhost(self, node, url):
        """Regression: previously only `localhost`/`127.0.0.1` HTTP URLs were converted; #27 needed any host."""
        with (
            patch("runwayml.video_to_video.File") as MockFile,
            patch.object(RunwayML_VideoToVideo, "_transcode_video_file", return_value=None),
        ):
            MockFile.return_value.read_bytes.return_value = b"x"
            result = node._coerce_video_uri(url)

        assert result is not None
        assert result.startswith("data:video/mp4;base64,")

    def test_http_download_failure_raises_with_source(self, node):
        """`File` failures are wrapped as a ValueError so `validate_node` surfaces them."""
        with patch("runwayml.video_to_video.File") as MockFile:
            MockFile.return_value.read_bytes.side_effect = RuntimeError("network down")
            with pytest.raises(ValueError, match="http://example.com/v.mp4"):
                node._coerce_video_uri("http://example.com/v.mp4")

    def test_local_path_read_failure_raises_with_source(self, node):
        with patch("runwayml.video_to_video.File") as MockFile:
            MockFile.return_value.read_bytes.side_effect = RuntimeError("no such file")
            with pytest.raises(ValueError, match="/missing/v.mp4"):
                node._coerce_video_uri("/missing/v.mp4")


class TestGetVideoDataUriDispatch:
    """Covers the input-shape dispatch in `_get_video_data_uri`: artifact, str, dict, None."""

    def test_returns_none_when_input_missing(self, node):
        node.set_parameter_value("video", None)
        assert node._get_video_data_uri("video") is None

    @pytest.mark.parametrize(
        "artifact_cls",
        [VideoUrlArtifact, ImageUrlArtifact, UrlArtifact],
    )
    def test_url_artifact_input_is_coerced(self, node, artifact_cls):
        url = "https://example.com/v.mp4"
        if artifact_cls is VideoUrlArtifact:
            artifact = artifact_cls(url=url)
        else:
            artifact = artifact_cls(value=url)
        node.set_parameter_value("video", artifact)
        assert node._get_video_data_uri("video") == url

    def test_string_input_is_coerced(self, node):
        node.set_parameter_value("video", "https://example.com/v.mp4")
        assert node._get_video_data_uri("video") == "https://example.com/v.mp4"

    def test_string_input_with_unsupported_scheme_raises(self, node):
        node.set_parameter_value("video", "s3://bad/v.mp4")
        with pytest.raises(ValueError):
            node._get_video_data_uri("video")

    def test_dict_url_input_is_coerced(self, node):
        node.set_parameter_value(
            "video",
            {"type": "VideoUrlArtifact", "value": "https://example.com/v.mp4"},
        )
        assert node._get_video_data_uri("video") == "https://example.com/v.mp4"

    def test_dict_url_input_with_unsupported_scheme_raises(self, node):
        """Regression: previously the dict path bypassed validation and returned the bad URI as-is."""
        node.set_parameter_value(
            "video",
            {"type": "VideoUrlArtifact", "value": "s3://bad/v.mp4"},
        )
        with pytest.raises(ValueError, match="must be an https://"):
            node._get_video_data_uri("video")

    def test_dict_base64_input_is_wrapped(self, node):
        node.set_parameter_value(
            "video",
            {"base64": "AAAA", "media_type": "video/mp4"},
        )
        assert node._get_video_data_uri("video") == "data:video/mp4;base64,AAAA"

    def test_dict_base64_input_already_prefixed_passes_through(self, node):
        node.set_parameter_value(
            "video",
            {"base64": "data:video/mp4;base64,AAAA", "media_type": "video/mp4"},
        )
        assert node._get_video_data_uri("video") == "data:video/mp4;base64,AAAA"


class TestValidateNodeSurfacingValueError:
    """Covers the new try/except in `validate_node` that surfaces video-URI errors as validation errors."""

    @pytest.fixture(autouse=True)
    def _stub_secret(self):
        with patch("runwayml.video_to_video.GriptapeNodes") as gn:
            gn.SecretsManager.return_value.get_secret.return_value = "fake-key"
            yield gn

    def test_unsupported_video_scheme_appears_as_validation_error(self, node):
        node.set_parameter_value("video", "s3://bad/v.mp4")
        node.set_parameter_value("prompt", "hello")
        node.set_parameter_value("model", "gen4_aleph")

        errors = node.validate_node()

        assert errors is not None
        assert any(isinstance(e, ValueError) and "must be an https://" in str(e) for e in errors), (
            f"expected scheme ValueError in {errors!r}"
        )

    def test_missing_video_yields_required_error(self, node):
        node.set_parameter_value("video", None)
        node.set_parameter_value("prompt", "hello")
        node.set_parameter_value("model", "gen4_aleph")

        errors = node.validate_node()

        assert errors is not None
        assert any("required" in str(e) for e in errors)

    def test_valid_inputs_produce_no_errors(self, node):
        node.set_parameter_value("video", "https://example.com/v.mp4")
        node.set_parameter_value("prompt", "hello")
        node.set_parameter_value("model", "gen4_aleph")

        assert node.validate_node() is None

    def test_http_url_is_downloaded_during_validation(self, node):
        node.set_parameter_value("video", "http://example.com/v.mp4")
        node.set_parameter_value("prompt", "hello")
        node.set_parameter_value("model", "gen4_aleph")

        with (
            patch("runwayml.video_to_video.File") as MockFile,
            patch.object(RunwayML_VideoToVideo, "_transcode_video_file", return_value=None),
        ):
            MockFile.return_value.read_bytes.return_value = b"video-bytes"
            assert node.validate_node() is None

    def test_local_file_path_is_read_during_validation(self, node):
        """Regression for #28 over-rejecting: a local path must validate, not surface as an error."""
        node.set_parameter_value("video", "/Volumes/griptape/inputs/v.mp4")
        node.set_parameter_value("prompt", "hello")
        node.set_parameter_value("model", "gen4_aleph")

        with (
            patch("runwayml.video_to_video.File") as MockFile,
            patch.object(RunwayML_VideoToVideo, "_transcode_video_file", return_value=None),
        ):
            MockFile.return_value.read_bytes.return_value = b"video-bytes"
            assert node.validate_node() is None
