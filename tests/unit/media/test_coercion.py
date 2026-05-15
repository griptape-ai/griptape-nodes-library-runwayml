"""Tests for the shared media coercion helpers.

Mirrors ``tests/unit/media/test_coercion.py`` in the standard library so
behavior stays consistent across the two helpers.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from act_two import VideoUrlArtifact
from griptape.artifacts import ImageUrlArtifact
from media import (
    DEFAULT_PASS_THROUGH_SCHEMES,
    coerce_media_url_or_data_uri,
    prepare_media_data_uri,
)
from media.coercion import MediaKind

# ---------------------------------------------------------------------------
# IMAGE
# ---------------------------------------------------------------------------


class TestCoerceImage:
    kind: MediaKind = "image"

    def test_none_returns_none(self) -> None:
        assert coerce_media_url_or_data_uri(None, kind=self.kind) is None

    @pytest.mark.parametrize("value", ["", "   ", "\t\n"])
    def test_empty_string_returns_none(self, value: str) -> None:
        assert coerce_media_url_or_data_uri(value, kind=self.kind) is None

    @pytest.mark.parametrize(
        "value",
        [
            "http://example.com/foo.png",
            "https://example.com/foo.png",
            "data:image/png;base64,QUJD",
            "data:image/jpeg;base64,QUJD",
            "runway://dataset/abc-123",
        ],
    )
    def test_known_uri_strings_pass_through(self, value: str) -> None:
        assert coerce_media_url_or_data_uri(value, kind=self.kind) == value

    def test_strings_are_stripped(self) -> None:
        assert (
            coerce_media_url_or_data_uri("  https://example.com/foo.png  ", kind=self.kind)
            == "https://example.com/foo.png"
        )

    @pytest.mark.parametrize(
        "value",
        ["{inputs}/foo.png", "{outputs}/color_bars.png", "{workspace}/scene/frame.jpeg"],
    )
    def test_macro_paths_pass_through_unwrapped(self, value: str) -> None:
        assert coerce_media_url_or_data_uri(value, kind=self.kind) == value

    @pytest.mark.parametrize(
        "value",
        ["/tmp/foo.png", "/abs/path/to/image.jpg", "relative/path/foo.webp", "image.png"],
    )
    def test_filesystem_paths_pass_through_unwrapped(self, value: str) -> None:
        assert coerce_media_url_or_data_uri(value, kind=self.kind) == value

    def test_raw_string_is_not_wrapped_as_base64(self) -> None:
        assert coerce_media_url_or_data_uri("QUJDREVG", kind=self.kind) == "QUJDREVG"

    def test_image_url_artifact_with_http_url(self) -> None:
        artifact = ImageUrlArtifact("https://example.com/foo.png")
        assert coerce_media_url_or_data_uri(artifact, kind=self.kind) == "https://example.com/foo.png"

    def test_image_url_artifact_with_macro_path(self) -> None:
        artifact = ImageUrlArtifact("{inputs}/foo.png")
        assert coerce_media_url_or_data_uri(artifact, kind=self.kind) == "{inputs}/foo.png"

    def test_image_url_artifact_with_filesystem_path(self) -> None:
        artifact = ImageUrlArtifact("/abs/path/foo.png")
        assert coerce_media_url_or_data_uri(artifact, kind=self.kind) == "/abs/path/foo.png"

    def test_artifact_value_is_stripped(self) -> None:
        artifact = SimpleNamespace(value="  {inputs}/foo.png  ", base64=None)
        assert coerce_media_url_or_data_uri(artifact, kind=self.kind) == "{inputs}/foo.png"

    def test_artifact_base64_raw_is_wrapped_as_data_uri(self) -> None:
        artifact = SimpleNamespace(value=None, base64="RAW_IMAGE_B64")
        assert coerce_media_url_or_data_uri(artifact, kind=self.kind) == "data:image/png;base64,RAW_IMAGE_B64"

    def test_artifact_base64_already_data_uri_passes_through(self) -> None:
        artifact = SimpleNamespace(value=None, base64="data:image/jpeg;base64,RAW")
        assert coerce_media_url_or_data_uri(artifact, kind=self.kind) == "data:image/jpeg;base64,RAW"

    def test_artifact_value_takes_precedence_over_base64(self) -> None:
        artifact = SimpleNamespace(value="https://example.com/foo.png", base64="UNUSED")
        assert coerce_media_url_or_data_uri(artifact, kind=self.kind) == "https://example.com/foo.png"

    def test_empty_artifact_returns_none(self) -> None:
        artifact = SimpleNamespace(value="", base64="")
        assert coerce_media_url_or_data_uri(artifact, kind=self.kind) is None

    def test_artifact_with_no_known_attrs_returns_none(self) -> None:
        assert coerce_media_url_or_data_uri(SimpleNamespace(), kind=self.kind) is None

    def test_artifact_with_non_string_value_falls_back_to_base64(self) -> None:
        artifact = SimpleNamespace(value=123, base64="RAW")
        assert coerce_media_url_or_data_uri(artifact, kind=self.kind) == "data:image/png;base64,RAW"


# ---------------------------------------------------------------------------
# VIDEO
# ---------------------------------------------------------------------------


class TestCoerceVideo:
    kind: MediaKind = "video"

    def test_none_returns_none(self) -> None:
        assert coerce_media_url_or_data_uri(None, kind=self.kind) is None

    @pytest.mark.parametrize("value", ["", "   "])
    def test_empty_string_returns_none(self, value: str) -> None:
        assert coerce_media_url_or_data_uri(value, kind=self.kind) is None

    @pytest.mark.parametrize(
        "value",
        [
            "http://example.com/foo.mp4",
            "https://example.com/foo.mp4",
            "data:video/mp4;base64,QUJD",
            "data:video/webm;base64,QUJD",
            "runway://dataset/abc-123",
        ],
    )
    def test_known_uri_strings_pass_through(self, value: str) -> None:
        assert coerce_media_url_or_data_uri(value, kind=self.kind) == value

    @pytest.mark.parametrize(
        "value",
        ["{inputs}/foo.mp4", "{outputs}/render.mp4", "/tmp/clip.mov", "relative/clip.mp4"],
    )
    def test_macro_paths_and_filesystem_paths_pass_through_unwrapped(self, value: str) -> None:
        assert coerce_media_url_or_data_uri(value, kind=self.kind) == value

    def test_video_url_artifact_with_http_url(self) -> None:
        artifact = VideoUrlArtifact("https://example.com/foo.mp4")
        assert coerce_media_url_or_data_uri(artifact, kind=self.kind) == "https://example.com/foo.mp4"

    def test_video_url_artifact_with_macro_path(self) -> None:
        artifact = VideoUrlArtifact("{inputs}/foo.mp4")
        assert coerce_media_url_or_data_uri(artifact, kind=self.kind) == "{inputs}/foo.mp4"

    def test_artifact_base64_raw_is_wrapped_as_video_data_uri(self) -> None:
        artifact = SimpleNamespace(value=None, base64="RAW_VIDEO_B64")
        assert coerce_media_url_or_data_uri(artifact, kind=self.kind) == "data:video/mp4;base64,RAW_VIDEO_B64"

    def test_artifact_base64_already_data_uri_passes_through(self) -> None:
        artifact = SimpleNamespace(value=None, base64="data:video/webm;base64,RAW")
        assert coerce_media_url_or_data_uri(artifact, kind=self.kind) == "data:video/webm;base64,RAW"

    def test_strings_are_stripped(self) -> None:
        assert (
            coerce_media_url_or_data_uri("  https://example.com/foo.mp4  ", kind=self.kind)
            == "https://example.com/foo.mp4"
        )

    def test_artifact_with_no_known_attrs_returns_none(self) -> None:
        assert coerce_media_url_or_data_uri(SimpleNamespace(), kind=self.kind) is None


# ---------------------------------------------------------------------------
# Serialized-dict path
# ---------------------------------------------------------------------------


class TestCoerceImageDictPath:
    """Serialized image artifact dicts must round-trip cleanly for any caller."""

    def test_dict_with_image_url_artifact_type_passes_value_through(self) -> None:
        result = coerce_media_url_or_data_uri(
            {"type": "ImageUrlArtifact", "value": "https://example.com/foo.png"}, kind="image"
        )
        assert result == "https://example.com/foo.png"

    def test_dict_with_image_url_artifact_type_passes_macro_path_through(self) -> None:
        result = coerce_media_url_or_data_uri({"type": "ImageUrlArtifact", "value": "{inputs}/foo.png"}, kind="image")
        assert result == "{inputs}/foo.png"

    def test_dict_with_image_url_artifact_type_passes_runway_uri_through(self) -> None:
        result = coerce_media_url_or_data_uri(
            {"type": "ImageUrlArtifact", "value": "runway://dataset/abc"}, kind="image"
        )
        assert result == "runway://dataset/abc"

    def test_dict_with_image_url_artifact_type_strips_whitespace(self) -> None:
        result = coerce_media_url_or_data_uri(
            {"type": "ImageUrlArtifact", "value": "  {inputs}/foo.png  "}, kind="image"
        )
        assert result == "{inputs}/foo.png"

    def test_dict_with_url_value_passes_through_regardless_of_type(self) -> None:
        result = coerce_media_url_or_data_uri(
            {"type": "SomethingElse", "value": "https://example.com/foo.png"}, kind="image"
        )
        assert result == "https://example.com/foo.png"

    def test_dict_with_data_uri_value_passes_through(self) -> None:
        result = coerce_media_url_or_data_uri(
            {"type": "ImageArtifact", "value": "data:image/png;base64,RAW"}, kind="image"
        )
        assert result == "data:image/png;base64,RAW"

    def test_dict_with_image_artifact_wraps_with_format(self) -> None:
        result = coerce_media_url_or_data_uri({"type": "ImageArtifact", "value": "RAW", "format": "jpeg"}, kind="image")
        assert result == "data:image/jpeg;base64,RAW"

    def test_dict_with_image_artifact_defaults_to_png_format(self) -> None:
        result = coerce_media_url_or_data_uri({"type": "ImageArtifact", "value": "RAW"}, kind="image")
        assert result == "data:image/png;base64,RAW"

    def test_dict_with_image_artifact_lowercases_format(self) -> None:
        result = coerce_media_url_or_data_uri({"type": "ImageArtifact", "value": "RAW", "format": "JPEG"}, kind="image")
        assert result == "data:image/jpeg;base64,RAW"

    def test_dict_with_unknown_type_and_macro_path_passes_through(self) -> None:
        result = coerce_media_url_or_data_uri({"type": "RandomShape", "value": "{inputs}/foo.png"}, kind="image")
        assert result == "{inputs}/foo.png"

    def test_dict_with_unknown_type_and_filesystem_path_passes_through(self) -> None:
        result = coerce_media_url_or_data_uri({"type": "RandomShape", "value": "/tmp/foo.png"}, kind="image")
        assert result == "/tmp/foo.png"

    def test_dict_with_no_type_and_macro_path_passes_through(self) -> None:
        result = coerce_media_url_or_data_uri({"value": "{inputs}/foo.png"}, kind="image")
        assert result == "{inputs}/foo.png"

    def test_dict_with_empty_value_returns_none(self) -> None:
        assert coerce_media_url_or_data_uri({"type": "ImageArtifact", "value": ""}, kind="image") is None
        assert coerce_media_url_or_data_uri({"type": "ImageArtifact", "value": "   "}, kind="image") is None

    def test_dict_with_non_string_value_returns_none(self) -> None:
        assert coerce_media_url_or_data_uri({"type": "ImageArtifact", "value": 123}, kind="image") is None
        assert coerce_media_url_or_data_uri({"type": "ImageArtifact", "value": None}, kind="image") is None

    def test_dict_with_no_value_key_returns_none(self) -> None:
        assert coerce_media_url_or_data_uri({"type": "ImageArtifact"}, kind="image") is None

    def test_dict_falls_back_to_url_key_when_value_is_missing(self) -> None:
        result = coerce_media_url_or_data_uri({"url": "https://example.com/foo.png"}, kind="image")
        assert result == "https://example.com/foo.png"

    def test_dict_falls_back_to_url_key_when_value_is_empty_string(self) -> None:
        result = coerce_media_url_or_data_uri({"value": "", "url": "https://example.com/foo.png"}, kind="image")
        assert result == "https://example.com/foo.png"

    def test_dict_value_takes_precedence_over_url(self) -> None:
        result = coerce_media_url_or_data_uri(
            {"value": "{inputs}/foo.png", "url": "https://example.com/unused.png"}, kind="image"
        )
        assert result == "{inputs}/foo.png"

    def test_object_with_to_dict_routes_through_dict_path(self) -> None:
        class Fake:
            def to_dict(self) -> dict:
                return {"type": "ImageUrlArtifact", "value": "{inputs}/foo.png"}

        assert coerce_media_url_or_data_uri(Fake(), kind="image") == "{inputs}/foo.png"

    def test_object_to_dict_returning_non_dict_falls_back_to_attrs(self) -> None:
        class Fake:
            def to_dict(self) -> str:
                return "not-a-dict"

            value = "{inputs}/fallback.png"
            base64 = None

        assert coerce_media_url_or_data_uri(Fake(), kind="image") == "{inputs}/fallback.png"

    def test_object_with_failing_to_dict_returns_none_safely(self) -> None:
        class Fake:
            def to_dict(self) -> dict:
                raise RuntimeError("boom")

        assert coerce_media_url_or_data_uri(Fake(), kind="image") is None

    def test_plain_string_macro_path_passes_through(self) -> None:
        assert coerce_media_url_or_data_uri("{inputs}/foo.png", kind="image") == "{inputs}/foo.png"


# ---------------------------------------------------------------------------
# prepare_media_data_uri
# ---------------------------------------------------------------------------


class TestPrepareMediaDataUri:
    """Verifies the prepare wrapper threads coerced values into ``File()``."""

    def test_macro_path_string_is_handed_to_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, str] = {}

        from griptape_nodes.files import file as file_module

        original_init = file_module.File.__init__

        def capture_init(self: file_module.File, path: str, *args: Any, **kwargs: Any) -> None:
            captured["path"] = path
            original_init(self, path, *args, **kwargs)

        def fake_read(self: file_module.File, fallback_mime: str = "application/octet-stream") -> str:
            return "data:video/mp4;base64,RESOLVED"

        monkeypatch.setattr(file_module.File, "__init__", capture_init)
        monkeypatch.setattr(file_module.File, "read_data_uri", fake_read)

        result = prepare_media_data_uri("{inputs}/clip.mp4", kind="video")
        assert result == "data:video/mp4;base64,RESOLVED"
        assert captured["path"] == "{inputs}/clip.mp4"

    def test_data_uri_input_is_returned_without_calling_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from griptape_nodes.files import file as file_module

        def fail_read(self: file_module.File, fallback_mime: str = "application/octet-stream") -> str:
            raise AssertionError("File.read_data_uri must not be called for data: URIs")

        monkeypatch.setattr(file_module.File, "read_data_uri", fail_read)

        result = prepare_media_data_uri("data:video/mp4;base64,ALREADY_BASE64", kind="video")
        assert result == "data:video/mp4;base64,ALREADY_BASE64"

    @pytest.mark.parametrize("scheme", DEFAULT_PASS_THROUGH_SCHEMES)
    def test_default_pass_through_schemes_skip_file(self, monkeypatch: pytest.MonkeyPatch, scheme: str) -> None:
        from griptape_nodes.files import file as file_module

        def fail_read(self: file_module.File, fallback_mime: str = "application/octet-stream") -> str:
            raise AssertionError(f"File.read_data_uri must not be called for {scheme} URIs")

        monkeypatch.setattr(file_module.File, "read_data_uri", fail_read)

        url = f"{scheme}example.com/clip.mp4" if scheme == "https://" else f"{scheme}dataset/abc"
        assert prepare_media_data_uri(url, kind="video") == url

    def test_pass_through_schemes_can_be_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty ``pass_through_schemes`` forces every non-data: URI through ``File``."""
        captured: dict[str, str] = {}

        from griptape_nodes.files import file as file_module

        original_init = file_module.File.__init__

        def capture_init(self: file_module.File, path: str, *args: Any, **kwargs: Any) -> None:
            captured["path"] = path
            original_init(self, path, *args, **kwargs)

        def fake_read(self: file_module.File, fallback_mime: str = "application/octet-stream") -> str:
            return "data:video/mp4;base64,DOWNLOADED"

        monkeypatch.setattr(file_module.File, "__init__", capture_init)
        monkeypatch.setattr(file_module.File, "read_data_uri", fake_read)

        result = prepare_media_data_uri("https://example.com/clip.mp4", kind="video", pass_through_schemes=())
        assert result == "data:video/mp4;base64,DOWNLOADED"
        assert captured["path"] == "https://example.com/clip.mp4"

    def test_returns_none_for_falsy_input(self) -> None:
        assert prepare_media_data_uri(None, kind="video") is None
        assert prepare_media_data_uri("", kind="video") is None
        assert prepare_media_data_uri(0, kind="image") is None

    def test_returns_none_when_helper_resolves_to_none(self) -> None:
        assert prepare_media_data_uri(SimpleNamespace(), kind="video") is None

    def test_file_load_error_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from griptape_nodes.files import file as file_module
        from griptape_nodes.retained_mode.events.os_events import FileIOFailureReason

        def fail_read(self: file_module.File, fallback_mime: str = "application/octet-stream") -> str:
            raise file_module.FileLoadError(FileIOFailureReason.FILE_NOT_FOUND, "nope")

        monkeypatch.setattr(file_module.File, "read_data_uri", fail_read)

        assert prepare_media_data_uri("{inputs}/missing.mp4", kind="video") is None

    def test_fallback_mime_override_is_passed_to_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, str] = {}

        from griptape_nodes.files import file as file_module

        def fake_read(self: file_module.File, fallback_mime: str = "application/octet-stream") -> str:
            captured["fallback_mime"] = fallback_mime
            return "data:image/jpeg;base64,RESOLVED"

        monkeypatch.setattr(file_module.File, "read_data_uri", fake_read)

        prepare_media_data_uri("{inputs}/foo.jpg", kind="image", fallback_mime="image/jpeg")
        assert captured["fallback_mime"] == "image/jpeg"

    def test_default_fallback_mime_matches_kind(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, str] = {}

        from griptape_nodes.files import file as file_module

        def fake_read(self: file_module.File, fallback_mime: str = "application/octet-stream") -> str:
            captured["fallback_mime"] = fallback_mime
            return "data:image/png;base64,RESOLVED"

        monkeypatch.setattr(file_module.File, "read_data_uri", fake_read)

        prepare_media_data_uri("{inputs}/foo.png", kind="image")
        assert captured["fallback_mime"] == "image/png"
