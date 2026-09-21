"""Tests for the reference-image path: the artifact, the node that builds it, and the node
that resolves it into a request.

This path spends money when it goes wrong -- a reference that is silently dropped still bills a
full generation that ignored it -- and it had the thinnest coverage in the library.
"""

from __future__ import annotations

import asyncio
import base64
import io
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from artifacts import (
    MAX_REFERENCE_ASPECT_RATIO,
    MIN_REFERENCE_ASPECT_RATIO,
    ReferenceImageArtifact,
    unpack_reference_image,
)
from griptape.artifacts import ImageUrlArtifact


def png_bytes(width: int, height: int) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (128, 128, 128)).save(buffer, format="PNG")
    return buffer.getvalue()


def png_data_uri(width: int = 64, height: int = 64) -> str:
    return f"data:image/png;base64,{base64.b64encode(png_bytes(width, height)).decode()}"


class TestReferenceImageArtifact:
    def test_image_and_tag_read_from_value(self) -> None:
        """One copy of the data, so a deserialized artifact cannot disagree with itself."""
        artifact = ReferenceImageArtifact(image="https://e.com/i.png", tag="eiffel_tower")

        assert artifact.value == {"image": "https://e.com/i.png", "tag": "eiffel_tower"}
        assert artifact.image == "https://e.com/i.png"
        assert artifact.tag == "eiffel_tower"

    def test_a_mutated_value_is_reflected_by_the_properties(self) -> None:
        artifact = ReferenceImageArtifact(image="a", tag="first_tag")
        artifact.value["tag"] = "second_tag"

        assert artifact.tag == "second_tag"

    @pytest.mark.parametrize(("given", "expected"), [("  spaced  ", "spaced"), ("", ""), ("   ", "")])
    def test_tag_is_stripped(self, given: str, expected: str) -> None:
        assert ReferenceImageArtifact(image="a", tag=given).tag == expected

    def test_an_explicit_name_is_kept(self) -> None:
        assert ReferenceImageArtifact(image="a", tag="t", name="reference_t").name == "reference_t"

    def test_to_text_names_the_wrapped_type_and_truncates(self) -> None:
        artifact = ReferenceImageArtifact(image=ImageUrlArtifact("https://e.com/" + "x" * 200), tag="t")
        text = artifact.to_text()

        assert "tag='t'" in text
        assert "ImageUrlArtifact" in text
        assert "..." in text
        assert str(artifact) == text
        assert repr(artifact) == text


class TestUnpackReferenceImage:
    def test_from_an_artifact(self) -> None:
        artifact = ReferenceImageArtifact(image="https://e.com/i.png", tag="t")
        assert unpack_reference_image(artifact) == ("https://e.com/i.png", "t")

    def test_from_a_plain_dict(self) -> None:
        """A saved workflow round-trips the pairing as a dict."""
        assert unpack_reference_image({"image": "a", "tag": "t"}) == ("a", "t")

    def test_from_an_artifact_whose_value_is_the_dict(self) -> None:
        """What `BaseArtifact` deserialization produces."""
        from types import SimpleNamespace

        assert unpack_reference_image(SimpleNamespace(value={"image": "a", "tag": "t"})) == ("a", "t")

    @pytest.mark.parametrize("candidate", [None, "just a string", 42, {}, {"tag": "t"}, []])
    def test_anything_else_yields_nothing(self, candidate: Any) -> None:
        assert unpack_reference_image(candidate) == (None, "")

    def test_a_missing_tag_becomes_an_empty_string(self) -> None:
        assert unpack_reference_image({"image": "a"}) == ("a", "")


class TestCreateReferenceImageNode:
    @staticmethod
    def build() -> Any:
        from create_reference_image import RunwayML_CreateReferenceImage

        return RunwayML_CreateReferenceImage(name="Runway Create Reference Image")

    def test_no_image_is_rejected(self) -> None:
        node = self.build()
        with pytest.raises(ValueError, match="no image is set"):
            asyncio.run(node.aprocess())

    def test_a_square_image_produces_a_tagged_artifact(self) -> None:
        node = self.build()
        node.set_parameter_value("image", ImageUrlArtifact("https://e.com/i.png"))
        node.set_parameter_value("tag", "eiffel_tower")

        with patch("create_reference_image.File") as mock_file:
            mock_file.return_value.read_bytes.return_value = png_bytes(512, 512)
            asyncio.run(node.aprocess())

        artifact = node.parameter_output_values["reference_image"]
        assert artifact.tag == "eiffel_tower"
        assert artifact.name == "reference_eiffel_tower"

    @pytest.mark.parametrize(("width", "height"), [(3000, 600), (600, 3000)])
    def test_an_out_of_range_aspect_ratio_is_rejected_here(self, width: int, height: int) -> None:
        """Rejected at this node so the error names the image, not the generation node."""
        node = self.build()
        node.set_parameter_value("image", ImageUrlArtifact("https://e.com/pano.png"))

        with patch("create_reference_image.File") as mock_file:
            mock_file.return_value.read_bytes.return_value = png_bytes(width, height)
            with pytest.raises(ValueError, match="aspect ratio"):
                asyncio.run(node.aprocess())

    @pytest.mark.parametrize("ratio", [MIN_REFERENCE_ASPECT_RATIO, 1.0, MAX_REFERENCE_ASPECT_RATIO])
    def test_the_accepted_range_is_inclusive(self, ratio: float) -> None:
        node = self.build()
        node.set_parameter_value("image", {"meta": {"aspectRatio": ratio}})
        asyncio.run(node.aprocess())

        assert node.parameter_output_values["reference_image"] is not None

    @pytest.mark.parametrize("meta", [None, {}, {"aspectRatio": "not a number"}, "not a dict"])
    def test_an_unusable_meta_block_does_not_crash(self, meta: Any) -> None:
        """`meta` can be present but null, and aspectRatio can be a string."""
        node = self.build()
        node.set_parameter_value("image", {"meta": meta})
        asyncio.run(node.aprocess())

        assert node.parameter_output_values["reference_image"] is not None

    def test_an_unmeasurable_image_passes_through(self) -> None:
        """Measuring is a courtesy check; failing to measure is not a reason to reject."""
        from griptape_nodes.files.file import FileLoadError
        from griptape_nodes.retained_mode.events.os_events import FileIOFailureReason

        node = self.build()
        node.set_parameter_value("image", ImageUrlArtifact("https://e.com/gone.png"))

        with patch("create_reference_image.File") as mock_file:
            mock_file.return_value.read_bytes.side_effect = FileLoadError(
                FileIOFailureReason.FILE_NOT_FOUND, "no such file"
            )
            asyncio.run(node.aprocess())

        assert node.parameter_output_values["reference_image"] is not None

    def test_a_generated_tag_is_stable_across_processes(self) -> None:
        """A digest, not `hash()`: a salted hash would change on every engine restart and a
        prompt written against the tag would silently stop matching."""
        first, second = self.build(), self.build()
        for node in (first, second):
            node.set_parameter_value("image", {"meta": {"aspectRatio": 1.0}})
            asyncio.run(node.aprocess())

        tag = first.parameter_output_values["reference_image"].tag
        assert tag == second.parameter_output_values["reference_image"].tag
        assert tag.startswith("ref_")

    def test_a_bare_path_is_measurable(self) -> None:
        """The parameter declares `str`, so a file-browser path must reach the measurement."""
        node = self.build()
        node.set_parameter_value("image", "{inputs}/pano.png")

        with patch("create_reference_image.File") as mock_file:
            mock_file.return_value.read_bytes.return_value = png_bytes(3000, 600)
            with pytest.raises(ValueError, match="aspect ratio"):
                asyncio.run(node.aprocess())

        assert mock_file.call_args.args[0] == "{inputs}/pano.png"


class TestTextToImageReferenceResolution:
    @staticmethod
    def build(tag: str = "eiffel_tower", image: Any = "data:image/png;base64,AAAA") -> Any:
        from text_to_image import RunwayML_TextToImage

        with patch("runway_node.get_api_key", return_value="k"):
            node = RunwayML_TextToImage(name="t2i")
        node.set_parameter_value("prompt_text", "a scene")
        child = node.get_parameter_by_name("reference_images").add_child_parameter()
        node.set_parameter_value(child.name, ReferenceImageArtifact(image=image, tag=tag))
        return node

    def test_a_supported_data_uri_passes_through(self) -> None:
        node = self.build(image=png_data_uri())
        references = node.build_payload()["referenceImages"]

        assert len(references) == 1
        assert references[0]["tag"] == "eiffel_tower"
        assert references[0]["uri"].startswith("data:image/png;base64,")

    def test_an_https_reference_is_sent_as_is(self) -> None:
        """HTTPS is not subject to the inline size cap, so it must not be downloaded."""
        node = self.build(image=ImageUrlArtifact("https://e.com/i.png"))
        assert node.build_payload()["referenceImages"][0]["uri"] == "https://e.com/i.png"

    def test_an_untagged_reference_gets_a_positional_tag(self) -> None:
        """A tag is what makes the image addressable as @tag, so one is always sent."""
        node = self.build(tag="", image=png_data_uri())
        assert node.build_payload()["referenceImages"][0]["tag"] == "ref_1"

    def test_an_unsupported_format_is_converted_to_png(self) -> None:
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("P", (64, 64)).save(buffer, format="GIF")
        gif = f"data:image/gif;base64,{base64.b64encode(buffer.getvalue()).decode()}"

        node = self.build(image=gif)
        assert node.build_payload()["referenceImages"][0]["uri"].startswith("data:image/png;base64,")

    def test_a_data_uri_that_is_not_base64_is_reported_as_a_format_problem(self) -> None:
        """Inline SVG arrives verbatim; unpacking it would surface as an unpacking error."""
        node = self.build(image="data:image/svg+xml,<svg/>")
        with pytest.raises(ValueError, match="not in a format RunwayML can read"):
            node.build_payload()

    def test_undecodable_image_bytes_are_reported_as_a_format_problem(self) -> None:
        node = self.build(image="data:image/tiff;base64,bm90YW5pbWFnZQ==")
        with pytest.raises(ValueError, match="not in a format RunwayML can read"):
            node.build_payload()

    def test_an_oversized_reference_is_refused_before_upload(self) -> None:
        from api_surface import MAX_DATA_URI_BYTES

        oversized = "data:image/png;base64," + "A" * (MAX_DATA_URI_BYTES + 1)
        node = self.build(image=oversized)
        with pytest.raises(ValueError, match="too large"):
            node.build_payload()

    def test_an_unreadable_reference_is_reported(self) -> None:
        node = self.build(image=ImageUrlArtifact("https://e.com/i.png"))
        with patch("text_to_image.prepare_media_data_uri", return_value=None):
            with pytest.raises(ValueError, match="could not be read"):
                node.build_payload()

    def test_a_non_reference_artifact_is_reported(self) -> None:
        node = self.build()
        child = node.get_parameter_by_name("reference_images").add_child_parameter()
        node.set_parameter_value(child.name, "not a reference image")

        with pytest.raises(ValueError, match="is not a reference image"):
            node.build_payload()

    def test_references_are_omitted_when_none_are_connected(self) -> None:
        from text_to_image import RunwayML_TextToImage

        with patch("runway_node.get_api_key", return_value="k"):
            node = RunwayML_TextToImage(name="t2i")
        node.set_parameter_value("prompt_text", "a scene")

        assert "referenceImages" not in node.build_payload()

    def test_the_saved_output_is_an_image_artifact(self) -> None:
        from types import SimpleNamespace

        node = self.build(image=png_data_uri())
        written = SimpleNamespace(location="{outputs}/output_v001.png")
        destination = SimpleNamespace(awrite_bytes=AsyncMock(return_value=written))

        with (
            patch("runway_node.File") as mock_file,
            patch.object(node._output_file, "build_file", return_value=destination),
        ):
            mock_file.return_value.aread_bytes = AsyncMock(return_value=b"png-bytes")
            artifact = asyncio.run(node._save_output("https://runway.example/out.png"))

        assert type(artifact).__name__ == "ImageUrlArtifact"
        assert artifact.value == "{outputs}/output_v001.png"
