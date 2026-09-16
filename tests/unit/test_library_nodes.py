"""Integrity checks across every node the library declares.

The manifest is the contract the engine loads, so a node that is renamed, moved, or given
an endpoint that no longer exists must fail here rather than at library-load time in the
editor.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
from api_surface import MODELS
from griptape_nodes.exe_types.node_types import BaseNode
from runway_node import RunwayTaskNode

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "runwayml" / "griptape_nodes_library.json"
MANIFEST: dict[str, Any] = json.loads(MANIFEST_PATH.read_text())
NODE_ENTRIES: list[dict[str, Any]] = MANIFEST["nodes"]
KNOWN_ENDPOINTS = {spec.endpoint for spec in MODELS}


def build_node(entry: dict[str, Any]) -> Any:
    module = importlib.import_module(entry["file_path"].removesuffix(".py"))
    node_class = getattr(module, entry["class_name"])
    with patch("runway_node.get_api_key", return_value="fake-key"):
        return node_class(name=entry["class_name"])


@pytest.mark.parametrize("entry", NODE_ENTRIES, ids=lambda e: e["class_name"])
class TestDeclaredNodes:
    def test_class_is_importable_and_constructs(self, entry: dict[str, Any]) -> None:
        assert isinstance(build_node(entry), BaseNode)

    def test_declared_file_exists(self, entry: dict[str, Any]) -> None:
        assert (MANIFEST_PATH.parent / entry["file_path"]).is_file()

    def test_category_is_declared_in_the_manifest(self, entry: dict[str, Any]) -> None:
        declared = {name for category in MANIFEST["categories"] for name in category}
        assert entry["metadata"]["category"] in declared

    def test_endpoint_is_one_the_api_surface_knows(self, entry: dict[str, Any]) -> None:
        """A node pointed at an endpoint absent from api_surface would bypass the drift check."""
        node = build_node(entry)
        if isinstance(node, RunwayTaskNode):
            assert node.endpoint in KNOWN_ENDPOINTS

    def test_task_nodes_declare_their_output_parameter(self, entry: dict[str, Any]) -> None:
        node = build_node(entry)
        if isinstance(node, RunwayTaskNode):
            assert node.get_parameter_by_name(node.output_parameter_name) is not None
            # aprocess publishes the task id, so it must exist on every task node.
            assert node.get_parameter_by_name("task_id_output") is not None


class TestNewNodePayloads:
    def test_text_to_video_sends_prompt_ratio_and_duration(self) -> None:
        from text_to_video import RunwayML_TextToVideo

        with patch("runway_node.get_api_key", return_value="k"):
            node = RunwayML_TextToVideo(name="t2v")
        node.set_parameter_value("prompt", "a neon alley")
        node.set_parameter_value("seed_control", "fixed")
        node.set_parameter_value("seed", 5)

        payload = node.build_payload()
        assert payload["model"] == "gen4.5"
        assert payload["promptText"] == "a neon alley"
        assert payload["ratio"] == "1280:720"
        assert payload["duration"] == 5
        assert payload["seed"] == 5
        # mp4 is the API default, so it is left out rather than sent redundantly.
        assert "outputFormat" not in payload

    def test_text_to_video_prores_carries_a_profile(self) -> None:
        from text_to_video import RunwayML_TextToVideo

        with patch("runway_node.get_api_key", return_value="k"):
            node = RunwayML_TextToVideo(name="t2v")
        node.set_parameter_value("prompt", "a neon alley")
        node.set_parameter_value("output_format", "prores")

        payload = node.build_payload()
        assert payload["outputFormat"] == "prores"
        assert payload["proresProfile"] == "4444"
        # RunwayML does the encode; the extension has to follow the container it returns.
        assert str(node.get_parameter_value("output_file")).endswith(".mov")

    def test_text_to_video_rejects_an_empty_prompt(self) -> None:
        from text_to_video import RunwayML_TextToVideo

        with patch("runway_node.get_api_key", return_value="k"):
            node = RunwayML_TextToVideo(name="t2v")
        errors = node.validate_before_node_run()
        assert errors is not None
        assert any("prompt is empty" in str(e) for e in errors)

    def test_video_to_hdr_defaults_to_hdr10_without_a_prores_profile(self) -> None:
        from video_to_hdr import RunwayML_VideoToHDR

        with patch("runway_node.get_api_key", return_value="k"):
            node = RunwayML_VideoToHDR(name="hdr")
        node.set_parameter_value("video", "https://example.com/v.mp4")

        payload = node.build_payload()
        assert payload["model"] == "ruby"
        assert payload["outputFormat"] == "hdr10"
        assert "proresProfile" not in payload
        # Ruby takes no seed, so none must be invented for it.
        assert "seed" not in payload

    def test_video_to_hdr_prores_uses_the_hdr_safe_profiles(self) -> None:
        from video_to_hdr import RunwayML_VideoToHDR

        with patch("runway_node.get_api_key", return_value="k"):
            node = RunwayML_VideoToHDR(name="hdr")
        node.set_parameter_value("video", "https://example.com/v.mp4")
        node.set_parameter_value("output_format", "hdr_prores")

        payload = node.build_payload()
        assert payload["proresProfile"] == "422 HQ"
        assert str(node.get_parameter_value("output_file")).endswith(".mov")

    def test_video_to_hdr_requires_a_video(self) -> None:
        from video_to_hdr import RunwayML_VideoToHDR

        with patch("runway_node.get_api_key", return_value="k"):
            node = RunwayML_VideoToHDR(name="hdr")
        errors = node.validate_before_node_run()
        assert errors is not None
        assert any("no input video is set" in str(e) for e in errors)


@pytest.mark.parametrize("entry", NODE_ENTRIES, ids=lambda e: e["class_name"])
def test_every_task_node_rejects_a_retired_model_in_the_editor(entry: dict[str, Any]) -> None:
    """A stale model must be caught before the run, not after it starts.

    `Options` snaps a stale property to choices[0], but `model` accepts an incoming connection
    that bypasses the trait, so the validation hook is the real gate.
    """
    node = build_node(entry)
    if not isinstance(node, RunwayTaskNode):
        pytest.skip("not a task node")

    node.parameter_values["model"] = "gen4_aleph"
    errors = node.validate_before_node_run() or []
    assert any("gen4_aleph" in str(e) for e in errors), (
        f"{entry['class_name']} does not validate its model at editor time"
    )


class TestPayloadFieldsAreDeclared:
    """`api_surface.payload_fields` is what the drift check compares against the live spec.

    If a node sends a key the declaration omits, the drift check validates the wrong set and a
    request that RunwayML rejects passes every offline check. That is exactly how `references`
    (a retired model's field) survived into aleph2 payloads.
    """

    # Values that drive every optional branch a node has, so conditionally-added keys are
    # actually emitted. A minimal payload would not have caught `references`, which was itself
    # only added when a reference image was set.
    MAXIMAL_VALUES: ClassVar[dict[str, dict[str, Any]]] = {
        "text_to_image": {"prompt_text": "x"},
        "text_to_video": {"prompt": "x", "output_format": "prores"},
        "image_to_video": {
            "image": "https://e.com/i.png",
            "prompt": "x",
            "model": "gen4.5",
            "output_format": "prores",
        },
        "video_to_video": {
            "video": "https://e.com/v.mp4",
            "prompt": "x",
            "reference_image": "data:image/png;base64,AAAA",
            "target_aspect_ratio": "16:9",
            "output_format": "prores",
        },
        "video_upscale": {"video": "https://e.com/v.mp4"},
        "video_to_hdr": {"video": "https://e.com/v.mp4", "output_format": "hdr_prores"},
        "character_performance": {
            "character_image": "https://e.com/i.png",
            "reference_video": "https://e.com/v.mp4",
        },
    }

    def test_every_task_node_is_covered(self) -> None:
        """Coverage comes from the manifest, so a new node cannot be silently left unchecked."""
        endpoints = set()
        for entry in NODE_ENTRIES:
            node = build_node(entry)
            if isinstance(node, RunwayTaskNode):
                endpoints.add(node.endpoint)

        uncovered = endpoints - set(self.MAXIMAL_VALUES)
        assert not uncovered, f"no maximal payload declared for: {sorted(uncovered)}"

    @pytest.mark.parametrize("entry", NODE_ENTRIES, ids=lambda e: e["class_name"])
    def test_built_payload_only_uses_declared_fields(self, entry: dict[str, Any]) -> None:
        from api_surface import get_model

        node = build_node(entry)
        if not isinstance(node, RunwayTaskNode):
            pytest.skip("not a task node")

        for name, value in self.MAXIMAL_VALUES[node.endpoint].items():
            node.set_parameter_value(name, value)

        payload = node.build_payload()
        sent = set(payload)
        # Keyed off the model the payload actually names, so flipping a node's DEFAULT_MODEL
        # cannot leave this validating some other model's declared set.
        declared = get_model(payload["model"], node.endpoint).payload_fields
        assert sent <= declared, f"{entry['class_name']} sends undeclared field(s): {sorted(sent - declared)}"

    def test_video_to_video_sends_keyframes_not_the_retired_references_field(self) -> None:
        """aleph2 declares additionalProperties:false and has no `references` property."""
        from video_to_video import RunwayML_VideoToVideo

        with patch("runway_node.get_api_key", return_value="k"):
            node = RunwayML_VideoToVideo(name="v2v")
        node.set_parameter_value("video", "https://e.com/v.mp4")
        node.set_parameter_value("reference_image", "data:image/png;base64,AAAA")

        payload = node.build_payload()
        assert "references" not in payload
        assert payload["keyframes"] == [{"uri": "data:image/png;base64,AAAA", "seconds": 0}]


class TestProResProfileIsOmittedWhenEmpty:
    """`proresProfile` is optional upstream; sending "" manufactures a rejection.

    Reachable because `prores_profile` declares ParameterMode.INPUT, so a connection can feed
    an empty string even though the declared default is a valid tier.
    """

    @pytest.mark.parametrize(
        ("module_name", "class_name", "values"),
        [
            ("text_to_video", "RunwayML_TextToVideo", {"prompt": "x", "output_format": "prores"}),
            (
                "video_to_video",
                "RunwayML_VideoToVideo",
                {"video": "https://e.com/v.mp4", "output_format": "prores"},
            ),
            (
                "video_to_hdr",
                "RunwayML_VideoToHDR",
                {"video": "https://e.com/v.mp4", "output_format": "hdr_prores"},
            ),
        ],
    )
    def test_an_empty_tier_omits_the_key(self, module_name: str, class_name: str, values: dict[str, Any]) -> None:
        module = importlib.import_module(module_name)
        with patch("runway_node.get_api_key", return_value="k"):
            node = getattr(module, class_name)(name="n")
        for name, value in values.items():
            node.set_parameter_value(name, value)
        node.parameter_values["prores_profile"] = ""

        payload = node.build_payload()
        assert payload["outputFormat"] == values["output_format"]
        assert "proresProfile" not in payload

    def test_a_chosen_tier_is_sent_verbatim(self) -> None:
        """Whatever the user picked goes through unjudged; RunwayML rules on the pairing."""
        from text_to_video import RunwayML_TextToVideo

        with patch("runway_node.get_api_key", return_value="k"):
            node = RunwayML_TextToVideo(name="t2v")
        node.set_parameter_value("prompt", "x")
        node.set_parameter_value("output_format", "prores")
        node.set_parameter_value("prores_profile", "422 Proxy")

        assert node.build_payload()["proresProfile"] == "422 Proxy"


class TestProOutputIsResetWhenUnavailable:
    def test_switching_off_a_pro_model_clears_a_stale_output_format(self) -> None:
        """Left stale, `prores` is dropped from the payload and the user is billed for mp4."""
        from image_to_video import RunwayML_ImageToVideo

        with patch("runway_node.get_api_key", return_value="k"):
            node = RunwayML_ImageToVideo(name="i2v")

        node.set_parameter_value("model", "gen4.5")
        node.set_parameter_value("output_format", "prores")
        assert str(node.get_parameter_value("output_file")).endswith(".mov")

        node.set_parameter_value("model", "gen4_turbo")

        assert node.get_parameter_value("output_format") == "mp4"
        assert str(node.get_parameter_value("output_file")).endswith(".mp4")

    def test_switching_model_keeps_a_ratio_the_new_model_still_accepts(self) -> None:
        """Both i2v models share one ratio list, so a reset here is pure loss."""
        from image_to_video import RunwayML_ImageToVideo

        with patch("runway_node.get_api_key", return_value="k"):
            node = RunwayML_ImageToVideo(name="i2v")

        node.set_parameter_value("ratio", "720:1280")
        node.set_parameter_value("model", "gen4.5")

        assert node.get_parameter_value("ratio") == "720:1280"


class TestFrameSequenceFormatsAreNotOffered:
    """Sequence formats need unpacking into a Sequence, which no node does yet.

    Offering one would download a zip and hand it downstream labelled as a video.
    """

    @pytest.mark.parametrize(
        ("module_name", "class_name"),
        [
            ("text_to_video", "RunwayML_TextToVideo"),
            ("image_to_video", "RunwayML_ImageToVideo"),
            ("video_to_video", "RunwayML_VideoToVideo"),
            ("video_to_hdr", "RunwayML_VideoToHDR"),
        ],
    )
    def test_no_zip_format_in_the_dropdown(self, module_name: str, class_name: str) -> None:
        from api_surface import FRAME_SEQUENCE_OUTPUT_FORMATS
        from griptape_nodes.traits.options import Options

        module = importlib.import_module(module_name)
        with patch("runway_node.get_api_key", return_value="k"):
            node = getattr(module, class_name)(name="n")

        parameter = node.get_parameter_by_name("output_format")
        assert parameter is not None
        for trait in parameter.find_elements_by_type(Options):
            assert set(trait.choices).isdisjoint(FRAME_SEQUENCE_OUTPUT_FORMATS)
