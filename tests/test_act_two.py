"""Node-level checks for ``RunwayML_ActTwo``.

Coercion of arbitrary input shapes is exercised in
``tests/unit/media/test_coercion.py``; this module only covers the wiring
between Act Two's parameters and the shared helper.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest
from act_two import RunwayML_ActTwo
from griptape.artifacts import ImageUrlArtifact, VideoUrlArtifact


@pytest.fixture
def node() -> RunwayML_ActTwo:
    return RunwayML_ActTwo(name="test")


@pytest.fixture(autouse=True)
def _stub_secret() -> Iterator[None]:
    with patch("runway_node.get_api_key", return_value="fake-key"):
        yield


def _set_minimal_required(node: RunwayML_ActTwo) -> None:
    node.set_parameter_value("character_type", "image")
    node.set_parameter_value("ratio", "1280:720")
    node.set_parameter_value("expression_intensity", 3)
    node.set_parameter_value("model", "act_two")


def test_https_image_and_video_validate_without_loading(node: RunwayML_ActTwo) -> None:
    _set_minimal_required(node)
    node.set_parameter_value("character_image", "https://example.com/i.png")
    node.set_parameter_value("reference_video", "https://example.com/v.mp4")

    assert node.validate_before_node_run() is None


def test_url_artifact_with_macro_path_resolves_via_file(node: RunwayML_ActTwo) -> None:
    """Regression: ``LoadImage``/``LoadVideo`` outputs use ``{inputs}/...`` macro paths.

    Asserted against ``build_payload`` rather than validation: validation is deliberately a
    presence check, because the engine calls it on its event loop.
    """
    _set_minimal_required(node)
    node.set_parameter_value("character_image", ImageUrlArtifact("{inputs}/image_196.png"))
    node.set_parameter_value("reference_video", VideoUrlArtifact("{inputs}/clip.mp4"))

    with patch("media.coercion.File") as MockFile:
        MockFile.return_value.read_data_uri.side_effect = [
            "data:image/png;base64,IMG",
            "data:video/mp4;base64,VID",
        ]
        payload = node.build_payload()

    assert payload["character"]["uri"] == "data:image/png;base64,IMG"
    assert payload["reference"]["uri"] == "data:video/mp4;base64,VID"
    assert MockFile.call_args_list[0].args[0] == "{inputs}/image_196.png"
    assert MockFile.call_args_list[1].args[0] == "{inputs}/clip.mp4"


def test_missing_character_image_yields_required_error(node: RunwayML_ActTwo) -> None:
    _set_minimal_required(node)
    node.set_parameter_value("character_image", None)
    node.set_parameter_value("reference_video", "https://example.com/v.mp4")

    errors = node.validate_before_node_run()
    assert errors is not None
    assert any("no character image is set" in str(e) for e in errors)


def test_missing_reference_video_yields_required_error(node: RunwayML_ActTwo) -> None:
    _set_minimal_required(node)
    node.set_parameter_value("character_image", "https://example.com/i.png")
    node.set_parameter_value("reference_video", None)

    errors = node.validate_before_node_run()
    assert errors is not None
    assert any("no reference video" in str(e) for e in errors)
