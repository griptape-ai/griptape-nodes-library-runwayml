"""Tests for the Runway API drift guard.

A drift guard that cannot fail is worse than no guard, because it reads as proof the
library is current. These tests drive `check_model` against synthetic specs so each
kind of drift is known to be caught offline, without touching the network.
"""

from __future__ import annotations

from typing import Any

import pytest
from api_surface import MODELS, RETIRED_MODELS, ModelSpec, get_model, model_choices
from check_api_drift import Spec, check_model, check_retired_still_dead

ENDPOINT = "image_to_video"
MODEL_ID = "test_model"


def build_spec(properties: dict[str, Any], required: list[str], endpoint: str = ENDPOINT) -> Spec:
    """Wrap one request-body variant in the minimum spec shape `Spec` understands."""
    return Spec(
        {
            "info": {"version": "2024-11-06"},
            "paths": {
                f"/v1/{endpoint}": {
                    "post": {
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "oneOf": [
                                            {
                                                "properties": {"model": {"const": MODEL_ID}, **properties},
                                                "required": required,
                                            }
                                        ]
                                    }
                                }
                            }
                        }
                    }
                }
            },
        }
    )


@pytest.fixture
def model() -> ModelSpec:
    return ModelSpec(
        model_id=MODEL_ID,
        endpoint=ENDPOINT,
        ratios=("1280:720", "720:1280"),
        duration_min=2,
        duration_max=10,
        required_fields=frozenset({"model", "promptImage"}),
        payload_fields=frozenset({"model"}),
    )


@pytest.fixture
def matching_properties() -> dict[str, Any]:
    return {
        "ratio": {"enum": ["1280:720", "720:1280"]},
        "duration": {"type": "integer", "minimum": 2, "maximum": 10},
    }


def test_no_drift_when_library_matches_spec(model: ModelSpec, matching_properties: dict[str, Any]) -> None:
    spec = build_spec(matching_properties, required=["model", "promptImage"])
    assert check_model(spec, model) == []


def test_ratio_order_is_not_drift(model: ModelSpec) -> None:
    """Dropdown ordering is cosmetic, so only membership should be compared."""
    spec = build_spec(
        {"ratio": {"enum": ["720:1280", "1280:720"]}, "duration": {"minimum": 2, "maximum": 10}},
        required=["model", "promptImage"],
    )
    assert check_model(spec, model) == []


def test_model_removed_from_endpoint_is_reported(model: ModelSpec) -> None:
    spec = build_spec({"model": {"const": "some_other_model"}}, required=["model"])
    problems = check_model(spec, model)
    assert len(problems) == 1
    assert "GONE from the spec" in problems[0]


def test_endpoint_absent_entirely_is_reported(model: ModelSpec) -> None:
    spec = build_spec({}, required=[], endpoint="a_different_endpoint")
    problems = check_model(spec, model)
    assert "GONE from the spec" in problems[0]
    assert "<endpoint absent>" in problems[0]


def test_value_retired_upstream_is_reported(model: ModelSpec) -> None:
    """The dangerous direction: the library still offers something Runway rejects."""
    spec = build_spec(
        {"ratio": {"enum": ["1280:720"]}, "duration": {"minimum": 2, "maximum": 10}},
        required=["model", "promptImage"],
    )
    problems = check_model(spec, model)
    assert len(problems) == 1
    assert "REMOVED upstream" in problems[0]
    assert "720:1280" in problems[0]


def test_new_upstream_value_is_reported(model: ModelSpec) -> None:
    spec = build_spec(
        {
            "ratio": {"enum": ["1280:720", "720:1280", "960:960"]},
            "duration": {"minimum": 2, "maximum": 10},
        },
        required=["model", "promptImage"],
    )
    problems = check_model(spec, model)
    assert len(problems) == 1
    assert "not offered by this library" in problems[0]
    assert "960:960" in problems[0]


def test_widened_duration_bound_is_reported(model: ModelSpec) -> None:
    """Silent drift: the node keeps working while withholding newly allowed values."""
    spec = build_spec(
        {"ratio": {"enum": ["1280:720", "720:1280"]}, "duration": {"minimum": 2, "maximum": 15}},
        required=["model", "promptImage"],
    )
    problems = check_model(spec, model)
    assert len(problems) == 1
    assert "duration_max: library=10 spec=15" in problems[0]


def test_duration_narrowed_to_enum_is_reported(model: ModelSpec) -> None:
    spec = build_spec(
        {
            "ratio": {"enum": ["1280:720", "720:1280"]},
            "duration": {"enum": [5, 10], "minimum": 2, "maximum": 10},
        },
        required=["model", "promptImage"],
    )
    problems = check_model(spec, model)
    assert any("constrains it to an enum" in p for p in problems)


def test_newly_required_field_is_reported(model: ModelSpec, matching_properties: dict[str, Any]) -> None:
    spec = build_spec(matching_properties, required=["model", "promptImage", "promptText"])
    problems = check_model(spec, model)
    assert len(problems) == 1
    assert "now required upstream: ['promptText']" in problems[0]


def test_field_no_longer_required_is_reported(model: ModelSpec, matching_properties: dict[str, Any]) -> None:
    spec = build_spec(matching_properties, required=["model"])
    problems = check_model(spec, model)
    assert "no longer required upstream: ['promptImage']" in problems[0]


def test_declared_field_missing_from_spec_is_reported() -> None:
    model = ModelSpec(
        model_id=MODEL_ID,
        endpoint=ENDPOINT,
        output_formats=("mp4", "prores"),
        required_fields=frozenset({"model"}),
        payload_fields=frozenset({"model"}),
    )
    spec = build_spec({}, required=["model"])
    problems = check_model(spec, model)
    assert any("no such field" in p for p in problems)


def test_maxitems_drift_is_reported() -> None:
    model = ModelSpec(
        model_id=MODEL_ID,
        endpoint=ENDPOINT,
        max_reference_images=3,
        required_fields=frozenset({"model"}),
        payload_fields=frozenset({"model"}),
    )
    spec = build_spec({"referenceImages": {"type": "array", "maxItems": 10}}, required=["model"])
    problems = check_model(spec, model)
    assert "referenceImages maxItems: library=3 spec=10" in problems[0]


def test_seed_bound_drift_is_reported() -> None:
    model = ModelSpec(
        model_id=MODEL_ID,
        endpoint=ENDPOINT,
        required_fields=frozenset({"model"}),
        payload_fields=frozenset({"model"}),
    )
    spec = build_spec({"seed": {"minimum": 1, "maximum": 999}}, required=["model"])
    problems = check_model(spec, model)
    assert any("seed minimum" in p for p in problems)
    assert any("seed maximum" in p for p in problems)


def test_a_payload_key_the_model_does_not_accept_is_reported() -> None:
    """The class of bug that shipped: a retired model's field carried into its replacement."""
    model = ModelSpec(
        model_id=MODEL_ID,
        endpoint=ENDPOINT,
        required_fields=frozenset({"model"}),
        payload_fields=frozenset({"model", "references"}),
    )
    spec = build_spec({}, required=["model"])
    problems = check_model(spec, model)
    assert any("references" in p and "not accepted upstream" in p for p in problems)


def test_strict_models_are_called_out_as_rejecting_unknown_keys() -> None:
    model = ModelSpec(
        model_id=MODEL_ID,
        endpoint=ENDPOINT,
        required_fields=frozenset({"model"}),
        payload_fields=frozenset({"model", "references"}),
    )
    spec = Spec(
        {
            "paths": {
                f"/v1/{ENDPOINT}": {
                    "post": {
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "oneOf": [
                                            {
                                                "properties": {"model": {"const": MODEL_ID}},
                                                "required": ["model"],
                                                "additionalProperties": False,
                                            }
                                        ]
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    )
    problems = check_model(spec, model)
    assert any("rejects unknown keys" in p for p in problems)


def test_undeclared_payload_fields_are_reported_as_unchecked() -> None:
    """Silence here would mean the drift check validates nothing about what a node sends."""
    model = ModelSpec(model_id=MODEL_ID, endpoint=ENDPOINT, required_fields=frozenset({"model"}))
    spec = build_spec({}, required=["model"])
    problems = check_model(spec, model)
    assert any("payload_fields: not declared" in p for p in problems)


def test_every_supported_model_declares_its_payload_fields() -> None:
    assert [m.model_id for m in MODELS if not m.payload_fields] == []


def test_resurrected_retired_model_is_reported() -> None:
    """If Runway ever re-lists a retired id, the migration hint becomes a lie."""
    spec = build_spec({"model": {"const": "gen4_aleph"}}, required=["model"])
    problems = check_retired_still_dead(spec)
    assert len(problems) == 1
    assert "gen4_aleph" in problems[0]


def test_billing_endpoints_do_not_resurrect_retired_models() -> None:
    """Retired ids legitimately persist in billing enums; that must not count as alive."""
    spec = Spec(
        {
            "paths": {
                "/v1/organization/usage": {
                    "post": {
                        "requestBody": {
                            "content": {
                                "application/json": {"schema": {"properties": {"model": {"enum": ["gen4_aleph"]}}}}
                            }
                        }
                    }
                }
            }
        }
    )
    assert check_retired_still_dead(spec) == []


class TestApiSurfaceLookups:
    def test_get_model_returns_the_endpoint_specific_spec(self) -> None:
        """`gen4.5` differs per endpoint, so lookup must be keyed on both."""
        image_to_video = get_model("gen4.5", "image_to_video")
        text_to_video = get_model("gen4.5", "text_to_video")
        assert len(image_to_video.ratios) > len(text_to_video.ratios)

    def test_retired_model_raises_with_migration_hint(self) -> None:
        with pytest.raises(ValueError, match="aleph2"):
            get_model("gen4_aleph", "video_to_video")

    def test_unknown_model_lists_supported_models(self) -> None:
        with pytest.raises(ValueError, match="gen4_turbo"):
            get_model("not_a_runway_model", "image_to_video")

    def test_model_choices_are_scoped_to_the_endpoint(self) -> None:
        assert "gen4_aleph" not in model_choices("video_to_video")
        assert model_choices("video_to_video") == ["aleph2"]

    def test_prompt_text_required_derives_from_required_fields(self) -> None:
        assert get_model("gen4.5", "text_to_video").prompt_text_required
        assert not get_model("aleph2", "video_to_video").prompt_text_required

    def test_every_offered_output_format_has_a_known_container(self) -> None:
        """`_sync_output_extension` silently no-ops on an unmapped format.

        The drift check invites this: it reports a new upstream `outputFormat` value, and the
        natural response is to add it to `output_formats` alone. A sequence format added that
        way would reach a video dropdown and hand a zip downstream as a video.
        """
        from api_surface import OUTPUT_FORMAT_EXTENSIONS

        offered = {fmt for spec in MODELS for fmt in spec.output_formats}
        assert offered
        assert offered <= set(OUTPUT_FORMAT_EXTENSIONS)

    def test_every_frame_sequence_format_is_mapped_to_a_zip(self) -> None:
        from api_surface import FRAME_SEQUENCE_OUTPUT_FORMATS, OUTPUT_FORMAT_EXTENSIONS

        for fmt in FRAME_SEQUENCE_OUTPUT_FORMATS:
            assert OUTPUT_FORMAT_EXTENSIONS[fmt] == ".zip"

    def test_retired_models_are_absent_from_every_endpoint(self) -> None:
        endpoints = {spec.endpoint for spec in MODELS}
        offered = {model_id for endpoint in endpoints for model_id in model_choices(endpoint)}
        assert offered.isdisjoint(RETIRED_MODELS)

    @pytest.mark.parametrize(
        ("endpoint", "expected_first"),
        [("image_to_video", "gen4_turbo"), ("text_to_image", "gen4_image")],
    )
    def test_first_choice_is_the_cheaper_migration_target(self, endpoint: str, expected_first: str) -> None:
        """An `Options` trait snaps a retired value to choices[0].

        That makes ordering a pricing decision: a workflow saved against a retired model
        silently lands on whichever model leads the list, so it must not be the priciest.
        """
        assert model_choices(endpoint)[0] == expected_first
