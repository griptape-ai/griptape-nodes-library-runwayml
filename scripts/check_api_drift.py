"""Diff `runwayml/api_surface.py` against Runway's published OpenAPI spec.

Runway retires models and widens parameter bounds without touching its dated API
version, so the only way to know this library is still current is to compare it to
the spec. Run via ``make api/drift``. This is deliberately not part of ``make
check``: it needs the network, and a Runway outage must not fail an unrelated CI run.

Exits 1 if the library and the spec disagree.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runwayml"))

from api_surface import (  # noqa: E402
    MAX_PROMPT_LENGTH,
    MODELS,
    RETIRED_MODELS,
    RUNWAY_API_VERSION,
    SEED_MAX,
    SEED_MIN,
    ModelSpec,
)

SPEC_URL = "https://docs.dev.runwayml.com/openapi.json"
REQUEST_TIMEOUT_SECONDS = 60

# Retired identifiers stay listed in Runway's billing enums so historical usage can still
# be reported. Only request schemas say what is generatable, so the walk is scoped to them.
BILLING_PATH_PREFIXES = ("/v1/organization",)


def fetch_spec(url: str = SPEC_URL) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
        return json.load(response)


class Spec:
    """Read-only view over the parts of Runway's spec this library depends on."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw
        self._schemas = raw.get("components", {}).get("schemas", {})

    @property
    def api_version(self) -> str:
        return self._raw.get("info", {}).get("version", "")

    def _deref(self, node: Any) -> Any:
        # $refs can chain, but a fixed bound keeps a malformed spec from hanging the check.
        for _ in range(20):
            if not (isinstance(node, dict) and "$ref" in node):
                return node
            node = self._schemas.get(node["$ref"].rsplit("/", 1)[-1], {})
        return node

    def request_variants(self, endpoint: str) -> list[dict[str, Any]]:
        """Return the request-body variants for `POST /v1/<endpoint>`.

        Runway models each model as its own `oneOf` variant rather than sharing one
        schema, so per-model constraints have to be read off the matching variant.
        """
        path = self._raw.get("paths", {}).get(f"/v1/{endpoint}", {})
        body = path.get("post", {}).get("requestBody", {})
        schema = self._deref(body.get("content", {}).get("application/json", {}).get("schema", {}))
        if not schema:
            return []
        variants = schema.get("oneOf") or schema.get("anyOf") or [schema]
        return [self._deref(v) for v in variants]

    def variant_for(self, endpoint: str, model_id: str) -> dict[str, Any] | None:
        for variant in self.request_variants(endpoint):
            if model_id in self._model_ids(variant):
                return variant
        return None

    def models_for(self, endpoint: str) -> set[str]:
        ids: set[str] = set()
        for variant in self.request_variants(endpoint):
            ids.update(self._model_ids(variant))
        return ids

    def _model_ids(self, variant: dict[str, Any]) -> list[str]:
        model = self._deref(variant.get("properties", {}).get("model", {}))
        if "enum" in model:
            return list(model["enum"])
        if "const" in model:
            return [model["const"]]
        return []

    def prop(self, variant: dict[str, Any], name: str) -> dict[str, Any] | None:
        props = variant.get("properties", {})
        if name not in props:
            return None
        return self._deref(props[name])

    def enum_of(self, variant: dict[str, Any], name: str) -> tuple[str, ...] | None:
        prop = self.prop(variant, name)
        if prop is None:
            return None
        if "enum" in prop:
            return tuple(prop["enum"])
        if "const" in prop:
            return (prop["const"],)
        return None

    def generatable_models(self) -> set[str]:
        """Every model identifier reachable through a generation request."""
        ids: set[str] = set()
        for path, methods in self._raw.get("paths", {}).items():
            if path.startswith(BILLING_PATH_PREFIXES):
                continue
            if "post" not in methods:
                continue
            endpoint = path.removeprefix("/v1/")
            ids.update(self.models_for(endpoint))
        return ids


def compare_set(label: str, declared: tuple[str, ...], actual: tuple[str, ...] | None) -> list[str]:
    """Compare an unordered choice list, reporting each direction separately.

    Ordering is cosmetic (it only drives dropdown order), so only membership is checked.
    """
    if actual is None:
        if declared:
            return [f"{label}: library declares {list(declared)} but the spec has no such field"]
        return []

    problems = []
    removed = [v for v in declared if v not in actual]
    added = [v for v in actual if v not in declared]
    if removed:
        problems.append(f"{label}: REMOVED upstream, still offered by this library: {removed}")
    if added:
        problems.append(f"{label}: new upstream, not offered by this library: {added}")
    return problems


def check_model(spec: Spec, model: ModelSpec) -> list[str]:
    where = f"{model.model_id} @ {model.endpoint}"
    variant = spec.variant_for(model.endpoint, model.model_id)
    if variant is None:
        available = sorted(spec.models_for(model.endpoint)) or ["<endpoint absent>"]
        return [f"{where}: GONE from the spec. Available on this endpoint now: {available}"]

    problems = compare_set(f"{where} ratio", model.ratios, spec.enum_of(variant, "ratio"))
    problems += compare_set(
        f"{where} targetAspectRatio", model.target_aspect_ratios, spec.enum_of(variant, "targetAspectRatio")
    )
    problems += compare_set(f"{where} outputFormat", model.output_formats, spec.enum_of(variant, "outputFormat"))
    problems += compare_set(f"{where} proresProfile", model.prores_profiles, spec.enum_of(variant, "proresProfile"))
    problems += compare_set(f"{where} resolution", model.resolutions, spec.enum_of(variant, "resolution"))

    problems += check_bounds(spec, variant, model, where)
    problems += check_payload_fields(variant, model, where)

    actual_required = frozenset(variant.get("required", []))
    if actual_required != model.required_fields:
        newly = sorted(actual_required - model.required_fields)
        no_longer = sorted(model.required_fields - actual_required)
        detail = []
        if newly:
            detail.append(f"now required upstream: {newly}")
        if no_longer:
            detail.append(f"no longer required upstream: {no_longer}")
        problems.append(f"{where} required: {'; '.join(detail)}")

    return problems


def check_payload_fields(variant: dict[str, Any], model: ModelSpec, where: str) -> list[str]:
    """Check that every key the node sends is one this model accepts.

    Three endpoints declare `additionalProperties: false`, so a single unrecognized key fails
    the whole request. Nothing else here catches that: the enum and bound checks only look at
    fields already declared on `ModelSpec`, so a field that was valid for a retired model and
    carried forward into its replacement passes every other check and 400s on every run.
    """
    if not model.payload_fields:
        return [f"{where} payload_fields: not declared, so the keys this node sends are unchecked"]

    accepted = set(variant.get("properties", {}))
    unknown = sorted(model.payload_fields - accepted)
    if not unknown:
        return []

    strictness = "rejects unknown keys" if variant.get("additionalProperties") is False else "ignores unknown keys"
    return [f"{where} payload_fields: {unknown} not accepted upstream (this model {strictness})"]


def check_bounds(spec: Spec, variant: dict[str, Any], model: ModelSpec, where: str) -> list[str]:
    """Compare numeric and array bounds, which drift far more quietly than enums.

    A widened `duration` range is invisible to users -- the node keeps working, it just
    silently withholds the new values.
    """
    problems = []

    duration = spec.prop(variant, "duration")
    if duration is not None:
        for label, declared, actual in (
            ("duration_min", model.duration_min, duration.get("minimum")),
            ("duration_max", model.duration_max, duration.get("maximum")),
        ):
            if declared != actual:
                problems.append(f"{where} {label}: library={declared} spec={actual}")
        if "enum" in duration:
            problems.append(f"{where} duration: spec now constrains it to an enum {duration['enum']}")

    for field_name, declared_max in (
        ("referenceImages", model.max_reference_images),
        ("keyframes", model.max_keyframes),
    ):
        prop = spec.prop(variant, field_name)
        actual_max = prop.get("maxItems") if prop is not None else None
        if declared_max != actual_max:
            problems.append(f"{where} {field_name} maxItems: library={declared_max} spec={actual_max}")

    prompt_text = spec.prop(variant, "promptText")
    if prompt_text is not None and prompt_text.get("maxLength") not in (None, MAX_PROMPT_LENGTH):
        problems.append(f"{where} promptText maxLength: library={MAX_PROMPT_LENGTH} spec={prompt_text['maxLength']}")

    seed = spec.prop(variant, "seed")
    if seed is not None:
        if seed.get("minimum") != SEED_MIN:
            problems.append(f"{where} seed minimum: library={SEED_MIN} spec={seed.get('minimum')}")
        if seed.get("maximum") != SEED_MAX:
            problems.append(f"{where} seed maximum: library={SEED_MAX} spec={seed.get('maximum')}")

    return problems


def check_for_new_models(spec: Spec) -> list[str]:
    """Report Runway-native models on covered endpoints that this library does not offer.

    Reported separately from drift because Runway proxies many third-party models through the
    same endpoints, so most of what turns up here is deliberately out of scope. Without it the
    guard is one-directional: it catches a model going away but reports green forever while the
    library falls behind on new ones.
    """
    declared_by_endpoint: dict[str, set[str]] = {}
    for model in MODELS:
        declared_by_endpoint.setdefault(model.endpoint, set()).add(model.model_id)

    notes = []
    for endpoint, declared in sorted(declared_by_endpoint.items()):
        upstream = spec.models_for(endpoint) - declared - set(RETIRED_MODELS)
        if upstream:
            notes.append(f"{endpoint}: upstream also offers {sorted(upstream)}")
    return notes


def check_retired_still_dead(spec: Spec) -> list[str]:
    """Flag any identifier this library calls retired that Runway has brought back."""
    generatable = spec.generatable_models()
    return [
        f"{model_id}: listed as retired here, but the spec accepts it again"
        for model_id in RETIRED_MODELS
        if model_id in generatable
    ]


def main() -> int:
    try:
        spec = Spec(fetch_spec())
    except OSError as e:
        print(f"Could not fetch Runway's OpenAPI spec from {SPEC_URL}: {e}", file=sys.stderr)
        return 2

    problems: list[str] = []

    if spec.api_version != RUNWAY_API_VERSION:
        problems.append(f"X-Runway-Version: library sends {RUNWAY_API_VERSION}, spec publishes {spec.api_version}")

    for model in MODELS:
        problems.extend(check_model(spec, model))
    problems.extend(check_retired_still_dead(spec))

    new_models = check_for_new_models(spec)
    if new_models:
        print("Models available upstream that this library does not offer:")
        for note in new_models:
            print(f"  - {note}")
        print("  (Most will be third-party models Runway proxies, which are out of scope.)\n")

    checked = f"{len(MODELS)} models across {len({m.endpoint for m in MODELS})} endpoints"
    if problems:
        print(f"Runway API drift detected ({checked}):\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            f"\n{len(problems)} discrepancies. Update runwayml/api_surface.py (and any node that "
            "mirrors it) to match, then re-run.",
            file=sys.stderr,
        )
        return 1

    print(f"No drift: {checked} match Runway's spec (API version {spec.api_version}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
