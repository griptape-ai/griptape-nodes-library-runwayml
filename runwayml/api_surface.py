"""Single source of truth for the slice of Runway's API this library exposes.

Every model identifier, ratio, and request bound that ``scripts/check_api_drift.py``
can verify against Runway's published OpenAPI spec lives here, so the drift check has
one module to diff. Runway retires models on its own schedule and the API rejects the
old identifier outright, so a constant copied into a node file is a constant that
rots silently.

Limits Runway enforces outside the request schema are recorded here too, but the
drift check cannot confirm them -- they come from live responses and the changelog,
and are marked where they appear.

Only Runway's own models are listed. Runway proxies many third-party models
(Google, OpenAI, ByteDance, MiniMax, xAI, Alibaba, Meta, ElevenLabs) through these
same endpoints; those belong to their vendors and are out of scope for this
library.
"""

from dataclasses import dataclass, field

RUNWAY_API_BASE = "https://api.dev.runwayml.com/v1"

# Runway pins behaviour to a dated API version rather than a semver range. It is sent
# on every request as the `X-Runway-Version` header and must match a version Runway
# still publishes, so it is verified against the spec's `info.version` by the drift check.
RUNWAY_API_VERSION = "2024-11-06"

# Applies to every endpoint that accepts a prompt, measured in UTF-16 code units.
MAX_PROMPT_LENGTH = 1000


def prompt_length(prompt: str) -> int:
    """Measure a prompt the way Runway measures it.

    Runway counts UTF-16 code units, so `len()` under-counts anything outside the BMP. An
    emoji-heavy prompt would pass a `len()` check and then come back as a raw 400, which is
    exactly what a local pre-check exists to prevent.
    """
    return len(prompt.encode("utf-16-le")) // 2


# Runway rejects an inlined data URI above this size. The limit applies to the encoded
# string, so the usable source file is roughly three quarters of it. Enforced while fetching
# the request, so the drift check cannot see it.
MAX_DATA_URI_BYTES = 5 * 1024 * 1024

# Runway rejects seeds outside this range rather than clamping them.
SEED_MIN = 0
SEED_MAX = 4294967295

ENDPOINT_TEXT_TO_IMAGE = "text_to_image"
ENDPOINT_TEXT_TO_VIDEO = "text_to_video"
ENDPOINT_IMAGE_TO_VIDEO = "image_to_video"
ENDPOINT_VIDEO_TO_VIDEO = "video_to_video"
ENDPOINT_VIDEO_UPSCALE = "video_upscale"
ENDPOINT_VIDEO_TO_HDR = "video_to_hdr"
ENDPOINT_CHARACTER_PERFORMANCE = "character_performance"

# ProRes and EXR outputs are encoded by Runway and downloaded as finished files. This
# library must never encode ProRes itself: the only encoders available to it are
# reverse-engineered and unlicensed, whereas Runway's server-side encode is licensed.

# The output formats that carry a ProRes stream and therefore accept `proresProfile`.
# Defined once: four nodes need the same rule, and a fifth ProRes container would otherwise
# have to be remembered in four places.
PRORES_OUTPUT_FORMATS = frozenset({"prores", "hdr_prores"})

PRORES_PROFILES_FULL = ("422 Proxy", "422 LT", "422", "422 HQ", "4444", "4444 XQ")
PRORES_PROFILES_HDR = ("422", "422 HQ", "4444")


# `hdr_exr_*` and `png_sequence` deliver a zip of frames rather than a single video
# file, so callers must branch on the container before building an output artifact.
FRAME_SEQUENCE_OUTPUT_FORMATS = (
    "png_sequence",
    "hdr_png_sequence",
    "hdr_exr_sequence",
    "hdr_exr_acescg_sequence_1_3",
    "hdr_exr_acescg_sequence_2_0",
)

# The file extension each output format actually arrives as. Saving a ProRes stream into a
# file named `.mp4` produces something no player will open, so the container has to be
# chosen from the requested format rather than from the node's default filename.
OUTPUT_FORMAT_EXTENSIONS: dict[str, str] = {
    "mp4": ".mp4",
    "hdr10": ".mp4",
    "hlg": ".mp4",
    "sdr_rec709_10bit": ".mp4",
    "prores": ".mov",
    "hdr_prores": ".mov",
    "hdr_pq_12bit_master": ".mov",
    "png_sequence": ".zip",
    "hdr_png_sequence": ".zip",
    "hdr_exr_sequence": ".zip",
    "hdr_exr_acescg_sequence_1_3": ".zip",
    "hdr_exr_acescg_sequence_2_0": ".zip",
}


def single_file_output_formats(formats: tuple[str, ...]) -> list[str]:
    """Drop the formats that arrive as a zip of frames.

    A frame sequence needs unpacking into a `Sequence` before anything downstream can read
    it, so offering one from a node that only saves a single file would hand the user a zip
    dressed as a video.
    """
    return [f for f in formats if f not in FRAME_SEQUENCE_OUTPUT_FORMATS]


@dataclass(frozen=True)
class ModelSpec:
    """The request constraints for one Runway model on one endpoint.

    A model that serves several endpoints gets one entry per endpoint, because the
    constraints genuinely differ: `gen4.5` accepts six ratios on image-to-video but
    only two on text-to-video.
    """

    model_id: str
    endpoint: str
    ratios: tuple[str, ...] = ()
    target_aspect_ratios: tuple[str, ...] = ()
    duration_min: int | None = None
    duration_max: int | None = None
    max_reference_images: int | None = None
    max_keyframes: int | None = None
    # Input-media limits RunwayML enforces while fetching the asset rather than in the
    # request schema, so they are recorded here from live responses and the changelog.
    min_input_video_seconds: int | None = None
    max_input_video_seconds: int | None = None
    output_formats: tuple[str, ...] = ()
    prores_profiles: tuple[str, ...] = ()
    resolutions: tuple[str, ...] = ()
    required_fields: frozenset[str] = field(default_factory=frozenset)
    # Every key the node sends for this model. Three endpoints declare
    # `additionalProperties: false`, so one key Runway does not recognize fails the whole
    # request -- which is how `references` (a gen4_aleph field) survived into aleph2 payloads.
    # Checked against the spec's properties by the drift check, and against the node's actual
    # `build_payload()` output by the test suite.
    payload_fields: frozenset[str] = field(default_factory=frozenset)

    @property
    def prompt_text_required(self) -> bool:
        return "promptText" in self.required_fields


GEN4_5_OUTPUT_FORMATS = (
    "mp4",
    "prores",
    "png_sequence",
    "hdr10",
    "hlg",
    "sdr_rec709_10bit",
    "hdr_pq_12bit_master",
    "hdr_prores",
    "hdr_png_sequence",
    "hdr_exr_sequence",
    "hdr_exr_acescg_sequence_1_3",
    "hdr_exr_acescg_sequence_2_0",
)

GEN4_IMAGE_RATIOS = (
    "720:720",
    "720:960",
    "720:1280",
    "960:720",
    "1024:1024",
    "1080:1080",
    "1080:1440",
    "1080:1920",
    "1168:880",
    "1280:720",
    "1360:768",
    "1440:1080",
    "1680:720",
    "1808:768",
    "1920:1080",
    "2112:912",
)

GEN4_VIDEO_RATIOS = ("1280:720", "720:1280", "1104:832", "832:1104", "960:960", "1584:672")

MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="gen4_image",
        endpoint=ENDPOINT_TEXT_TO_IMAGE,
        payload_fields=frozenset({"model", "promptText", "ratio", "seed", "contentModeration", "referenceImages"}),
        ratios=GEN4_IMAGE_RATIOS,
        max_reference_images=3,
        required_fields=frozenset({"model", "promptText", "ratio"}),
    ),
    # Unlike `gen4_image`, the turbo variant cannot run without references.
    ModelSpec(
        model_id="gen4_image_turbo",
        endpoint=ENDPOINT_TEXT_TO_IMAGE,
        payload_fields=frozenset({"model", "promptText", "ratio", "seed", "contentModeration", "referenceImages"}),
        ratios=GEN4_IMAGE_RATIOS,
        max_reference_images=3,
        required_fields=frozenset({"model", "promptText", "ratio", "referenceImages"}),
    ),
    # Declaration order is the dropdown order, and an `Options` trait snaps an
    # unrecognized value to the first choice. A workflow saved against the retired
    # `gen3a_turbo` therefore lands on whichever model leads this list, so the cheaper
    # direct successor goes first rather than the pricier `gen4.5`.
    ModelSpec(
        model_id="gen4_turbo",
        endpoint=ENDPOINT_IMAGE_TO_VIDEO,
        payload_fields=frozenset(
            {"model", "promptImage", "promptText", "ratio", "duration", "seed", "contentModeration"}
        ),
        ratios=GEN4_VIDEO_RATIOS,
        duration_min=2,
        duration_max=10,
        required_fields=frozenset({"model", "promptImage", "ratio"}),
    ),
    ModelSpec(
        model_id="gen4.5",
        endpoint=ENDPOINT_IMAGE_TO_VIDEO,
        ratios=GEN4_VIDEO_RATIOS,
        duration_min=2,
        duration_max=10,
        output_formats=GEN4_5_OUTPUT_FORMATS,
        prores_profiles=PRORES_PROFILES_FULL,
        required_fields=frozenset({"model", "promptImage", "promptText", "ratio", "duration"}),
        payload_fields=frozenset(
            {
                "model",
                "promptImage",
                "promptText",
                "ratio",
                "duration",
                "seed",
                "contentModeration",
                "outputFormat",
                "proresProfile",
            }
        ),
    ),
    ModelSpec(
        model_id="gen4.5",
        endpoint=ENDPOINT_TEXT_TO_VIDEO,
        ratios=("1280:720", "720:1280"),
        duration_min=2,
        duration_max=10,
        output_formats=GEN4_5_OUTPUT_FORMATS,
        prores_profiles=PRORES_PROFILES_FULL,
        required_fields=frozenset({"model", "promptText", "ratio", "duration"}),
        payload_fields=frozenset(
            {"model", "promptText", "ratio", "duration", "seed", "contentModeration", "outputFormat", "proresProfile"}
        ),
    ),
    # `ratio` is deprecated for aleph2; `targetAspectRatio` letterboxes the input for
    # expand/outpaint instead. The 2-30 second input window is enforced by RunwayML when it
    # fetches the asset during request validation, not by the request schema, so the drift
    # check cannot see it -- a shorter clip is rejected with "Asset duration must be at
    # least 2 seconds".
    ModelSpec(
        model_id="aleph2",
        endpoint=ENDPOINT_VIDEO_TO_VIDEO,
        min_input_video_seconds=2,
        max_input_video_seconds=30,
        target_aspect_ratios=("21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16"),
        max_keyframes=5,
        output_formats=("mp4", "prores", "png_sequence", "sdr_rec709_10bit"),
        prores_profiles=PRORES_PROFILES_FULL,
        required_fields=frozenset({"model", "videoUri"}),
        payload_fields=frozenset(
            {
                "model",
                "videoUri",
                "promptText",
                "targetAspectRatio",
                "keyframes",
                "seed",
                "contentModeration",
                "outputFormat",
                "proresProfile",
            }
        ),
    ),
    ModelSpec(
        model_id="magnific_video_upscaler_creative",
        endpoint=ENDPOINT_VIDEO_UPSCALE,
        payload_fields=frozenset(
            {"model", "videoUri", "resolution", "flavor", "creativity", "sharpen", "smartGrain", "fpsBoost"}
        ),
        resolutions=("720p", "1k", "2k", "4k"),
        max_input_video_seconds=30,
        required_fields=frozenset({"model", "videoUri"}),
    ),
    ModelSpec(
        model_id="ruby",
        endpoint=ENDPOINT_VIDEO_TO_HDR,
        payload_fields=frozenset({"model", "videoUri", "outputFormat", "proresProfile"}),
        max_input_video_seconds=30,
        output_formats=(
            "hdr10",
            "hlg",
            "hdr_prores",
            "hdr_exr_sequence",
            "hdr_exr_acescg_sequence_1_3",
            "hdr_exr_acescg_sequence_2_0",
        ),
        prores_profiles=PRORES_PROFILES_HDR,
        required_fields=frozenset({"model", "videoUri"}),
    ),
    ModelSpec(
        model_id="act_two",
        endpoint=ENDPOINT_CHARACTER_PERFORMANCE,
        payload_fields=frozenset(
            {
                "model",
                "character",
                "reference",
                "bodyControl",
                "expressionIntensity",
                "ratio",
                "seed",
                "contentModeration",
            }
        ),
        ratios=GEN4_VIDEO_RATIOS,
        required_fields=frozenset({"model", "character", "reference"}),
    ),
)

# Models Runway has retired. Kept so nodes can explain a stale saved workflow instead of
# forwarding a dead identifier and surfacing a bare 400. Retired identifiers stay valid in
# Runway's billing history, so their absence from a request schema is what marks them dead.
RETIRED_MODELS: dict[str, str] = {
    "gen3a_turbo": "Retired by Runway on 2026-07-30. Use 'gen4.5' for quality or 'gen4_turbo' for speed.",
    "gen4_aleph": "Retired by Runway on 2026-07-30. Use 'aleph2'.",
    "aleph2_alpha": "Deprecated alias. Use 'aleph2'.",
    "upscale_v1": "Replaced by Runway's Magnific upscaler. Use 'magnific_video_upscaler_creative'.",
}


def models_for(endpoint: str) -> tuple[ModelSpec, ...]:
    """Return every supported model for one endpoint, in declaration order."""
    return tuple(spec for spec in MODELS if spec.endpoint == endpoint)


def model_choices(endpoint: str) -> list[str]:
    """Return the model identifiers for one endpoint, for an ``Options`` trait."""
    return [spec.model_id for spec in models_for(endpoint)]


def get_model(model_id: str, endpoint: str) -> ModelSpec:
    """Look up one model's constraints.

    Raises:
        ValueError: If the model is unknown or retired for this endpoint. Retired
            identifiers get the migration hint from ``RETIRED_MODELS``.
    """
    for spec in MODELS:
        if spec.model_id == model_id and spec.endpoint == endpoint:
            return spec

    if model_id in RETIRED_MODELS:
        msg = f"Model '{model_id}' is no longer available from Runway. {RETIRED_MODELS[model_id]}"
        raise ValueError(msg)

    supported = ", ".join(model_choices(endpoint)) or "none"
    msg = f"Model '{model_id}' is not supported on Runway's {endpoint} endpoint. Supported models: {supported}."
    raise ValueError(msg)
