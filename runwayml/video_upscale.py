from typing import Any

from api_surface import ENDPOINT_VIDEO_UPSCALE, get_model, model_choices
from griptape.artifacts import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options
from media import prepare_media_data_uri
from runway_node import RunwayTaskNode

DEFAULT_MODEL = "magnific_video_upscaler_creative"

DEFAULT_RESOLUTION = "2k"
FLAVORS = ["vivid", "natural"]
DEFAULT_FLAVOR = "natural"

# Runway scores creativity, sharpen, and smart grain on the same 0-100 scale.
STRENGTH_MIN = 0
STRENGTH_MAX = 100


class RunwayML_VideoUpscale(RunwayTaskNode):
    endpoint = ENDPOINT_VIDEO_UPSCALE
    output_filename = "output.mp4"
    output_parameter_name = "video_output"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        default_spec = get_model(DEFAULT_MODEL, ENDPOINT_VIDEO_UPSCALE)

        self.category = "AI/RunwayML"
        self.description = "Upscales a video using RunwayML's Magnific video upscaler."

        # Plain Parameter, not ParameterString: ParameterString hardcodes type/output_type to
        # "str" and silently ignores the ones passed in, which would break artifact connections.
        self.add_parameter(
            Parameter(
                name="video",
                input_types=["VideoUrlArtifact", "VideoArtifact"],
                type="VideoUrlArtifact",
                tooltip=(
                    "Input video. Allowed content-types: video/mp4, video/webm, video/quicktime, "
                    f"video/mov, video/ogg, video/h264. Max duration {default_spec.max_input_video_seconds}s."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "clickable_file_browser": True,
                    "expander": True,
                    "display_name": "Video or Path to Video",
                },
            )
        )

        model = ParameterString(
            name="model",
            default_value=DEFAULT_MODEL,
            tooltip="RunwayML model variant to use for upscaling.",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        model.add_trait(Options(choices=model_choices(ENDPOINT_VIDEO_UPSCALE)))
        self.add_parameter(model)

        resolution = ParameterString(
            name="resolution",
            default_value=DEFAULT_RESOLUTION,
            tooltip="Output resolution. Billing is per output frame, so 4k costs considerably more than 720p.",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        resolution.add_trait(Options(choices=list(default_spec.resolutions)))
        self.add_parameter(resolution)

        with ParameterGroup(name="Upscaler Settings", collapsed=True) as settings:
            flavor = ParameterString(
                name="flavor",
                default_value=DEFAULT_FLAVOR,
                tooltip="Overall look: 'natural' stays closer to the source, 'vivid' pushes contrast and colour.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            flavor.add_trait(Options(choices=FLAVORS))

            ParameterInt(
                name="creativity",
                default_value=STRENGTH_MIN,
                min_val=STRENGTH_MIN,
                max_val=STRENGTH_MAX,
                slider=True,
                validate_min_max=True,
                tooltip=(
                    "How much detail the upscaler is allowed to invent. Higher values add detail "
                    "that was not in the source, which can drift from the original footage."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            ParameterInt(
                name="sharpen",
                default_value=STRENGTH_MIN,
                min_val=STRENGTH_MIN,
                max_val=STRENGTH_MAX,
                slider=True,
                validate_min_max=True,
                tooltip="Edge sharpening applied to the upscaled frames.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            ParameterInt(
                name="smart_grain",
                default_value=STRENGTH_MIN,
                min_val=STRENGTH_MIN,
                max_val=STRENGTH_MAX,
                slider=True,
                validate_min_max=True,
                tooltip="Film grain added after upscaling, to keep the result from looking overly clean.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            ParameterBool(
                name="fps_boost",
                default_value=False,
                tooltip="Interpolate additional frames to raise the output frame rate.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        self.add_node_element(settings)

        self.add_parameter(
            Parameter(
                name="video_output",
                output_type="VideoUrlArtifact",
                type="VideoUrlArtifact",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="The upscaled video, saved into project files.",
                ui_options={"placeholder_text": "", "is_full_width": True, "pulse_on_run": True},
            )
        )
        self._add_output_file_parameter()
        self._create_status_parameters(result_details_placeholder="Upscaling progress will appear here.")

    def _get_video_uri(self) -> str | None:
        """Resolve the ``video`` input to a value the /v1/video_upscale endpoint accepts.

        ``https://`` URLs and ``data:video/...`` URIs pass through; macro paths,
        local files, and ``http://`` URLs are read via ``File`` and returned as
        ``data:video/mp4;base64,...`` URIs.
        """
        return prepare_media_data_uri(
            self.get_parameter_value("video"),
            kind="video",
            node_name=self.name,
        )

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = super().validate_before_node_run() or []

        # Presence only: the engine calls this synchronously on its event loop, so reading and
        # encoding media here would stall every other node. `build_payload` runs in a thread
        # and reports anything unreadable from there.
        if not self.get_parameter_value("video"):
            errors.append(
                ValueError(
                    f"Attempted to upscale a video on '{self.name}'. Failed because no input video is set. "
                    "Connect a video or choose a file."
                )
            )

        try:
            get_model(str(self.get_parameter_value("model") or ""), ENDPOINT_VIDEO_UPSCALE)
        except ValueError as e:
            errors.append(e)

        return errors or None

    def build_payload(self) -> dict[str, Any]:
        model_name = str(self.get_parameter_value("model") or DEFAULT_MODEL)
        get_model(model_name, ENDPOINT_VIDEO_UPSCALE)

        video_uri = self._get_video_uri()
        if not video_uri:
            msg = f"Attempted to upscale a video on '{self.name}'. Failed because the input video could not be read."
            raise ValueError(msg)

        return {
            "model": model_name,
            "videoUri": video_uri,
            "resolution": str(self.get_parameter_value("resolution") or DEFAULT_RESOLUTION),
            "flavor": str(self.get_parameter_value("flavor") or DEFAULT_FLAVOR),
            "creativity": int(self.get_parameter_value("creativity") or STRENGTH_MIN),
            "sharpen": int(self.get_parameter_value("sharpen") or STRENGTH_MIN),
            "smartGrain": int(self.get_parameter_value("smart_grain") or STRENGTH_MIN),
            "fpsBoost": bool(self.get_parameter_value("fps_boost")),
        }

    def build_artifact(self, location: str) -> VideoUrlArtifact:
        return VideoUrlArtifact(value=location, name="runwayml_upscaled_video")
