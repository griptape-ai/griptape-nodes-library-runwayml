from typing import Any

from api_surface import (
    ENDPOINT_VIDEO_TO_HDR,
    get_model,
    model_choices,
    single_file_output_formats,
)
from griptape.artifacts import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options
from media import prepare_media_data_uri
from runway_node import RunwayTaskNode

DEFAULT_MODEL = "ruby"

# Ruby extends the source's own luma and colour into the HDR range rather than
# re-synthesising pixels, so the input must be genuine SDR: RunwayML rejects a video that
# is already HDR-tagged.
MAX_INPUT_PIXELS_PER_SIDE = 4096

DEFAULT_OUTPUT_FORMAT = "hdr10"
DEFAULT_PRORES_PROFILE = "422 HQ"


class RunwayML_VideoToHDR(RunwayTaskNode):
    """Upconverts an SDR video to true HDR using RunwayML's Ruby grading model."""

    endpoint = ENDPOINT_VIDEO_TO_HDR
    output_filename = "output.mp4"
    output_parameter_name = "video_output"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "AI/RunwayML"
        self.description = "Converts an SDR video to true HDR using RunwayML's Ruby grading model."

        default_spec = get_model(DEFAULT_MODEL, ENDPOINT_VIDEO_TO_HDR)

        self.add_parameter(
            Parameter(
                name="video",
                input_types=["VideoUrlArtifact", "VideoArtifact"],
                type="VideoUrlArtifact",
                tooltip=(
                    f"SDR video to convert. Must be genuinely SDR (an HDR-tagged video is rejected), "
                    f"at most {default_spec.max_input_video_seconds}s, and under "
                    f"{MAX_INPUT_PIXELS_PER_SIDE}px per side."
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
            tooltip="RunwayML model to use for HDR conversion.",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        model.add_trait(Options(choices=model_choices(ENDPOINT_VIDEO_TO_HDR)))
        self.add_parameter(model)

        # Every format here is encoded by RunwayML and downloaded as a finished file.
        # Nothing in this library encodes ProRes locally.
        output_format = ParameterString(
            name="output_format",
            default_value=DEFAULT_OUTPUT_FORMAT,
            tooltip=(
                "Delivery profile. hdr10 and hlg are streaming HDR; hdr_prores is a BT.2020 + PQ "
                "editorial mezzanine. Billed per second, and at double rate above 4 megapixels."
            ),
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        output_format.add_trait(Options(choices=single_file_output_formats(default_spec.output_formats)))
        self.add_parameter(output_format)

        prores_profile = ParameterString(
            name="prores_profile",
            default_value=DEFAULT_PRORES_PROFILE,
            tooltip=(
                "ProRes tier, used only when the output format is hdr_prores. 422 Proxy and 422 LT "
                "are unavailable here because they quantize too heavily to hold HDR gradients."
            ),
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        prores_profile.add_trait(Options(choices=list(default_spec.prores_profiles)))
        self.add_parameter(prores_profile)

        self.add_parameter(
            Parameter(
                name="video_output",
                output_type="VideoUrlArtifact",
                type="VideoUrlArtifact",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="The HDR video, saved into project files.",
                ui_options={"placeholder_text": "", "is_full_width": True, "pulse_on_run": True},
            )
        )

        self._add_output_file_parameter()
        self._create_status_parameters(result_details_placeholder="Conversion progress will appear here.")

        self._sync_prores_visibility(DEFAULT_OUTPUT_FORMAT)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "output_format":
            output_format = str(value or DEFAULT_OUTPUT_FORMAT)
            self._sync_output_extension(output_format)
            self._sync_prores_visibility(output_format)

        return super().after_value_set(parameter, value)

    def _get_video_uri(self) -> str | None:
        """Resolve the ``video`` input to a value the /v1/video_to_hdr endpoint accepts."""
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
                    f"Attempted to convert a video to HDR on '{self.name}'. Failed because no input video is "
                    "set. Connect a video or choose a file."
                )
            )

        try:
            get_model(str(self.get_parameter_value("model") or ""), ENDPOINT_VIDEO_TO_HDR)
        except ValueError as e:
            errors.append(e)

        return errors or None

    def build_payload(self) -> dict[str, Any]:
        model_name = str(self.get_parameter_value("model") or DEFAULT_MODEL)
        get_model(model_name, ENDPOINT_VIDEO_TO_HDR)

        video_uri = self._get_video_uri()
        if not video_uri:
            msg = (
                f"Attempted to convert a video to HDR on '{self.name}'. Failed because the input video "
                "could not be read."
            )
            raise ValueError(msg)

        payload: dict[str, Any] = {"model": model_name, "videoUri": video_uri}
        self._attach_output_format(
            payload,
            get_model(model_name, ENDPOINT_VIDEO_TO_HDR),
            default_format=DEFAULT_OUTPUT_FORMAT,
            always_send=True,
        )

        return payload

    def build_artifact(self, location: str) -> VideoUrlArtifact:
        return VideoUrlArtifact(value=location, name="runwayml_hdr_video")
