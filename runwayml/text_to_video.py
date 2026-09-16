from typing import Any

from api_surface import (
    ENDPOINT_TEXT_TO_VIDEO,
    MAX_PROMPT_LENGTH,
    PRORES_OUTPUT_FORMATS,
    get_model,
    model_choices,
    prompt_length,
    prores_profiles_for,
    single_file_output_formats,
)
from griptape.artifacts import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options
from runway_node import RunwayTaskNode

DEFAULT_MODEL = "gen4.5"
DEFAULT_ASPECT_RATIO = "1280:720"
DEFAULT_DURATION = 5
DEFAULT_OUTPUT_FORMAT = "mp4"
DEFAULT_PRORES_PROFILE = "4444"


class RunwayML_TextToVideo(RunwayTaskNode):
    """Generates video from a text prompt alone, with no starting image."""

    endpoint = ENDPOINT_TEXT_TO_VIDEO
    output_filename = "output.mp4"
    output_parameter_name = "video_output"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "AI/RunwayML"
        self.description = "Generates a video from a text prompt using RunwayML."

        default_spec = get_model(DEFAULT_MODEL, ENDPOINT_TEXT_TO_VIDEO)

        self.add_parameter(
            ParameterString(
                name="prompt",
                default_value="",
                tooltip=f"Text prompt describing the video to generate (max {MAX_PROMPT_LENGTH} characters).",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                multiline=True,
                placeholder_text="e.g., a slow dolly through a rain-soaked neon alley at night",
            )
        )

        model = ParameterString(
            name="model",
            default_value=DEFAULT_MODEL,
            tooltip="RunwayML model to use for generation.",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        model.add_trait(Options(choices=model_choices(ENDPOINT_TEXT_TO_VIDEO)))
        self.add_parameter(model)

        ratio = ParameterString(
            name="ratio",
            default_value=DEFAULT_ASPECT_RATIO,
            tooltip="Aspect ratio for the output video.",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        ratio.add_trait(Options(choices=list(default_spec.ratios)))
        self.add_parameter(ratio)

        self.add_parameter(
            ParameterInt(
                name="duration",
                default_value=DEFAULT_DURATION,
                min_val=default_spec.duration_min,
                max_val=default_spec.duration_max,
                validate_min_max=True,
                tooltip=(
                    f"Duration of the output video in seconds "
                    f"({default_spec.duration_min}-{default_spec.duration_max}). Billed per second."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        with ParameterGroup(name="Generation Settings", collapsed=True) as settings:
            self._add_seed_parameters(settings)
            content_moderation = ParameterString(
                name="content_moderation",
                default_value="auto",
                tooltip="Content moderation level. 'low' is less strict about public figures.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            content_moderation.add_trait(Options(choices=["auto", "low"]))
        self.add_node_element(settings)

        # ProRes and the 10/12-bit HDR profiles are encoded by RunwayML and downloaded as
        # finished files. Nothing here encodes them locally.
        with ParameterGroup(name="Professional Output", collapsed=True) as pro_output:
            output_format = ParameterString(
                name="output_format",
                default_value=DEFAULT_OUTPUT_FORMAT,
                tooltip="Delivery container. Anything other than mp4 carries a per-second credit surcharge.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            output_format.add_trait(Options(choices=single_file_output_formats(default_spec.output_formats)))

            prores_profile = ParameterString(
                name="prores_profile",
                default_value=DEFAULT_PRORES_PROFILE,
                tooltip="ProRes tier. Only applies when the output format is prores or hdr_prores.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            prores_profile.add_trait(Options(choices=list(default_spec.prores_profiles)))
        self.add_node_element(pro_output)

        self.add_parameter(
            Parameter(
                name="video_output",
                output_type="VideoUrlArtifact",
                type="VideoUrlArtifact",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="The generated video, saved into project files.",
                ui_options={"placeholder_text": "", "is_full_width": True, "pulse_on_run": True},
            )
        )

        self._add_output_file_parameter()
        self._create_status_parameters(result_details_placeholder="Generation progress will appear here.")

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "output_format":
            self._sync_output_extension(str(value or DEFAULT_OUTPUT_FORMAT))

        return super().after_value_set(parameter, value)

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = super().validate_before_node_run() or []

        prompt = str(self.get_parameter_value("prompt") or "").strip()
        if not prompt:
            errors.append(
                ValueError(f"Attempted to generate a video on '{self.name}'. Failed because the prompt is empty.")
            )
        elif prompt_length(prompt) > MAX_PROMPT_LENGTH:
            errors.append(
                ValueError(
                    f"Attempted to generate a video on '{self.name}'. Failed because the prompt is "
                    f"{prompt_length(prompt)} characters and RunwayML allows at most {MAX_PROMPT_LENGTH}."
                )
            )

        try:
            get_model(str(self.get_parameter_value("model") or ""), ENDPOINT_TEXT_TO_VIDEO)
        except ValueError as e:
            errors.append(e)

        return errors or None

    def build_payload(self) -> dict[str, Any]:
        model_name = str(self.get_parameter_value("model") or DEFAULT_MODEL)
        get_model(model_name, ENDPOINT_TEXT_TO_VIDEO)

        payload: dict[str, Any] = {
            "model": model_name,
            "promptText": str(self.get_parameter_value("prompt") or "").strip(),
            "ratio": str(self.get_parameter_value("ratio") or DEFAULT_ASPECT_RATIO),
            "duration": int(self.get_parameter_value("duration") or DEFAULT_DURATION),
            "seed": self._resolve_seed(),
            "contentModeration": self._content_moderation(),
        }

        output_format = str(self.get_parameter_value("output_format") or DEFAULT_OUTPUT_FORMAT)
        if output_format != DEFAULT_OUTPUT_FORMAT:
            # `output_format` accepts a connection, so the dropdown's frame-sequence filter is
            # not a gate on its own. Saving a zip of frames as a video is worse than refusing it.
            deliverable = single_file_output_formats(get_model(model_name, ENDPOINT_TEXT_TO_VIDEO).output_formats)
            if output_format not in deliverable:
                msg = (
                    f"Attempted to generate a video on '{self.name}' as '{output_format}'. Failed "
                    f"because {model_name} cannot deliver that format. It supports: {', '.join(deliverable)}."
                )
                raise ValueError(msg)

            payload["outputFormat"] = output_format
            if output_format in PRORES_OUTPUT_FORMATS:
                spec = get_model(model_name, ENDPOINT_TEXT_TO_VIDEO)
                profile = str(self.get_parameter_value("prores_profile") or DEFAULT_PRORES_PROFILE)
                allowed = prores_profiles_for(output_format, spec.prores_profiles)
                if profile not in allowed:
                    msg = (
                        f"Attempted to deliver '{output_format}' from '{self.name}' as ProRes {profile}. "
                        f"Failed because that container serves only: {', '.join(allowed)}."
                    )
                    raise ValueError(msg)
                payload["proresProfile"] = profile

        return payload

    def build_artifact(self, location: str) -> VideoUrlArtifact:
        return VideoUrlArtifact(value=location, name="runwayml_video")
