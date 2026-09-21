from typing import Any

from api_surface import (
    ENDPOINT_IMAGE_TO_VIDEO,
    MAX_PROMPT_LENGTH,
    get_model,
    model_choices,
    single_file_output_formats,
)
from griptape.artifacts import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options
from media import prepare_media_data_uri
from runway_node import RunwayTaskNode

# Kept as the default over the higher-quality `gen4.5` so existing workflows do not
# silently move to a different model, and a different per-second price, on upgrade.
DEFAULT_MODEL = "gen4_turbo"

DEFAULT_ASPECT_RATIO = "1280:720"
DEFAULT_DURATION = 10
DEFAULT_OUTPUT_FORMAT = "mp4"

# Only `gen4.5` accepts the professional delivery formats; `gen4_turbo` returns H.264 mp4
# only, so the group is hidden unless it can do anything.
PRO_OUTPUT_MODELS = frozenset({"gen4.5"})


class RunwayML_ImageToVideo(RunwayTaskNode):
    endpoint = ENDPOINT_IMAGE_TO_VIDEO
    output_filename = "output.mp4"
    output_parameter_name = "video_output"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "AI/RunwayML"
        self.description = "Generates a video from an image and prompt using RunwayML."

        self.add_parameter(
            Parameter(
                name="image",
                input_types=["ImageUrlArtifact", "ImageArtifact", "str"],
                type="ImageUrlArtifact",
                tooltip="Starting frame for the video. Accepts an image artifact, a public URL, or a data URI.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"clickable_file_browser": True},
            )
        )
        self.add_parameter(
            ParameterString(
                name="prompt",
                default_value="",
                tooltip=f"Text prompt describing the desired video content (max {MAX_PROMPT_LENGTH} characters).",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                multiline=True,
                placeholder_text="e.g., a cinematic shot of a car driving down a road",
            )
        )

        model = ParameterString(
            name="model",
            default_value=DEFAULT_MODEL,
            tooltip="RunwayML model to use for generation. gen4.5 is higher quality; gen4_turbo is faster and cheaper.",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        model.add_trait(Options(choices=model_choices(ENDPOINT_IMAGE_TO_VIDEO)))
        self.add_parameter(model)

        default_spec = get_model(DEFAULT_MODEL, ENDPOINT_IMAGE_TO_VIDEO)

        ratio = ParameterString(
            name="ratio",
            default_value=DEFAULT_ASPECT_RATIO,
            tooltip="Aspect ratio for the output video. Available ratios depend on the selected model.",
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
                tooltip=(
                    "Delivery container. Anything other than mp4 carries a per-second credit "
                    "surcharge. Only available on gen4.5."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            output_format.add_trait(
                Options(choices=single_file_output_formats(get_model("gen4.5", ENDPOINT_IMAGE_TO_VIDEO).output_formats))
            )

            prores_profile = ParameterString(
                name="prores_profile",
                default_value="4444",
                tooltip="ProRes tier. Only applies when the output format is prores or hdr_prores.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            prores_profile.add_trait(
                Options(choices=list(get_model("gen4.5", ENDPOINT_IMAGE_TO_VIDEO).prores_profiles))
            )
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

        self._sync_model_dependent_parameters(DEFAULT_MODEL)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "model":
            self._sync_model_dependent_parameters(str(value or DEFAULT_MODEL))
        elif parameter.name == "output_format":
            output_format = str(value or DEFAULT_OUTPUT_FORMAT)
            self._sync_output_extension(output_format)
            self._sync_prores_visibility(output_format)

        return super().after_value_set(parameter, value)

    def _sync_model_dependent_parameters(self, model_name: str) -> None:
        """Narrow the ratio list and hide the pro-output group for models that lack it."""
        try:
            spec = get_model(model_name, ENDPOINT_IMAGE_TO_VIDEO)
        except ValueError:
            # A stale saved workflow can name a retired model. Leave the choices alone;
            # build_payload reports the model itself as the problem.
            return

        # `_update_option_choices` always writes the default it is given, so passing a fixed
        # default would discard a ratio the new model still supports -- and both models here
        # share one ratio list, which would make every model switch a silent reset.
        current = self.get_parameter_value("ratio")
        if current in spec.ratios:
            default = str(current)
        elif DEFAULT_ASPECT_RATIO in spec.ratios:
            default = DEFAULT_ASPECT_RATIO
        else:
            default = spec.ratios[0]
        self._update_option_choices(param="ratio", choices=list(spec.ratios), default=default)

        if model_name in PRO_OUTPUT_MODELS:
            self.show_parameter_by_name("output_format")
            self.show_parameter_by_name("prores_profile")
            return

        self.hide_parameter_by_name("output_format")
        self.hide_parameter_by_name("prores_profile")
        # Reset rather than just hide. A stale `prores` left on a model that cannot deliver it
        # is dropped from the payload, so the user would be billed for an mp4 while the node
        # still displayed ProRes and a `.mov` filename.
        if self.get_parameter_value("output_format") != DEFAULT_OUTPUT_FORMAT:
            self.set_parameter_value("output_format", DEFAULT_OUTPUT_FORMAT)
            self._sync_output_extension(DEFAULT_OUTPUT_FORMAT)

    def _get_image_data_uri(self) -> str | None:
        """Resolve the image input to a value the /v1/image_to_video endpoint accepts."""
        return prepare_media_data_uri(
            self.get_parameter_value("image"),
            kind="image",
            node_name=self.name,
        )

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = super().validate_before_node_run() or []

        # Presence only: the engine calls this synchronously on its event loop, so reading and
        # encoding media here would stall every other node. `build_payload` runs in a thread
        # and reports anything unreadable from there.
        if not self.get_parameter_value("image"):
            errors.append(
                ValueError(
                    f"Attempted to generate a video on '{self.name}'. Failed because no starting image is "
                    "set. Connect an image or choose a file."
                )
            )

        model_name = str(self.get_parameter_value("model") or DEFAULT_MODEL)
        try:
            spec = get_model(model_name, ENDPOINT_IMAGE_TO_VIDEO)
        except ValueError as e:
            errors.append(e)
            return errors or None

        prompt = str(self.get_parameter_value("prompt") or "").strip()
        if spec.prompt_text_required and not prompt:
            errors.append(
                ValueError(
                    f"Attempted to generate a video on '{self.name}' with {model_name}. "
                    "Failed because that model requires a text prompt."
                )
            )

        return errors or None

    def build_payload(self) -> dict[str, Any]:
        model_name = str(self.get_parameter_value("model") or DEFAULT_MODEL)
        spec = get_model(model_name, ENDPOINT_IMAGE_TO_VIDEO)

        image_data_uri = self._get_image_data_uri()
        if not image_data_uri:
            msg = (
                f"Attempted to generate a video on '{self.name}'. Failed because the starting image could not be read."
            )
            raise ValueError(msg)

        payload: dict[str, Any] = {
            "model": model_name,
            "promptImage": image_data_uri,
            "ratio": str(self.get_parameter_value("ratio") or DEFAULT_ASPECT_RATIO),
            "duration": int(self.get_parameter_value("duration") or DEFAULT_DURATION),
            "seed": self._resolve_seed(),
            "contentModeration": self._content_moderation(),
        }

        prompt = str(self.get_parameter_value("prompt") or "").strip()
        if prompt:
            payload["promptText"] = prompt

        self._attach_output_format(payload, spec, default_format=DEFAULT_OUTPUT_FORMAT)
        return payload

    def build_artifact(self, location: str) -> VideoUrlArtifact:
        return VideoUrlArtifact(value=location, name="runwayml_video")
