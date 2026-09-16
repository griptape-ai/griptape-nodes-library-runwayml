from typing import Any

from api_surface import ENDPOINT_CHARACTER_PERFORMANCE, get_model, model_choices
from griptape.artifacts import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options
from media import prepare_media_data_uri
from runway_node import RunwayTaskNode

DEFAULT_MODEL = "act_two"

CHARACTER_TYPE_IMAGE = "image"
CHARACTER_TYPE_VIDEO = "video"
CHARACTER_TYPES = [CHARACTER_TYPE_IMAGE, CHARACTER_TYPE_VIDEO]
DEFAULT_CHARACTER_TYPE = CHARACTER_TYPE_IMAGE

DEFAULT_ASPECT_RATIO = "1280:720"

EXPRESSION_INTENSITY_MIN = 1
EXPRESSION_INTENSITY_MAX = 5
DEFAULT_EXPRESSION_INTENSITY = 3


class RunwayML_ActTwo(RunwayTaskNode):
    endpoint = ENDPOINT_CHARACTER_PERFORMANCE
    output_filename = "output.mp4"
    output_parameter_name = "video_output"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "AI/RunwayML"
        self.description = "Generates a character performance video using RunwayML's Act Two API."

        character_type = ParameterString(
            name="character_type",
            default_value=DEFAULT_CHARACTER_TYPE,
            tooltip="Whether the character is supplied as a still image or as a video.",
            allowed_modes={ParameterMode.PROPERTY},
        )
        character_type.add_trait(Options(choices=CHARACTER_TYPES))
        self.add_parameter(character_type)

        self.add_parameter(
            Parameter(
                name="character_image",
                input_types=["ImageArtifact", "ImageUrlArtifact", "str"],
                type="ImageUrlArtifact",
                tooltip="Still image of the character to animate.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"clickable_file_browser": True},
            )
        )
        self.add_parameter(
            Parameter(
                name="character_video",
                input_types=["VideoUrlArtifact", "VideoArtifact"],
                type="VideoUrlArtifact",
                tooltip="Video of the character to animate.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "clickable_file_browser": True,
                    "expander": True,
                    "display_name": "Character Video or Path",
                },
            )
        )
        self.add_parameter(
            Parameter(
                name="reference_video",
                input_types=["VideoUrlArtifact", "VideoArtifact"],
                type="VideoUrlArtifact",
                tooltip="Driving performance. The character copies the motion and expression in this video.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "clickable_file_browser": True,
                    "expander": True,
                    "display_name": "Reference Video or Path",
                },
            )
        )

        with ParameterGroup(name="Settings") as settings:
            ParameterBool(
                name="body_control",
                default_value=True,
                tooltip="Transfer body motion in addition to facial expression.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            ParameterInt(
                name="expression_intensity",
                default_value=DEFAULT_EXPRESSION_INTENSITY,
                min_val=EXPRESSION_INTENSITY_MIN,
                max_val=EXPRESSION_INTENSITY_MAX,
                slider=True,
                validate_min_max=True,
                tooltip=(
                    f"How strongly the reference expression is applied "
                    f"({EXPRESSION_INTENSITY_MIN}-{EXPRESSION_INTENSITY_MAX})."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )

            ratio = ParameterString(
                name="ratio",
                default_value=DEFAULT_ASPECT_RATIO,
                tooltip="Aspect ratio for the output video.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            ratio.add_trait(Options(choices=list(get_model(DEFAULT_MODEL, ENDPOINT_CHARACTER_PERFORMANCE).ratios)))

            self._add_seed_parameters(settings)

            model = ParameterString(
                name="model",
                default_value=DEFAULT_MODEL,
                tooltip="RunwayML model to use for generation.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            model.add_trait(Options(choices=model_choices(ENDPOINT_CHARACTER_PERFORMANCE)))

            public_figure_threshold = ParameterString(
                name="public_figure_threshold",
                default_value="auto",
                tooltip="Public figure threshold for content moderation.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            public_figure_threshold.add_trait(Options(choices=["auto", "low"]))
        self.add_node_element(settings)

        self.add_parameter(
            Parameter(
                name="video_output",
                output_type="VideoUrlArtifact",
                type="VideoUrlArtifact",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="The generated performance video, saved into project files.",
                ui_options={"placeholder_text": "", "is_full_width": True, "pulse_on_run": True},
            )
        )

        self._add_output_file_parameter()
        self._create_status_parameters(result_details_placeholder="Generation progress will appear here.")

        self._update_parameter_visibility(DEFAULT_CHARACTER_TYPE)

    def _update_parameter_visibility(self, character_type: str) -> None:
        """Show only the character input that matches the selected type."""
        if character_type == CHARACTER_TYPE_IMAGE:
            self.show_parameter_by_name("character_image")
            self.hide_parameter_by_name("character_video")
        else:
            self.hide_parameter_by_name("character_image")
            self.show_parameter_by_name("character_video")

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "character_type":
            self._update_parameter_visibility(str(value or DEFAULT_CHARACTER_TYPE))

        return super().after_value_set(parameter, value)

    def _character_parameter_name(self) -> str:
        character_type = str(self.get_parameter_value("character_type") or DEFAULT_CHARACTER_TYPE)
        if character_type == CHARACTER_TYPE_IMAGE:
            return "character_image"
        return "character_video"

    def _get_data_uri(self, param_name: str) -> str | None:
        """Resolve an input to a value the character_performance endpoint accepts.

        Returns an ``https://``, ``runway://``, or ``data:<kind>/...`` URI, or ``None``
        when the parameter is unset or could not be loaded.
        """
        kind = "image" if param_name == "character_image" else "video"
        return prepare_media_data_uri(
            self.get_parameter_value(param_name),
            kind=kind,
            node_name=self.name,
        )

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = super().validate_before_node_run() or []

        character_type = str(self.get_parameter_value("character_type") or DEFAULT_CHARACTER_TYPE)
        if character_type not in CHARACTER_TYPES:
            errors.append(
                ValueError(
                    f"Attempted to generate a performance on '{self.name}'. Failed because the character type "
                    f"'{character_type}' is not one of {CHARACTER_TYPES}."
                )
            )
        elif not self.get_parameter_value(self._character_parameter_name()):
            errors.append(
                ValueError(
                    f"Attempted to generate a performance on '{self.name}'. Failed because no character "
                    f"{character_type} is set. Connect one or choose a file."
                )
            )

        if not self.get_parameter_value("reference_video"):
            errors.append(
                ValueError(
                    f"Attempted to generate a performance on '{self.name}'. Failed because no reference video "
                    "is set. Connect the driving performance video."
                )
            )

        try:
            get_model(str(self.get_parameter_value("model") or ""), ENDPOINT_CHARACTER_PERFORMANCE)
        except ValueError as e:
            errors.append(e)

        return errors or None

    def build_payload(self) -> dict[str, Any]:
        model_name = str(self.get_parameter_value("model") or DEFAULT_MODEL)
        get_model(model_name, ENDPOINT_CHARACTER_PERFORMANCE)

        character_type = str(self.get_parameter_value("character_type") or DEFAULT_CHARACTER_TYPE)
        character_uri = self._get_data_uri(self._character_parameter_name())
        reference_uri = self._get_data_uri("reference_video")

        if not character_uri or not reference_uri:
            msg = (
                f"Attempted to generate a performance on '{self.name}'. Failed because the character or "
                "reference video could not be read."
            )
            raise ValueError(msg)

        body_control = self.get_parameter_value("body_control")
        return {
            "model": model_name,
            "character": {"type": character_type, "uri": character_uri},
            "reference": {"type": "video", "uri": reference_uri},
            "bodyControl": True if body_control is None else bool(body_control),
            "expressionIntensity": int(
                self.get_parameter_value("expression_intensity") or DEFAULT_EXPRESSION_INTENSITY
            ),
            "ratio": str(self.get_parameter_value("ratio") or DEFAULT_ASPECT_RATIO),
            "seed": self._resolve_seed(),
            "contentModeration": self._content_moderation(),
        }

    def build_artifact(self, location: str) -> VideoUrlArtifact:
        return VideoUrlArtifact(value=location, name="runwayml_character_video")
