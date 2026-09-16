import base64
import io
from typing import Any

from api_surface import (
    ENDPOINT_TEXT_TO_IMAGE,
    MAX_DATA_URI_BYTES,
    MAX_PROMPT_LENGTH,
    get_model,
    model_choices,
    prompt_length,
)
from artifacts import ReferenceImageArtifact, unpack_reference_image
from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterList, ParameterMode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.griptape_nodes import logger
from griptape_nodes.traits.options import Options
from media import prepare_media_data_uri
from PIL import Image
from runway_node import RunwayTaskNode

# Re-exported so `from text_to_image import ReferenceImageArtifact` keeps working for saved
# workflows and for the Create Reference Image node.
__all__ = ["ReferenceImageArtifact", "RunwayML_TextToImage"]

DEFAULT_MODEL = "gen4_image"
DEFAULT_RATIO = "1024:1024"

# RunwayML accepts these image types directly; anything else is transcoded to PNG.
SUPPORTED_IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/jpg", "image/webp"})


class RunwayML_TextToImage(RunwayTaskNode):
    endpoint = ENDPOINT_TEXT_TO_IMAGE
    output_filename = "output.png"
    output_parameter_name = "image_output"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "AI/RunwayML"
        self.description = "Generates images from text prompts with optional reference images using RunwayML."

        default_spec = get_model(DEFAULT_MODEL, ENDPOINT_TEXT_TO_IMAGE)
        self._max_reference_images = default_spec.max_reference_images or 3

        with ParameterGroup(name="Prompt") as prompt_group:
            ParameterString(
                name="prompt_text",
                default_value="",
                tooltip=(
                    f"Text prompt describing the desired image (max {MAX_PROMPT_LENGTH} characters). "
                    "Use @tagname to address a connected reference image, for example "
                    "'@EiffelTower painted in the style of @StarryNight'."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                multiline=True,
                placeholder_text="e.g., @EiffelTower painted in the style of @StarryNight",
            )
        self.add_node_element(prompt_group)

        with ParameterGroup(name="Reference Images") as ref_images_group:
            ParameterList(
                name="reference_images",
                input_types=["ReferenceImageArtifact"],
                output_type="ReferenceImageArtifact",
                type="ReferenceImageArtifact",
                default_value=None,
                tooltip=(
                    f"Up to {self._max_reference_images} tagged reference images. Connect from "
                    "'Runway Create Reference Image' nodes."
                ),
                allowed_modes={ParameterMode.INPUT},
            )
        self.add_node_element(ref_images_group)

        with ParameterGroup(name="Generation Settings") as gen_settings_group:
            model = ParameterString(
                name="model",
                default_value=DEFAULT_MODEL,
                tooltip=(
                    "RunwayML model to use. gen4_image_turbo is faster and cheaper but requires "
                    "at least one reference image."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            model.add_trait(Options(choices=model_choices(ENDPOINT_TEXT_TO_IMAGE)))

            ratio = ParameterString(
                name="ratio",
                default_value=DEFAULT_RATIO,
                tooltip="Aspect ratio for the output image.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            ratio.add_trait(Options(choices=list(default_spec.ratios)))

            self._add_seed_parameters(gen_settings_group)

            content_moderation = ParameterString(
                name="content_moderation",
                default_value="auto",
                tooltip="Content moderation level. 'low' is less strict about public figures.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            content_moderation.add_trait(Options(choices=["auto", "low"]))
        self.add_node_element(gen_settings_group)

        self.add_parameter(
            Parameter(
                name="image_output",
                output_type="ImageUrlArtifact",
                type="ImageUrlArtifact",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="The generated image, saved into project files.",
                ui_options={"placeholder_text": "", "is_full_width": True, "pulse_on_run": True},
            )
        )

        self._add_output_file_parameter()
        self._create_status_parameters(result_details_placeholder="Generation progress will appear here.")

    def _to_supported_data_uri(self, data_uri: str) -> str | None:
        """Transcode a data URI to PNG unless RunwayML already accepts its format."""
        if not data_uri.startswith("data:image"):
            logger.warning("%s: not an image data URI: %s...", self.name, data_uri[:50])
            return None

        if ";base64," not in data_uri:
            # A non-base64 data URI (e.g. inline SVG) is passed through verbatim upstream, and
            # unpacking it here would surface as "not enough values to unpack".
            logger.warning("%s: image data URI is not base64-encoded: %s...", self.name, data_uri[:50])
            return None

        header, base64_data = data_uri.split(";base64,", 1)
        media_type = header.removeprefix("data:")
        if media_type in SUPPORTED_IMAGE_MEDIA_TYPES:
            return data_uri

        logger.info("%s: converting unsupported format %s to PNG", self.name, media_type)
        try:
            with Image.open(io.BytesIO(base64.b64decode(base64_data))) as img:
                if img.mode == "P":
                    img = img.convert("RGBA")
                elif img.mode not in ("RGB", "RGBA", "LA"):
                    img = img.convert("RGB")

                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
        except (OSError, ValueError) as e:
            logger.error("%s: failed to convert %s to PNG: %s", self.name, media_type, e)
            return None

        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"

    def _resolve_reference_image(self, image: Any) -> str:
        """Turn one reference image into a URI RunwayML accepts.

        Raises:
            ValueError: If the image cannot be read, converted, or is over the size limit.
        """
        resolved = prepare_media_data_uri(image, kind="image", node_name=self.name)
        if not resolved:
            msg = f"A reference image on '{self.name}' could not be read."
            raise ValueError(msg)

        # HTTPS and runway:// URIs are sent as-is and are not subject to the inline size cap.
        if not resolved.startswith("data:image"):
            return resolved

        converted = self._to_supported_data_uri(resolved)
        if not converted:
            msg = f"A reference image on '{self.name}' is not in a format RunwayML can read."
            raise ValueError(msg)

        size = len(converted.encode("utf-8"))
        if size > MAX_DATA_URI_BYTES:
            msg = (
                f"A reference image on '{self.name}' is too large ({size / (1024 * 1024):.1f}MB encoded). "
                f"RunwayML allows {MAX_DATA_URI_BYTES // (1024 * 1024)}MB inline, about 3.3MB of source file. "
                "Use a smaller image or a public HTTPS URL."
            )
            raise ValueError(msg)

        return converted

    def _build_reference_images(self) -> list[dict[str, str]]:
        """Resolve every connected reference image into RunwayML's payload shape.

        Raises:
            ValueError: If a reference image is unusable or too many are connected.
        """
        connected = self.get_parameter_list_value("reference_images") or []
        if len(connected) > self._max_reference_images:
            msg = (
                f"Attempted to generate an image on '{self.name}' with {len(connected)} reference images. "
                f"RunwayML accepts at most {self._max_reference_images}."
            )
            raise ValueError(msg)

        references = []
        for index, candidate in enumerate(connected):
            image, tag = unpack_reference_image(candidate)
            if image is None:
                msg = (
                    f"Reference image {index + 1} on '{self.name}' is not a reference image. "
                    "Connect a 'Runway Create Reference Image' node."
                )
                raise ValueError(msg)

            # A tag is what makes the image addressable as @tag in the prompt, so an
            # untagged image still needs one to be usable at all.
            references.append({"uri": self._resolve_reference_image(image), "tag": tag or f"ref_{index + 1}"})

        return references

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = super().validate_before_node_run() or []

        prompt = str(self.get_parameter_value("prompt_text") or "").strip()
        if not prompt:
            errors.append(
                ValueError(f"Attempted to generate an image on '{self.name}'. Failed because the prompt is empty.")
            )
        elif prompt_length(prompt) > MAX_PROMPT_LENGTH:
            errors.append(
                ValueError(
                    f"Attempted to generate an image on '{self.name}'. Failed because the prompt is "
                    f"{prompt_length(prompt)} characters and RunwayML allows at most {MAX_PROMPT_LENGTH}."
                )
            )

        model_name = str(self.get_parameter_value("model") or DEFAULT_MODEL)
        try:
            spec = get_model(model_name, ENDPOINT_TEXT_TO_IMAGE)
        except ValueError as e:
            errors.append(e)
            return errors or None

        # Counted, not resolved: the engine calls this synchronously on its event loop, and
        # resolving here would PIL-decode and base64 every reference image before the run.
        # `build_payload` runs in a thread and reports an unusable image from there.
        connected = self.get_parameter_list_value("reference_images") or []
        if len(connected) > self._max_reference_images:
            errors.append(
                ValueError(
                    f"Attempted to generate an image on '{self.name}' with {len(connected)} reference images. "
                    f"RunwayML accepts at most {self._max_reference_images}."
                )
            )

        if "referenceImages" in spec.required_fields and not connected:
            errors.append(
                ValueError(
                    f"Attempted to generate an image on '{self.name}' with {model_name}. Failed because that "
                    "model requires at least one reference image."
                )
            )

        return errors or None

    def build_payload(self) -> dict[str, Any]:
        model_name = str(self.get_parameter_value("model") or DEFAULT_MODEL)
        get_model(model_name, ENDPOINT_TEXT_TO_IMAGE)

        payload: dict[str, Any] = {
            "model": model_name,
            "promptText": str(self.get_parameter_value("prompt_text") or "").strip(),
            "ratio": str(self.get_parameter_value("ratio") or DEFAULT_RATIO),
            "seed": self._resolve_seed(),
            "contentModeration": self._content_moderation(),
        }

        references = self._build_reference_images()
        if references:
            payload["referenceImages"] = references

        return payload

    def build_artifact(self, location: str) -> ImageUrlArtifact:
        return ImageUrlArtifact(value=location, name="runwayml_generated_image")
