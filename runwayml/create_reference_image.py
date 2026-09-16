import asyncio
import hashlib
import io
from typing import Any

from artifacts import (
    MAX_REFERENCE_ASPECT_RATIO,
    MIN_REFERENCE_ASPECT_RATIO,
    ReferenceImageArtifact,
)
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.retained_mode.griptape_nodes import logger
from PIL import Image


class RunwayML_CreateReferenceImage(DataNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "AI/RunwayML"
        self.description = "Creates a reference image with a tag for use in RunwayML text-to-image generation."
        self.metadata["author"] = "Griptape"

        self.add_parameter(
            Parameter(
                name="image",
                input_types=["ImageUrlArtifact", "ImageArtifact", "str"],
                output_type="ImageUrlArtifact",
                type="ImageUrlArtifact",
                default_value=None,
                tooltip="The image to use as a reference. Accepts an image artifact or a URL string.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"clickable_file_browser": True},
            )
        )
        self.add_parameter(
            ParameterString(
                name="tag",
                default_value="",
                tooltip=(
                    "The tag used to address this image from a prompt, for example 'EiffelTower' "
                    "then '@EiffelTower' in the prompt. A tag is generated if left empty."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                placeholder_text="e.g., EiffelTower (optional)",
            )
        )
        self.add_parameter(
            Parameter(
                name="reference_image",
                output_type="ReferenceImageArtifact",
                type="ReferenceImageArtifact",
                default_value=None,
                tooltip="The reference image artifact with its tag. Connect to a Runway Text-to-Image node.",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def _measure_aspect_ratio(self, image: Any) -> float | None:
        """Return the image's aspect ratio, or None when it cannot be measured.

        Measuring is a courtesy check so an unusable reference is caught before a generation
        is paid for. Failing to measure is not itself a reason to reject the image.
        """
        # `File` handles an artifact's value, a bare path, and a `{inputs}/...` macro alike, so
        # every shape the parameter declares is measurable -- not just `ImageUrlArtifact`.
        source = getattr(image, "value", image)
        try:
            image_bytes = File(str(source)).read_bytes()
            with Image.open(io.BytesIO(image_bytes)) as img:
                return img.width / img.height
        except (FileLoadError, OSError, ValueError, ZeroDivisionError) as e:
            logger.debug("%s: could not measure aspect ratio: %s", self.name, e)
            return None

    async def aprocess(self) -> None:
        image = self.get_parameter_value("image")
        tag = str(self.get_parameter_value("tag") or "").strip()

        if not image:
            msg = (
                f"Attempted to create a reference image on '{self.name}'. Failed because no image is set. "
                "Connect an image or choose a file."
            )
            raise ValueError(msg)

        aspect_ratio = None
        if isinstance(image, dict):
            # `meta` can be present but null, and `aspectRatio` can be a string, so neither
            # is trusted to be a usable number.
            meta = image.get("meta")
            candidate = meta.get("aspectRatio") if isinstance(meta, dict) else None
            aspect_ratio = candidate if isinstance(candidate, (int, float)) else None
        else:
            # Off the loop: measuring downloads the image and PIL-decodes it, and the engine
            # runs a node's coroutine on its shared event loop.
            aspect_ratio = await asyncio.to_thread(self._measure_aspect_ratio, image)

        # An out-of-range reference is rejected outright rather than passed on, because
        # RunwayML refuses it and the resulting error names the generation node instead of
        # the image that caused it.
        if aspect_ratio is not None and not MIN_REFERENCE_ASPECT_RATIO <= aspect_ratio <= MAX_REFERENCE_ASPECT_RATIO:
            msg = (
                f"Attempted to create a reference image on '{self.name}'. Failed because its aspect ratio "
                f"is {aspect_ratio:.2f}, outside the {MIN_REFERENCE_ASPECT_RATIO}-{MAX_REFERENCE_ASPECT_RATIO} "
                "range RunwayML accepts. Crop the image to something closer to square."
            )
            raise ValueError(msg)

        if not tag:
            # A digest, not `hash()`: str hashing is salted per interpreter, so a hash-derived
            # tag changes on every engine restart and a prompt written against it silently
            # stops matching. Tags must also start with a letter, hence the prefix.
            digest = hashlib.sha256(self.name.encode("utf-8")).hexdigest()[:8]
            tag = f"ref_{digest}"
            logger.info("%s: no tag given, using '%s'", self.name, tag)

        self.parameter_output_values["reference_image"] = ReferenceImageArtifact(
            image=image, tag=tag, name=f"reference_{tag}"
        )
