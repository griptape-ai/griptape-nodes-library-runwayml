from typing import Optional
from griptape.artifacts import ImageArtifact, ImageUrlArtifact, BaseArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.retained_mode.griptape_nodes import logger
import requests
from PIL import Image
import io

class ReferenceImageArtifact(BaseArtifact):
    """
    A custom artifact that combines an image with a reference tag.
    Used for RunwayML text-to-image generation with reference images.
    """
    
    def __init__(
        self, 
        image: ImageArtifact | ImageUrlArtifact | str,
        tag: str,
        name: Optional[str] = None,
        **kwargs
    ):
        # Create the value dictionary for the base artifact
        value = {
            "image": image,
            "tag": tag.strip() if tag else ""
        }
        super().__init__(value=value, name=name, **kwargs)
        self.image = image
        self.tag = tag.strip() if tag else ""
    
    def to_text(self) -> str:
        """Return a text representation of the reference image."""
        image_type = type(self.image).__name__
        if hasattr(self.image, 'value'):
            image_preview = str(self.image.value)[:50] + "..." if len(str(self.image.value)) > 50 else str(self.image.value)
        else:
            image_preview = str(self.image)[:50] + "..." if len(str(self.image)) > 50 else str(self.image)
        
        return f"ReferenceImage(tag='{self.tag}', image={image_type}({image_preview}))"
    
    def __str__(self) -> str:
        return self.to_text()
    
    def __repr__(self) -> str:
        return self.to_text()


class RunwayML_CreateReferenceImage(DataNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "AI/RunwayML"
        self.description = "Creates a reference image with a tag for use in RunwayML text-to-image generation."
        self.metadata["author"] = "Griptape"
        
        # Input Parameters
        self.add_parameter(
            Parameter(
                name="image",
                input_types=["ImageArtifact", "ImageUrlArtifact", "str"],
                output_type="ImageArtifact",
                type="ImageArtifact",
                default_value=None,
                tooltip="The image to use as a reference. Can be ImageArtifact, ImageUrlArtifact, or URL string. Click to upload a file.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"clickable_file_browser": True}
            )
        )
        
        self.add_parameter(
            Parameter(
                name="tag",
                input_types=["str"],
                output_type="str",
                type="str",
                default_value="",
                tooltip="The tag to reference this image in prompts (e.g., 'EiffelTower'). Use @tag in your prompt. If empty, a default tag will be generated.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"placeholder_text": "e.g., EiffelTower (optional)"}
            )
        )
        
        # Output Parameter
        self.add_parameter(
            Parameter(
                name="reference_image",
                output_type="ReferenceImageArtifact",
                type="ReferenceImageArtifact",
                default_value=None,
                tooltip="The reference image artifact with tag.",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def process(self) -> None:
        image = self.get_parameter_value("image")
        tag = self.get_parameter_value("tag") or ""
        
        logger.info(f"CreateReferenceImage processing: image={type(image).__name__ if image else 'None'}, tag='{tag}'")
        
        if not image:
            logger.warning("CreateReferenceImage: No image provided, returning None")
            self.parameter_output_values["reference_image"] = None
            return
        
        # Note: ImageUrlArtifact doesn't have ratio attribute, so we skip ratio validation
        # If ratio validation is needed, the image would need to be downloaded and analyzed
        if isinstance(image, ImageUrlArtifact):
            logger.info("CreateReferenceImage: Downloading and analyzing ImageUrlArtifact for aspect ratio validation")
            
            # Download and analyze the image
            try:
                response = requests.get(image.value, timeout=10)
                response.raise_for_status()
                
                image_data = io.BytesIO(response.content)
                img = Image.open(image_data)
                aspect_ratio = img.width / img.height
                
                logger.info(f"CreateReferenceImage: Image dimensions: {img.width}x{img.height}, aspect ratio: {aspect_ratio:.2f}")
                
                if not 0.5 <= aspect_ratio <= 2.0:
                    logger.warning(f"CreateReferenceImage: Image aspect ratio {aspect_ratio:.2f} is outside valid range (0.5-2.0)")
                    self.parameter_output_values["reference_image"] = None
                    return
                    
            except Exception as e:
                logger.warning(f"CreateReferenceImage: Failed to download/analyze image: {e}")
                # Continue without validation rather than failing completely
                
        elif isinstance(image, ImageArtifact):
            # ImageArtifact has base64 data, we can analyze it directly
            try:
                import base64
                image_data = io.BytesIO(base64.b64decode(image.base64))
                img = Image.open(image_data)
                aspect_ratio = img.width / img.height
                
                logger.info(f"CreateReferenceImage: Image dimensions: {img.width}x{img.height}, aspect ratio: {aspect_ratio:.2f}")
                
                if not 0.5 <= aspect_ratio <= 2.0:
                    logger.warning(f"CreateReferenceImage: Image aspect ratio {aspect_ratio:.2f} is outside valid range (0.5-2.0)")
                    self.parameter_output_values["reference_image"] = None
                    return
                    
            except Exception as e:
                logger.warning(f"CreateReferenceImage: Failed to analyze ImageArtifact: {e}")
                # Continue without validation rather than failing completely
                
        elif isinstance(image, dict) and "meta" in image and "aspectRatio" in image["meta"]:
            # Handle dict format with meta information
            aspect_ratio = image["meta"]["aspectRatio"]
            logger.info(f"CreateReferenceImage: Using provided aspect ratio: {aspect_ratio}")
            
            if not 0.5 <= aspect_ratio <= 2.0:
                logger.warning(f"CreateReferenceImage: Image aspect ratio {aspect_ratio} is outside valid range (0.5-2.0)")
                self.parameter_output_values["reference_image"] = None
                return
        else:
            logger.info("CreateReferenceImage: Skipping aspect ratio validation for unsupported image type")

        # Generate default tag if empty
        if not tag.strip():
            # Use a simple counter-based default tag
            import time
            timestamp = str(int(time.time() * 1000))[-6:]  # Last 6 digits of timestamp
            tag = f"image_{timestamp}"
            logger.info(f"CreateReferenceImage: Generated default tag '{tag}' for empty tag input")
        
        # Create the reference image artifact
        reference_image = ReferenceImageArtifact(
            image=image,
            tag=tag.strip(),
            name=f"reference_{tag.strip()}"
        )
        
        logger.info(f"CreateReferenceImage: Created ReferenceImageArtifact with tag '{tag.strip()}'")
        self.parameter_output_values["reference_image"] = reference_image 