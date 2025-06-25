import time
import base64
import runwayml
import requests
import io
from urllib.parse import urlparse
import logging
from typing import Optional
from PIL import Image

from griptape.artifacts import TextArtifact, UrlArtifact, ImageArtifact, ImageUrlArtifact, ErrorArtifact, BaseArtifact
from griptape_nodes.traits.options import Options

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterGroup, ParameterList
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.retained_mode.griptape_nodes import logger, GriptapeNodes


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

# Create a file logger for debugging
debug_logger = logging.getLogger('runwayml_debug')
debug_logger.setLevel(logging.INFO)
if not debug_logger.handlers:
    file_handler = logging.FileHandler('/tmp/runwayml_debug.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    debug_logger.addHandler(file_handler)

SERVICE = "RunwayML"
API_KEY_ENV_VAR = "RUNWAYML_API_SECRET"
DEFAULT_MODEL = "gen4_image"

# Text-to-image specific ratio values
RUNWAY_TEXT_TO_IMAGE_RATIOS = [
    "1920:1080",
    "1080:1920",
    "1024:1024",
    "1360:768",
    "1080:1080",
    "1168:880",
    "1440:1080",
    "1080:1440",
    "1808:768",
    "2112:912",
    "1280:720",
    "720:1280",
    "720:720",
    "960:720",
    "720:960",
    "1680:720"
]
DEFAULT_TEXT_TO_IMAGE_RATIO = "1024:1024"


class RunwayML_TextToImage(ControlNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "AI/RunwayML"
        self.description = "Generates images from text prompts with optional reference images using RunwayML."
        self.metadata["author"] = "Griptape"
        self.metadata["dependencies"] = {"pip_dependencies": ["runwayml", "requests", "Pillow"]}

        # Main Prompt Group
        with ParameterGroup(name="Prompt") as prompt_group:
            Parameter(
                name="prompt_text",
                input_types=["str", "TextArtifact"],
                output_type="str",
                type="str",
                default_value="",
                tooltip="Text prompt describing the desired image. Use @tagname to reference images from connected ReferenceImageArtifact instances (e.g., '@EiffelTower painted in the style of @StarryNight'). To use an image for styling, add it as a reference image with a tag and reference it in this prompt.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"multiline": True, "placeholder_text": "e.g., @EiffelTower painted in the style of @StarryNight"},
            )
        self.add_node_element(prompt_group)

        # Reference Images Group
        with ParameterGroup(name="Reference Images") as ref_images_group:
            ParameterList(
                name="reference_images",
                input_types=["ReferenceImageArtifact"],
                output_type="ReferenceImageArtifact",
                type="ReferenceImageArtifact",
                default_value=None,
                tooltip="Reference images with tags for the generation. Connect from 'Create Reference Image' nodes or other sources of ReferenceImageArtifact.",
                allowed_modes={ParameterMode.INPUT},
            )
        self.add_node_element(ref_images_group)

        # Generation Settings Group
        with ParameterGroup(name="Generation Settings") as gen_settings_group:
            Parameter(
                name="model",
                input_types=["str"],
                output_type="str",
                type="str",
                default_value=DEFAULT_MODEL,
                tooltip="RunwayML model to use for generation.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["gen4_image"])}
            )
            Parameter(
                name="ratio",
                input_types=["str"],
                output_type="str",
                type="str",
                default_value=DEFAULT_TEXT_TO_IMAGE_RATIO,
                tooltip="Aspect ratio for the output image.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=RUNWAY_TEXT_TO_IMAGE_RATIOS)}
            )
            Parameter(
                name="seed",
                input_types=["int"],
                output_type="int",
                type="int",
                default_value=0,
                tooltip="Seed for generation. 0 for random.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        self.add_node_element(gen_settings_group)

        # Output Parameters
        self.add_parameter(
            Parameter(
                name="image_output",
                output_type="ImageUrlArtifact",
                type="ImageUrlArtifact",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Output URL of the generated image.",
                ui_options={"placeholder_text": "", "is_full_width": True, "pulse_on_run": True}
            )
        )
        self.add_parameter(
            Parameter(
                name="task_id_output",
                output_type="str",
                type="str",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="The Task ID of the generation job from RunwayML.",
                ui_options={"placeholder_text": ""}
            )
        )

    def _get_image_data_uri(self, image_input) -> str | None:
        """Convert various image input types to data URI or URL format."""
        logger.info(f"RunwayML T2I: _get_image_data_uri called with type: {type(image_input).__name__}")
        
        if not image_input:
            logger.warning("RunwayML T2I: _get_image_data_uri received None/empty input")
            return None

        # Supported formats by RunwayML
        SUPPORTED_FORMATS = ["image/png", "image/jpeg", "image/jpg", "image/webp"]
        
        def _validate_and_convert_data_uri(data_uri: str) -> str:
            """Validate data URI format and convert unsupported formats to PNG."""
            if not data_uri.startswith("data:image"):
                logger.warning(f"RunwayML T2I: Invalid data URI format: {data_uri[:50]}...")
                return None
            
            # Extract media type
            try:
                header, base64_data = data_uri.split(";base64,", 1)
                media_type = header.replace("data:", "")
                
                if media_type in SUPPORTED_FORMATS:
                    return data_uri
                
                logger.info(f"RunwayML T2I: Converting unsupported format {media_type} to PNG")
                
                # Convert unsupported format to PNG
                # Decode base64 to bytes
                image_bytes = base64.b64decode(base64_data)
                
                # Open with PIL and convert to PNG
                with Image.open(io.BytesIO(image_bytes)) as img:
                    # Convert to RGB if necessary (for formats that might have transparency)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        # Keep transparency for PNG
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                    elif img.mode not in ('RGB', 'RGBA'):
                        img = img.convert('RGB')
                    
                    # Save as PNG
                    output_buffer = io.BytesIO()
                    img.save(output_buffer, format='PNG')
                    output_buffer.seek(0)
                    
                    # Encode back to base64
                    png_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
                    converted_uri = f"data:image/png;base64,{png_base64}"
                    
                    logger.info(f"RunwayML T2I: Successfully converted {media_type} to PNG")
                    return converted_uri
                    
            except Exception as e:
                logger.error(f"RunwayML T2I: Failed to convert data URI format: {e}")
                return None

        if isinstance(image_input, ImageArtifact):
            logger.info(f"RunwayML T2I: Processing ImageArtifact - media_type: {getattr(image_input, 'media_type', 'N/A')}, has_base64: {bool(getattr(image_input, 'base64', None))}")
            media_type = image_input.media_type or "image/png"
            if not image_input.base64.startswith(f"data:{media_type};base64,"):
                data_uri = f"data:{media_type};base64,{image_input.base64}"
            else:
                data_uri = image_input.base64
            
            # Validate and convert if necessary
            validated_uri = _validate_and_convert_data_uri(data_uri)
            if not validated_uri:
                logger.warning(f"RunwayML T2I: Failed to validate/convert ImageArtifact")
                return None
            
            # Check size limit (5MB for encoded data URI)
            if len(validated_uri.encode('utf-8')) > 5 * 1024 * 1024:
                logger.warning(f"RunwayML T2I: ImageArtifact data URI exceeds 5MB limit ({len(validated_uri.encode('utf-8')) / (1024*1024):.1f}MB)")
                return None
            
            logger.info(f"RunwayML T2I: Successfully processed ImageArtifact ({len(validated_uri)} chars)")
            return validated_uri
        elif isinstance(image_input, ImageUrlArtifact):
            logger.info(f"RunwayML T2I: Processing ImageUrlArtifact - value: {getattr(image_input, 'value', 'N/A')[:100]}...")
            url_value = image_input.value
            if url_value.startswith("data:image"):
                validated_uri = _validate_and_convert_data_uri(url_value)
                if validated_uri:
                    return validated_uri
                else:
                    logger.warning(f"RunwayML T2I: Failed to validate/convert ImageUrlArtifact data URI")
                    return None
            
            parsed_url = urlparse(url_value)
            if parsed_url.scheme == "http" and (parsed_url.hostname == "localhost" or parsed_url.hostname == "127.0.0.1"):
                logger.info(f"RunwayML T2I: Converting local HTTP URL to base64 data URI: {url_value}")
                try:
                    response = requests.get(url_value, timeout=10)
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "image/png")
                    base64_data = base64.b64encode(response.content).decode("utf-8")
                    data_uri = f"data:{content_type};base64,{base64_data}"
                    
                    # Validate and convert the generated data URI
                    validated_uri = _validate_and_convert_data_uri(data_uri)
                    if not validated_uri:
                        logger.warning(f"RunwayML T2I: Failed to validate/convert local URL data URI")
                        return None
                    
                    # Check size limit (5MB for encoded data URI)
                    if len(validated_uri.encode('utf-8')) > 5 * 1024 * 1024:
                        logger.warning(f"RunwayML T2I: Converted data URI exceeds 5MB limit ({len(validated_uri.encode('utf-8')) / (1024*1024):.1f}MB)")
                        return None
                    
                    return validated_uri
                except Exception as e:
                    logger.error(f"RunwayML T2I: Failed to convert local URL {url_value} to base64: {e}")
                    return None
            elif parsed_url.scheme == "https":
                logger.info(f"RunwayML T2I: Using public HTTPS URL for image: {url_value}")
                return url_value
            else:
                logger.warning(f"RunwayML T2I: ImageUrlArtifact with non-HTTPS/non-local-HTTP URL provided: {url_value}")
                return url_value

        elif isinstance(image_input, str):
            logger.info(f"RunwayML T2I: Processing string input: {image_input[:100]}...")
            if image_input.strip().startswith("data:image"):
                validated_uri = _validate_and_convert_data_uri(image_input.strip())
                if validated_uri:
                    return validated_uri
                else:
                    logger.warning(f"RunwayML T2I: Failed to validate/convert string data URI")
                    return None
            
            parsed_url = urlparse(image_input.strip())
            if parsed_url.scheme == "http" and (parsed_url.hostname == "localhost" or parsed_url.hostname == "127.0.0.1"):
                logger.info(f"RunwayML T2I: Converting local HTTP URL string to base64 data URI: {image_input.strip()}")
                try:
                    response = requests.get(image_input.strip(), timeout=10)
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "image/png")
                    base64_data = base64.b64encode(response.content).decode("utf-8")
                    data_uri = f"data:{content_type};base64,{base64_data}"
                    
                    # Validate and convert the generated data URI
                    validated_uri = _validate_and_convert_data_uri(data_uri)
                    if not validated_uri:
                        logger.warning(f"RunwayML T2I: Failed to validate/convert local URL data URI")
                        return None
                    
                    # Check size limit (5MB for encoded data URI)
                    if len(validated_uri.encode('utf-8')) > 5 * 1024 * 1024:
                        logger.warning(f"RunwayML T2I: Converted data URI exceeds 5MB limit ({len(validated_uri.encode('utf-8')) / (1024*1024):.1f}MB)")
                        return None
                    
                    return validated_uri
                except Exception as e:
                    logger.error(f"RunwayML T2I: Failed to convert local URL string {image_input.strip()} to base64: {e}")
                    return None
            elif parsed_url.scheme == "https":
                logger.info(f"RunwayML T2I: Using public HTTPS URL string for image: {image_input.strip()}")
                return image_input.strip()
            else:
                logger.warning(f"RunwayML T2I: String input is not a data URI, HTTPS URL, or local HTTP URL: {image_input.strip()}")
                return image_input.strip()

        elif isinstance(image_input, dict):
            logger.info(f"RunwayML T2I: Processing dictionary representation of image artifact")
            # Handle dictionary representation of ImageUrlArtifact (from file upload)
            if "value" in image_input and "type" in image_input:
                if image_input["type"] == "ImageUrlArtifact":
                    url_value = image_input["value"]
                    logger.info(f"RunwayML T2I: Dictionary contains ImageUrlArtifact with URL: {url_value}")
                    
                    if url_value.startswith("data:image"):
                        return url_value
                    
                    parsed_url = urlparse(url_value)
                    if parsed_url.scheme == "http" and (parsed_url.hostname == "localhost" or parsed_url.hostname == "127.0.0.1"):
                        logger.info(f"RunwayML T2I: Converting local HTTP URL from dict to base64 data URI: {url_value}")
                        try:
                            response = requests.get(url_value, timeout=10)
                            response.raise_for_status()
                            content_type = response.headers.get("Content-Type", "image/png")
                            base64_data = base64.b64encode(response.content).decode("utf-8")
                            data_uri = f"data:{content_type};base64,{base64_data}"
                            
                            # Validate and convert the generated data URI
                            validated_uri = _validate_and_convert_data_uri(data_uri)
                            if not validated_uri:
                                logger.warning(f"RunwayML T2I: Failed to validate/convert local URL data URI")
                                return None
                            
                            # Check size limit (5MB for encoded data URI)
                            if len(validated_uri.encode('utf-8')) > 5 * 1024 * 1024:
                                logger.warning(f"RunwayML T2I: Converted data URI exceeds 5MB limit ({len(validated_uri.encode('utf-8')) / (1024*1024):.1f}MB)")
                                return None
                            
                            logger.info(f"RunwayML T2I: Successfully converted dict URL to data URI ({len(validated_uri)} chars)")
                            return validated_uri
                        except Exception as e:
                            logger.error(f"RunwayML T2I: Failed to convert dict URL {url_value} to base64: {e}")
                            return None
                    elif parsed_url.scheme == "https":
                        logger.info(f"RunwayML T2I: Using public HTTPS URL from dict: {url_value}")
                        return url_value
                    else:
                        logger.warning(f"RunwayML T2I: Dict URL is not HTTPS or local HTTP: {url_value}")
                        return url_value
                elif image_input["type"] == "ImageArtifact":
                    logger.info(f"RunwayML T2I: Dictionary contains ImageArtifact")
                    # Handle dictionary representation of ImageArtifact
                    if "base64" in image_input:
                        media_type = image_input.get("media_type", "image/png")
                        base64_data = image_input["base64"]
                        
                        if not base64_data.startswith(f"data:{media_type};base64,"):
                            data_uri = f"data:{media_type};base64,{base64_data}"
                        else:
                            data_uri = base64_data
                        
                        # Validate and convert the data URI
                        validated_uri = _validate_and_convert_data_uri(data_uri)
                        if not validated_uri:
                            logger.warning(f"RunwayML T2I: Failed to validate/convert dict ImageArtifact data URI")
                            return None
                        
                        # Check size limit (5MB for encoded data URI)
                        if len(validated_uri.encode('utf-8')) > 5 * 1024 * 1024:
                            logger.warning(f"RunwayML T2I: Dict ImageArtifact data URI exceeds 5MB limit ({len(validated_uri.encode('utf-8')) / (1024*1024):.1f}MB)")
                            return None
                        
                        logger.info(f"RunwayML T2I: Successfully processed dict ImageArtifact ({len(validated_uri)} chars)")
                        return validated_uri
            
            logger.warning(f"RunwayML T2I: Dictionary does not contain expected image artifact structure: {list(image_input.keys())}")
            return None

        logger.warning(f"RunwayML T2I: Unhandled image input type: {type(image_input)}")
        return None

    def _download_and_store_image(self, image_url: str, task_id: str = None) -> ImageUrlArtifact:
        """Download image from URL and store via StaticFilesManager."""
        try:
            logger.info(f"RunwayML T2I: Downloading image from {image_url}")
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Determine file extension from URL or content type
            content_type = response.headers.get("Content-Type", "image/png")
            if "jpeg" in content_type or "jpg" in content_type:
                extension = "jpg"
            elif "png" in content_type:
                extension = "png"
            elif "webp" in content_type:
                extension = "webp"
            else:
                extension = "jpg"  # Default fallback
            
            # Generate filename using task ID if provided, otherwise use timestamp
            if task_id:
                filename = f"runwayml_generated_image_{task_id}.{extension}"
            else:
                timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
                filename = f"runwayml_generated_image_{timestamp}.{extension}"
            
            # Save via StaticFilesManager
            static_url = GriptapeNodes.StaticFilesManager().save_static_file(response.content, filename)
            
            return ImageUrlArtifact(value=static_url, name="runwayml_generated_image")
            
        except Exception as e:
            logger.error(f"RunwayML T2I: Failed to download and store image: {e}")
            # Fallback to original URL if download fails
            return ImageUrlArtifact(value=image_url, name="runwayml_image")

    def validate_node(self) -> list[Exception] | None:
        errors = []
        api_key = self.get_config_value(service=SERVICE, value=API_KEY_ENV_VAR)

        if not api_key:
            errors.append(ValueError(f"RunwayML API key not found. Set {API_KEY_ENV_VAR} in environment variables or Griptape Cloud."))

        prompt_val = self.get_parameter_value("prompt_text")
        if not prompt_val or not str(prompt_val).strip():
            errors.append(ValueError("Text prompt ('prompt_text') cannot be empty."))

        # Validate reference images and tags
        ref_images_list = self.get_parameter_value("reference_images") or []
        
        # Debug logging to see what we're actually receiving
        logger.info(f"RunwayML T2I validation: received {len(ref_images_list)} reference images")
        for i, item in enumerate(ref_images_list):
            logger.info(f"RunwayML T2I validation: item {i+1} type={type(item).__name__}, value={getattr(item, 'value', 'N/A')}")
        
        # Validate each reference image artifact
        for i, ref_artifact in enumerate(ref_images_list):
            if not ref_artifact:
                errors.append(ValueError(f"Reference image {i+1} is missing."))
                continue
            
            # Check if it's a ReferenceImageArtifact or a dictionary representation
            if hasattr(ref_artifact, 'image') and hasattr(ref_artifact, 'tag'):
                # ReferenceImageArtifact instance with direct attributes
                logger.info(f"RunwayML T2I validation: Reference image {i+1} has direct image and tag attributes")
                image = ref_artifact.image
                tag = ref_artifact.tag
            elif hasattr(ref_artifact, 'value') and isinstance(ref_artifact.value, dict) and "image" in ref_artifact.value:
                # ReferenceImageArtifact with data in value attribute
                logger.info(f"RunwayML T2I validation: Reference image {i+1} has image in value attribute")
                image = ref_artifact.value["image"]
                tag = ref_artifact.value.get("tag", "")  # Tag is optional
            elif isinstance(ref_artifact, dict) and "image" in ref_artifact:
                # Dictionary representation (likely from serialization)
                logger.info(f"RunwayML T2I validation: Reference image {i+1} is a dictionary, extracting image and tag")
                image = ref_artifact["image"]
                tag = ref_artifact.get("tag", "")  # Tag is optional
            else:
                logger.error(f"RunwayML T2I validation: Reference image {i+1} is type {type(ref_artifact).__name__}, expected object with image data")
                errors.append(ValueError(f"Reference image {i+1} must have image data."))
                continue
            
            logger.info(f"RunwayML T2I validation: Reference image {i+1} - image type: {type(image).__name__}, tag: '{tag}'")
            
            if not image:
                errors.append(ValueError(f"Reference image {i+1} (tag: '{tag}') is missing the image."))
                continue
            
            if not tag or not str(tag).strip():
                # Empty tag is now allowed - we'll use a default tag
                tag = f"image_{i+1}"
                logger.info(f"RunwayML T2I: Reference image {i+1} has empty tag, using default: '{tag}'")
            
            logger.info(f"RunwayML T2I validation: About to process image {i+1} with _get_image_data_uri")
            image_uri = self._get_image_data_uri(image)
            logger.info(f"RunwayML T2I validation: _get_image_data_uri returned: {bool(image_uri)} (length: {len(image_uri) if image_uri else 0})")
            
            if not image_uri:
                logger.error(f"RunwayML T2I validation: Failed to convert image {i+1} to data URI. Image type: {type(image).__name__}")
                if hasattr(image, 'value'):
                    logger.error(f"RunwayML T2I validation: Image value preview: {str(image.value)[:100]}...")
                if hasattr(image, 'base64'):
                    logger.error(f"RunwayML T2I validation: Image has base64 attribute: {bool(image.base64)}")
                if hasattr(image, 'media_type'):
                    logger.error(f"RunwayML T2I validation: Image media_type: {getattr(image, 'media_type', 'N/A')}")
                
                # Check if it's a size issue
                if isinstance(image, ImageArtifact):
                    media_type = image.media_type or "image/png"
                    if not image.base64.startswith(f"data:{media_type};base64,"):
                        test_uri = f"data:{media_type};base64,{image.base64}"
                    else:
                        test_uri = image.base64
                    
                    if len(test_uri.encode('utf-8')) > 5 * 1024 * 1024:
                        errors.append(ValueError(f"Reference image {i+1} (tag: '{tag}') is too large ({len(test_uri.encode('utf-8')) / (1024*1024):.1f}MB). RunwayML has a 5MB limit for data URIs (~3.3MB unencoded). Try using smaller images or HTTPS URLs."))
                        continue
                
                errors.append(ValueError(f"Reference image {i+1} (tag: '{tag}') is invalid or cannot be processed."))

        return errors if errors else None

    def process(self) -> AsyncResult:
        validation_errors = self.validate_node()
        if validation_errors:
            error_message = "; ".join(str(e) for e in validation_errors)
            logger.error(f"RunwayML T2I validation failed: {error_message}")
            self.publish_update_to_parameter("image_output", ErrorArtifact(error_message))
            raise ValueError(f"Validation failed: {error_message}")

        def generate_image_async() -> ImageUrlArtifact | ErrorArtifact:
            try:
                # Clear previous output to prevent showing stale results
                self.publish_update_to_parameter("image_output", None)
                
                # Get parameter values - create fresh copies to ensure idempotency
                prompt_text = str(self.get_parameter_value("prompt_text") or "").strip()
                model_name = str(self.get_parameter_value("model") or DEFAULT_MODEL)
                
                ratio_input = self.get_parameter_value("ratio")
                if isinstance(ratio_input, tuple) and len(ratio_input) == 2:
                    ratio_val = str(ratio_input[1])
                else:
                    ratio_val = str(ratio_input or DEFAULT_TEXT_TO_IMAGE_RATIO)

                seed_val = self.get_parameter_value("seed") or 0

                # Get fresh copies of reference data to ensure idempotency
                ref_images_list_raw = self.get_parameter_value("reference_images")
                
                # Create fresh lists to avoid any state modification issues
                ref_images_list = list(ref_images_list_raw) if ref_images_list_raw else []
                
                # Build reference images array - use completely fresh local variables
                processed_reference_images = []

                logger.info(f"RunwayML T2I: Raw reference images list: {len(ref_images_list)} images")

                # Log to file for detailed debugging
                debug_logger.info("=== REFERENCE IMAGES PROCESSING START ===")
                debug_logger.info(f"Raw reference images list count: {len(ref_images_list)}")
                for i, ref_artifact in enumerate(ref_images_list):
                    if hasattr(ref_artifact, 'image') and hasattr(ref_artifact, 'tag'):
                        debug_logger.info(f"Image {i+1}: type={type(ref_artifact).__name__}, tag='{ref_artifact.tag}', image_type={type(ref_artifact.image).__name__}")
                    elif isinstance(ref_artifact, dict):
                        tag = ref_artifact.get("tag", "N/A")
                        image_type = type(ref_artifact.get("image", None)).__name__ if ref_artifact.get("image") else "None"
                        debug_logger.info(f"Image {i+1}: type=dict, tag='{tag}', image_type={image_type}")
                    else:
                        debug_logger.info(f"Image {i+1}: type={type(ref_artifact).__name__}, value={getattr(ref_artifact, 'value', 'N/A')}")

                # Process reference images if available
                if ref_images_list:
                    logger.info("RunwayML T2I: Processing reference images...")
                    for i, ref_artifact in enumerate(ref_images_list):
                        if not ref_artifact:
                            logger.warning(f"RunwayML T2I: Skipping reference image {i+1} - missing artifact")
                            continue
                        
                        # Check if it's a ReferenceImageArtifact or a dictionary representation
                        if hasattr(ref_artifact, 'image') and hasattr(ref_artifact, 'tag'):
                            # ReferenceImageArtifact instance with direct attributes
                            image = ref_artifact.image
                            tag = ref_artifact.tag
                        elif hasattr(ref_artifact, 'value') and isinstance(ref_artifact.value, dict) and "image" in ref_artifact.value:
                            # ReferenceImageArtifact with data in value attribute
                            logger.info(f"RunwayML T2I: Reference image {i+1} has image in value attribute")
                            image = ref_artifact.value["image"]
                            tag = ref_artifact.value.get("tag", "")  # Tag is optional
                        elif isinstance(ref_artifact, dict) and "image" in ref_artifact:
                            # Dictionary representation (likely from serialization)
                            logger.info(f"RunwayML T2I: Reference image {i+1} is a dictionary, extracting image and tag")
                            image = ref_artifact["image"]
                            tag = ref_artifact.get("tag", "")  # Tag is optional
                        else:
                            logger.warning(f"RunwayML T2I: Skipping reference image {i+1} - type {type(ref_artifact).__name__} doesn't have required attributes")
                            continue
                        
                        if not image:
                            logger.warning(f"RunwayML T2I: Skipping reference image {i+1} (tag '{tag}') - missing image")
                            continue
                        
                        if not tag or not str(tag).strip():
                            # Empty tag is now allowed - we'll use a default tag
                            tag = f"image_{i+1}"
                            logger.info(f"RunwayML T2I: Reference image {i+1} has empty tag, using default: '{tag}'")
                        
                        logger.info(f"RunwayML T2I: Processing reference image {i+1} with tag '{tag}': image={type(image).__name__}")
                        debug_logger.info(f"Processing image {i+1} with tag '{tag}': type={type(image).__name__}")
                        
                        logger.info(f"RunwayML T2I: Image artifact type: {type(image).__name__}, value preview: {str(image.value)[:100] if hasattr(image, 'value') else 'N/A'}...")
                        image_uri = self._get_image_data_uri(image)
                        
                        debug_logger.info(f"Image {i+1} (tag '{tag}') conversion result: success={bool(image_uri)}, length={len(image_uri) if image_uri else 0}")
                        if image_uri:
                            debug_logger.info(f"Image {i+1} (tag '{tag}') URI preview: {image_uri[:100]}...")
                        
                        if image_uri:
                            uri_preview = image_uri[:50] + "..." + image_uri[-20:] if len(image_uri) > 70 else image_uri
                            logger.info(f"RunwayML T2I: Converted image URI ({len(image_uri)} chars): {uri_preview}")
                        else:
                            logger.info(f"RunwayML T2I: Failed to convert image URI")
                        
                        if image_uri and tag and str(tag).strip():
                            processed_reference_images.append({
                                "uri": image_uri,
                                "tag": str(tag).strip()
                            })
                            logger.info(f"RunwayML T2I: Added reference image with tag '{tag}'")
                            debug_logger.info(f"Added image {i+1} with tag '{tag}' to processed_reference_images array")
                        else:
                            logger.warning(f"RunwayML T2I: Skipped reference image {i+1} (tag '{tag}') - image_uri={bool(image_uri)}")
                            debug_logger.warning(f"Skipped image {i+1} (tag '{tag}') - image_uri={bool(image_uri)}")

                debug_logger.info(f"Final processed_reference_images array length: {len(processed_reference_images)}")
                debug_logger.info("=== REFERENCE IMAGES PROCESSING END ===")
                debug_logger.info("")

                # Build task payload with fresh data
                task_payload = {
                    "model": model_name,
                    "promptText": prompt_text,
                    "ratio": ratio_val,
                }

                if seed_val and seed_val != 0:
                    task_payload["seed"] = seed_val

                if processed_reference_images:
                    task_payload["referenceImages"] = processed_reference_images
                    logger.info(f"RunwayML T2I: Using {len(processed_reference_images)} reference images")
                    debug_logger.info(f"Final payload referenceImages: {processed_reference_images}")
                    debug_logger.info(f"Payload keys: {list(task_payload.keys())}")
                else:
                    logger.info(f"RunwayML T2I: No valid reference images, not including referenceImages in payload")

                logger.info(f"RunwayML T2I: Creating task with payload keys: {list(task_payload.keys())}")
                debug_logger.info(f"Full payload structure (without data): {dict((k, v if k != 'referenceImages' else f'[{len(v)} items]') for k, v in task_payload.items())}")

                # Get API key
                api_key = self.get_config_value(service=SERVICE, value=API_KEY_ENV_VAR)
                
                # Create text-to-image task using direct HTTP request
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-Runway-Version": "2024-11-06"
                }
                
                # Debug logging
                debug_payload = {}
                for k, v in task_payload.items():
                    if k == 'referenceImages':
                        debug_payload[k] = f'[{len(v)} items]' if isinstance(v, list) else v
                    else:
                        debug_payload[k] = v
                logger.info(f"RunwayML T2I: Sending request to API with payload: {debug_payload}")
                
                response = requests.post(
                    "https://api.dev.runwayml.com/v1/text_to_image",
                    json=task_payload,
                    headers=headers,
                    timeout=60
                )
                
                if response.status_code != 200:
                    error_body = response.text
                    logger.error(f"RunwayML T2I: API returned {response.status_code}: {error_body}")
                    raise ValueError(f"RunwayML API Error ({response.status_code}): {error_body}")
                
                response.raise_for_status()
                task_response = response.json()
                task_id = task_response.get("id")
                if not task_id:
                    raise ValueError(f"No task ID returned from RunwayML API. Response: {task_response}")
                
                self.publish_update_to_parameter("task_id_output", task_id)
                logger.info(f"RunwayML T2I: Task created with ID: {task_id}")

                # Poll the task until it's complete
                max_retries = 120  # 120 retries * 10 seconds = 20 minutes timeout
                retry_delay = 10  # seconds

                for attempt in range(max_retries):
                    time.sleep(retry_delay)
                    
                    # Get task status using direct HTTP request
                    status_response = requests.get(
                        f"https://api.dev.runwayml.com/v1/tasks/{task_id}",
                        headers=headers,
                        timeout=30
                    )
                    status_response.raise_for_status()
                    task_status = status_response.json()
                    status = task_status.get("status")
                    
                    logger.info(f"RunwayML T2I generation status (Task ID: {task_id}): {status} (Attempt {attempt + 1}/{max_retries})")

                    if status == 'SUCCEEDED':
                        image_url = None
                        output = task_status.get("output")
                        if output:
                            if isinstance(output, list) and len(output) > 0:
                                output_item = output[0]
                                if isinstance(output_item, dict) and "url" in output_item:
                                    image_url = output_item["url"]
                                elif isinstance(output_item, str) and output_item.startswith(('http://', 'https://')):
                                    image_url = output_item
                            elif isinstance(output, dict) and "url" in output:
                                image_url = output["url"]
                            elif isinstance(output, str) and output.startswith(('http://', 'https://')):
                                image_url = output

                        if image_url:
                            logger.info(f"RunwayML T2I generation succeeded: {image_url}")
                            image_artifact = self._download_and_store_image(image_url, task_id)
                            self.publish_update_to_parameter("image_output", image_artifact)
                            return image_artifact
                        else:
                            logger.error(f"RunwayML T2I task SUCCEEDED but no output URL found. Output structure: {output}")
                            err_msg = "RunwayML T2I task SUCCEEDED but no output URL found."
                            self.publish_update_to_parameter("image_output", ErrorArtifact(err_msg))
                            return ErrorArtifact(err_msg)
                    
                    elif status == 'FAILED':
                        error_msg = f"RunwayML T2I generation failed (Task ID: {task_id})."
                        error_detail = task_status.get("error")
                        if error_detail:
                            error_msg += f" Reason: {error_detail}"
                        logger.error(error_msg)
                        self.publish_update_to_parameter("image_output", ErrorArtifact(error_msg))
                        return ErrorArtifact(error_msg)

                timeout_msg = f"RunwayML T2I task (ID: {task_id}) timed out after {max_retries * retry_delay} seconds."
                logger.error(timeout_msg)
                self.publish_update_to_parameter("image_output", ErrorArtifact(timeout_msg))
                return ErrorArtifact(timeout_msg)

            except Exception as e:
                error_message = f"RunwayML T2I unexpected error: {type(e).__name__} - {e}"
                
                # Handle specific API errors
                if hasattr(e, 'status_code') and e.status_code == 413:
                    error_message = "Image too large! RunwayML has a 5MB limit for data URIs (~3.3MB unencoded). Try using smaller images or HTTPS URLs instead of local files."
                elif hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 413:
                    error_message = "Image too large! RunwayML has a 5MB limit for data URIs (~3.3MB unencoded). Try using smaller images or HTTPS URLs instead of local files."
                elif "413" in str(e) or "Request Entity Too Large" in str(e):
                    error_message = "Image too large! RunwayML has a 5MB limit for data URIs (~3.3MB unencoded). Try using smaller images or HTTPS URLs instead of local files."
                elif "referenceImages: Invalid" in str(e):
                    error_message = "Invalid reference images! Check that all reference images are valid URLs or data URIs, and that each has a corresponding tag. Empty reference image arrays may also cause this error."
                elif hasattr(e, 'status') and hasattr(e, 'reason') and hasattr(e, 'body'):
                    error_message = f"RunwayML API Error: Status {getattr(e, 'status', 'N/A')} - Reason: {getattr(e, 'reason', 'N/A')} - Body: {getattr(e, 'body', 'N/A')}"
                
                logger.exception(error_message)
                self.publish_update_to_parameter("image_output", ErrorArtifact(error_message))
                return ErrorArtifact(error_message)

        yield generate_image_async 