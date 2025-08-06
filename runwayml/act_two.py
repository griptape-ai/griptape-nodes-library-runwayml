import time
import base64
import json
import os
import subprocess
import tempfile
import runwayml
import requests
from urllib.parse import urlparse
from typing import Any

from griptape.artifacts import TextArtifact, ImageUrlArtifact, ErrorArtifact
from griptape_nodes.traits.options import Options

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterGroup
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode, BaseNode
from griptape_nodes.retained_mode.griptape_nodes import logger

SERVICE = "RunwayML"
API_KEY_ENV_VAR = "RUNWAYML_API_SECRET"
DEFAULT_MODEL = "act_two"

# Character type options
CHARACTER_TYPES = ["image", "video"]
DEFAULT_CHARACTER_TYPE = "image"

# Aspect ratios supported by Runway API
ASPECT_RATIOS = [
    "1280:720", "720:1280", "1104:832", "832:1104", "960:960", "1584:672"
]
DEFAULT_ASPECT_RATIO = "1280:720"

# Expression intensity options
EXPRESSION_INTENSITY_OPTIONS = [1, 2, 3, 4, 5]
DEFAULT_EXPRESSION_INTENSITY = 3


class VideoUrlArtifact(ImageUrlArtifact):
    """
    Artifact that contains a URL to a video.
    """

    def __init__(self, url: str, name: str | None = None):
        super().__init__(value=url, name=name or self.__class__.__name__)


class RunwayML_ActTwo(ControlNode):
    # Class variable to track last used seed across instances (ComfyUI-style)
    _last_used_seed: int = 12345
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "AI/RunwayML"
        self.description = "Generates a character performance video using RunwayML's Act Two API."
        self.metadata["author"] = "Griptape"
        self.metadata["dependencies"] = {"pip_dependencies": ["runwayml", "requests"]}
            
        # Character Type Group
        with ParameterGroup(name="Character Type") as character_type_group:
            Parameter(
                name="character_type",
                input_types=["str"],
                output_type="str",
                type="str",
                default_value=DEFAULT_CHARACTER_TYPE,
                tooltip="Type of character input to use.",
                allowed_modes={ParameterMode.PROPERTY},
                traits={Options(choices=CHARACTER_TYPES)}
            )
        self.add_node_element(character_type_group)

        # Media Inputs Group
        with ParameterGroup(name="Media Inputs") as media_inputs_group:
            Parameter(
                name="character_image",
                input_types=["ImageArtifact", "ImageUrlArtifact", "str"],
                type="ImageUrlArtifact", 
                tooltip="Input image of the character. Accepts ImageUrlArtifact, a public URL string, or a base64 data URI string.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
                ui_options={"clickable_file_browser": True}
            )
            
            Parameter(
                name="character_video",
                input_types=["VideoUrlArtifact"],
                type="VideoUrlArtifact", 
                tooltip="Character video for the performance. Accepts VideoUrlArtifact, a public URL string, or a base64 data URI string.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT}
            )
            
            Parameter(
                name="reference_video",
                input_types=["VideoArtifact", "VideoUrlArtifact"],
                type="VideoArtifact", 
                tooltip="Reference video for the character. Accepts VideoUrlArtifact, a public URL string, or a base64 data URI string.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT}
            )
        self.add_node_element(media_inputs_group)

        # Settings Group
        with ParameterGroup(name="Settings") as settings_group:
            Parameter(
                name="body_control",
                input_types=["bool"],
                output_type="bool",
                type="bool",
                default_value=True,
                tooltip="[REQUIRED] Whether to enable body control.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}
            )
            
            Parameter(
                name="expression_intensity",
                input_types=["int"],
                output_type="int",
                type="int",
                default_value=DEFAULT_EXPRESSION_INTENSITY,
                tooltip="[REQUIRED] Expression intensity (1-5).",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=EXPRESSION_INTENSITY_OPTIONS)}
            )
            
            Parameter(
                name="ratio",
                input_types=["str"],
                output_type="str",
                type="str",
                default_value=DEFAULT_ASPECT_RATIO,
                tooltip="[REQUIRED] Aspect ratio for the output video.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=ASPECT_RATIOS)}
            )
            
            Parameter(
                name="seed",
                input_types=["int"],
                output_type="int",
                type="int",
                default_value=RunwayML_ActTwo._last_used_seed,
                tooltip="Seed value for reproducible generation",
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT, ParameterMode.OUTPUT},
            )
            Parameter(
                name="seed_control",
                input_types=["str"],
                output_type="str",
                type="str",
                default_value="randomize",
                tooltip="Seed control mode: Fixed (use exact value), Increment (+1 each run), Decrement (-1 each run), Randomize (new random each run)",
                allowed_modes={ParameterMode.PROPERTY},
                traits={Options(choices=["fixed", "increment", "decrement", "randomize"])}
            )
            
            Parameter(
                name="model",
                input_types=["str"],
                output_type="str",
                type="str",
                default_value=DEFAULT_MODEL,
                tooltip="[REQUIRED] RunwayML model to use for generation.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["act_two"])}
            )
            
            Parameter(
                name="public_figure_threshold",
                input_types=["str"],
                output_type="str",
                type="str",
                default_value="auto",
                tooltip="[OPTIONAL] Public figure threshold for content moderation.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["auto", "low"])}
            )
        self.add_node_element(settings_group)

        # Output Parameters
        self.add_parameter(
            Parameter(
                name="video_output",
                output_type="VideoUrlArtifact",
                type="VideoUrlArtifact",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Output URL of the generated video.",
                ui_options={"placeholder_text": "", "is_full_width": False, "pulse_on_run": True}
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
        
        # Initialize parameter visibility based on default character type
        self._update_parameter_visibility(DEFAULT_CHARACTER_TYPE)

    def _update_parameter_visibility(self, character_type: str) -> None:
        """Update parameter visibility based on character type selection."""
        if character_type == "image":
            self.show_parameter_by_name("character_image")
            self.hide_parameter_by_name("character_video")
        elif character_type == "video":
            self.hide_parameter_by_name("character_image")
            self.show_parameter_by_name("character_video")

    def after_value_set(self, parameter: Parameter, value) -> None:
        """Called after a parameter value is set. Handle dynamic UI updates."""
        if parameter.name == "character_type":
            self._update_parameter_visibility(value)
        return super().after_value_set(parameter, value)

    def _get_data_uri(self, param_name: str) -> str | None:
        """
        Extracts and processes data URI from different input types.
        Works with images and videos. Always returns a data URI, never a URL.
        """
        input_value = self.get_parameter_value(param_name)
        
        if not input_value:
            logger.info(f"RunwayML Act Two: No input value for {param_name}")
            return None
        
        logger.info(f"RunwayML Act Two: Input value for {param_name}: type={type(input_value).__name__}, repr={repr(input_value)[:200]}")

        # Get the content type prefix based on parameter name
        is_image = param_name == "character_image"
        content_type_prefix = "data:image" if is_image else "data:video"
        expected_media_type = "image/png" if is_image else "video/mp4"
        logger.info(f"RunwayML Act Two: Processing {param_name} with type {type(input_value).__name__}")
        logger.info(f"RunwayML Act Two: Input value attributes: {[attr for attr in dir(input_value) if not attr.startswith('_')][:10]}")

        # Handle URL artifacts (including VideoUrlArtifact from other modules)
        # Use duck typing - if it has a 'value' attribute, treat it as a URL artifact
        if hasattr(input_value, 'value'):
            logger.info(f"RunwayML Act Two: Detected URL artifact with 'value' attribute")
            url_value = getattr(input_value, 'value', None)
            logger.info(f"RunwayML Act Two: Extracted URL value: {str(url_value)[:100]}...")
            if not url_value:
                logger.warning(f"RunwayML Act Two: URL artifact has no value: {type(input_value).__name__}")
                return None
            if url_value.startswith(content_type_prefix):
                return url_value
            
            # All URLs need to be converted to data URIs
            parsed_url = urlparse(url_value)
            if parsed_url.scheme in ["http", "https"]:
                logger.info(f"RunwayML Act Two: Converting URL to base64 data URI: {url_value}")
                try:
                    response = requests.get(url_value, timeout=30)  # Increased timeout for larger files
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", expected_media_type)
                    base64_data = base64.b64encode(response.content).decode("utf-8")
                    return f"data:{content_type};base64,{base64_data}"
                except Exception as e:
                    logger.error(f"RunwayML Act Two: Failed to convert URL {url_value} to base64: {e}")
                    return None
            else:
                logger.warning(f"RunwayML Act Two: URL artifact with non-HTTP/HTTPS URL provided: {url_value}. Cannot process.")
                return None
                
        # Handle string input (URL or data URI)
        elif isinstance(input_value, str):
            logger.info(f"RunwayML Act Two: Detected string input")
            if input_value.strip().startswith(content_type_prefix):
                return input_value.strip()
            
            # All URLs need to be converted to data URIs
            parsed_url = urlparse(input_value.strip())
            if parsed_url.scheme in ["http", "https"]:
                logger.info(f"RunwayML Act Two: Converting URL string to base64 data URI: {input_value.strip()}")
                try:
                    response = requests.get(input_value.strip(), timeout=30)  # Increased timeout for larger files
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", expected_media_type)
                    base64_data = base64.b64encode(response.content).decode("utf-8")
                    return f"data:{content_type};base64,{base64_data}"
                except Exception as e:
                    logger.error(f"RunwayML Act Two: Failed to convert URL string {input_value.strip()} to base64: {e}")
                    return None
            else:
                logger.warning(f"RunwayML Act Two: String input for {param_name} is not a data URI or valid URL: {input_value.strip()}. Cannot process.")
                return None
                
        # Handle dictionary input
        elif isinstance(input_value, dict):
            logger.info(f"RunwayML Act Two: Detected dictionary input")
            logger.info(f"RunwayML Act Two: received dict for {param_name}: {input_value}")
            input_type = input_value.get("type")
            url_from_dict = input_value.get("value")
            base64_from_dict = input_value.get("base64")
            media_type_from_dict = input_value.get("media_type", "video/mp4")
            
            if ("UrlArtifact" in str(input_type)) and url_from_dict:
                if str(url_from_dict).startswith(content_type_prefix):
                    return str(url_from_dict)
                
                # Convert URL to data URI
                parsed_url = urlparse(str(url_from_dict))
                if parsed_url.scheme in ["http", "https"]:
                    logger.info(f"RunwayML Act Two: Converting dict URL to base64 data URI: {str(url_from_dict)[:50]}...")
                    try:
                        response = requests.get(str(url_from_dict), timeout=30)
                        response.raise_for_status()
                        content_type = response.headers.get("Content-Type", expected_media_type)
                        base64_data = base64.b64encode(response.content).decode("utf-8")
                        return f"data:{content_type};base64,{base64_data}"
                    except Exception as e:
                        logger.error(f"RunwayML Act Two: Failed to convert dict URL to base64: {e}")
                        return None
                else:
                    logger.warning(f"RunwayML Act Two: Dict URL with non-HTTP/HTTPS URL provided: {str(url_from_dict)[:50]}... Cannot process.")
                    return None
            elif base64_from_dict:
                if not str(base64_from_dict).startswith(f"data:{media_type_from_dict};base64,"):
                    return f"data:{media_type_from_dict};base64,{base64_from_dict}"
                return str(base64_from_dict)
            
            logger.warning(f"RunwayML Act Two: received unhandled dict structure for {param_name}: {input_value}")
            return None
            
        logger.warning(f"RunwayML Act Two: Unhandled input type for {param_name}: {type(input_value)}")
        return None

    def validate_node(self) -> list[Exception] | None:
        errors = []
        api_key = self.get_config_value(service=SERVICE, value=API_KEY_ENV_VAR)

        if not api_key:
            errors.append(ValueError(f"RunwayML API key not found. Set {API_KEY_ENV_VAR} in environment variables or Griptape Cloud."))
        
        # Get character type and validate appropriate input
        character_type = self.get_parameter_value("character_type") or DEFAULT_CHARACTER_TYPE
        
        if character_type == "image":
            character_image_uri = self._get_data_uri("character_image")
            if not character_image_uri:
                errors.append(ValueError("Character image is required when character type is 'image'."))
        elif character_type == "video":
            character_video_uri = self._get_data_uri("character_video")
            if not character_video_uri:
                errors.append(ValueError("Character video is required when character type is 'video'."))
        else:
            errors.append(ValueError(f"Invalid character type: {character_type}. Must be 'image' or 'video'."))
        
        # Validate reference video
        reference_video_uri = self._get_data_uri("reference_video")
        
        if not reference_video_uri:
            errors.append(ValueError("Reference video ('reference_video') is required."))
            
        # Validate aspect ratio
        ratio_val = self.get_parameter_value("ratio") 
        if not ratio_val or str(ratio_val) not in ASPECT_RATIOS:
            errors.append(ValueError(f"Valid aspect ratio is required. Supported values are: {', '.join(ASPECT_RATIOS)}"))
        
        # Validate expression intensity
        expression_intensity = self.get_parameter_value("expression_intensity")
        if expression_intensity is None or expression_intensity not in EXPRESSION_INTENSITY_OPTIONS:
            errors.append(ValueError(f"Valid expression intensity (1-5) is required."))
        
        # Validate model
        model_val = str(self.get_parameter_value("model") or "")
        if not model_val or model_val != "act_two":
            errors.append(ValueError("Model must be 'act_two' for character performance."))

        return errors if errors else None

    def process(self) -> AsyncResult:
        validation_errors = self.validate_node()
        if validation_errors:
            error_message = "; ".join(str(e) for e in validation_errors)
            logger.error(f"RunwayML Act Two validation failed: {error_message}")
            self.publish_update_to_parameter("video_output", ErrorArtifact(error_message))
            raise ValueError(f"Validation failed: {error_message}")
        
        # Get parameter values
        character_type = self.get_parameter_value("character_type") or DEFAULT_CHARACTER_TYPE
        
        # Get the appropriate character input based on type
        if character_type == "image":
            character_image_uri = self._get_data_uri("character_image")
            character_video_uri = None
        else:
            character_video_uri = self._get_data_uri("character_video")
            character_image_uri = None
            
        reference_video_uri = self._get_data_uri("reference_video")
        ratio_val = str(self.get_parameter_value("ratio") or DEFAULT_ASPECT_RATIO)
        
        # Handle seed control (ComfyUI-style)
        seed_value = int(self.get_parameter_value("seed") or RunwayML_ActTwo._last_used_seed)
        seed_control = self.get_parameter_value("seed_control") or "randomize"
        
        if seed_control == "fixed":
            actual_seed = seed_value
        elif seed_control == "increment":
            actual_seed = RunwayML_ActTwo._last_used_seed + 1
        elif seed_control == "decrement":
            actual_seed = RunwayML_ActTwo._last_used_seed - 1
        elif seed_control == "randomize":
            import random
            actual_seed = random.randint(0, 2**32 - 1)
        else:
            actual_seed = seed_value  # fallback
        
        # Ensure seed is in valid range for API (0 to 4294967295)
        actual_seed = max(0, min(actual_seed, 4294967295))
        
        # Update last used seed for next run
        RunwayML_ActTwo._last_used_seed = actual_seed
        
        model_val = str(self.get_parameter_value("model") or DEFAULT_MODEL)
        expression_intensity = self.get_parameter_value("expression_intensity") or DEFAULT_EXPRESSION_INTENSITY
        body_control = self.get_parameter_value("body_control")
        if body_control is None:  # If not set, default to True
            body_control = True
        public_figure_threshold = self.get_parameter_value("public_figure_threshold") or "auto"
            
        def generate_character_performance_async() -> VideoUrlArtifact | ErrorArtifact:
            try:
                api_key = self.get_config_value(service=SERVICE, value=API_KEY_ENV_VAR)

                # Build the payload according to the API format
                task_payload = {
                    "character": {
                        "type": character_type,
                        "uri": character_image_uri or character_video_uri
                    },

                    "reference": {
                        "type": "video",
                        "uri": reference_video_uri
                    },
                    "bodyControl": body_control,
                    "expressionIntensity": expression_intensity,
                    "model": model_val,
                    "ratio": ratio_val,
                    "contentModeration": {
                        "publicFigureThreshold": public_figure_threshold
                    }
                }
                
                # Add optional seed if non-zero
                if actual_seed != 0:
                    task_payload["seed"] = actual_seed

                logger.info(f"RunwayML Act Two: Creating task with payload keys: {list(task_payload.keys())}")

                # Make direct API call
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-Runway-Version": "2024-11-06"  # Use latest API version
                }
                
                response = requests.post(
                    "https://api.dev.runwayml.com/v1/character_performance",
                    json=task_payload,
                    headers=headers,
                    timeout=60
                )
                
                if response.status_code != 200:
                    error_body = response.text
                    logger.error(f"RunwayML Act Two: API returned {response.status_code}: {error_body}")
                                    
                    raise ValueError(f"RunwayML API Error ({response.status_code}): {error_body}")
                
                response_data = response.json()
                task_id = response_data.get("id")
                if not task_id:
                    raise ValueError("No task ID returned from RunwayML API")
                
                self.publish_update_to_parameter("task_id_output", task_id)
                logger.info(f"RunwayML Act Two: Task created with ID: {task_id}")

                # Poll the task until it's complete
                max_retries = 120  # 120 retries * 10 seconds = 20 minutes timeout
                retry_delay = 10  # seconds

                for attempt in range(max_retries):
                    time.sleep(retry_delay)
                    
                    # Get task status using direct API call
                    status_response = requests.get(
                        f"https://api.dev.runwayml.com/v1/tasks/{task_id}",
                        headers=headers,
                        timeout=30
                    )
                    
                    if status_response.status_code != 200:
                        raise ValueError(f"Failed to get task status: {status_response.status_code} - {status_response.text}")
                    
                    task_status = status_response.json()
                    status = task_status.get("status")
                    
                    logger.info(f"RunwayML Act Two generation status (Task ID: {task_id}): {status} (Attempt {attempt + 1}/{max_retries})")

                    if status == 'SUCCEEDED':
                        video_url = None
                        output = task_status.get("output")
                        
                        # Extract the video URL from the response
                        if output:
                            if isinstance(output, list) and len(output) > 0:
                                output_item = output[0]
                                if isinstance(output_item, dict) and "url" in output_item:
                                    video_url = output_item["url"]
                                elif isinstance(output_item, str) and output_item.startswith(('http://', 'https://')):
                                    video_url = output_item
                            elif isinstance(output, dict) and "url" in output:
                                video_url = output["url"]
                            elif isinstance(output, str) and output.startswith(('http://', 'https://')):
                                video_url = output

                        if video_url:
                            logger.info(f"RunwayML Act Two generation succeeded: {video_url}")
                            video_artifact = VideoUrlArtifact(url=video_url, name="runwayml_character_video")
                            self.publish_update_to_parameter("video_output", video_artifact)
                            self.publish_update_to_parameter("seed", actual_seed)
                            return video_artifact
                        else:
                            logger.error(f"RunwayML Act Two task SUCCEEDED but no output URL found. Output structure: {output}")
                            err_msg = "RunwayML Act Two task SUCCEEDED but no output URL found."
                            self.publish_update_to_parameter("video_output", ErrorArtifact(err_msg))
                            self.publish_update_to_parameter("seed", actual_seed)
                            return ErrorArtifact(err_msg)
                    
                    elif status == 'FAILED':
                        error_msg = f"RunwayML Act Two generation failed (Task ID: {task_id})."
                        error_detail = task_status.get("error")
                        if error_detail:
                            error_msg += f" Reason: {error_detail}"
                        logger.error(error_msg)
                        self.publish_update_to_parameter("video_output", ErrorArtifact(error_msg))
                        self.publish_update_to_parameter("seed", actual_seed)
                        return ErrorArtifact(error_msg)

                timeout_msg = f"RunwayML Act Two task (ID: {task_id}) timed out after {max_retries * retry_delay} seconds."
                logger.error(timeout_msg)
                self.publish_update_to_parameter("video_output", ErrorArtifact(timeout_msg))
                self.publish_update_to_parameter("seed", actual_seed)
                return ErrorArtifact(timeout_msg)

            except Exception as e:
                error_message = f"RunwayML Act Two unexpected error: {type(e).__name__} - {e}"
                
                # Handle specific API errors
                if hasattr(e, 'status_code') and e.status_code == 413:
                    error_message = "Media too large! RunwayML has a 5MB limit for data URIs. Try using smaller files or HTTPS URLs instead of local files."
                elif hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 413:
                    error_message = "Media too large! RunwayML has a 5MB limit for data URIs. Try using smaller files or HTTPS URLs instead of local files."
                elif "413" in str(e) or "Request Entity Too Large" in str(e):
                    error_message = "Media too large! RunwayML has a 5MB limit for data URIs. Try using smaller files or HTTPS URLs instead of local files."
                elif hasattr(e, 'status') and hasattr(e, 'reason') and hasattr(e, 'body'):
                    error_message = f"RunwayML API Error: Status {getattr(e, 'status', 'N/A')} - Reason: {getattr(e, 'reason', 'N/A')} - Body: {getattr(e, 'body', 'N/A')}"
                
                logger.exception(error_message)
                self.publish_update_to_parameter("video_output", ErrorArtifact(error_message))
                self.publish_update_to_parameter("seed", actual_seed if 'actual_seed' in locals() else RunwayML_ActTwo._last_used_seed)
                return ErrorArtifact(error_message)

        yield generate_character_performance_async