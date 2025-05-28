import time
import base64
import runwayml
from urllib.parse import urlparse


from griptape.artifacts import TextArtifact, UrlArtifact, ImageArtifact, ImageUrlArtifact, ErrorArtifact
from griptape_nodes.traits.options import Options

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterGroup
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.retained_mode.griptape_nodes import logger

SERVICE = "RunwayML"
API_KEY_ENV_VAR = "RUNWAYML_API_SECRET"
DEFAULT_MODEL = "gen4_turbo"

# Allowed ratio values from RunwayML API error message
# 'ratio' must be one of: 1280:720, 720:1280, 1104:832, 832:1104, 960:960, 1584:672
# For the workaround, we'll use these directly as choices.
RUNWAY_API_RATIO_VALUES = [
    "1280:720", # Approx 16:9 HD
    "720:1280", # Approx 9:16 Portrait HD
    "960:960",  # 1:1 Square
    "1104:832", # Approx 4:3 Traditional
    "832:1104", # Approx 3:4 Portrait Traditional
    "1584:672", # Approx 2.35:1 Widescreen
]
DEFAULT_API_RATIO = "1280:720" # Default to a common valid API string


class VideoUrlArtifact(UrlArtifact):
    """
    Artifact that contains a URL to a video.
    """

    def __init__(self, url: str, name: str | None = None):
        super().__init__(value=url, name=name or self.__class__.__name__)


class RunwayML_ImageToVideo(ControlNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "AI/RunwayML"
        self.description = "Generates a video from an image and prompt using RunwayML."
        self.metadata["author"] = "Griptape"
        self.metadata["dependencies"] = {"pip_dependencies": ["runwayml", "requests"]}


        # Inputs Group
        with ParameterGroup(name="Inputs") as inputs_group:
            Parameter(
                name="image",
                input_types=["ImageArtifact", "ImageUrlArtifact", "str"],
                type="ImageArtifact", 
                tooltip="Input image (required). Accepts ImageArtifact, ImageUrlArtifact, a public URL string, or a base64 data URI string.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            Parameter(
                name="prompt",
                input_types=["str", "TextArtifact"],
                output_type="str",
                type="str",
                default_value="",
                tooltip="Text prompt describing the desired video content.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"multiline": True, "placeholder_text": "e.g., a cinematic shot of a car driving down a road"},
            )
        self.add_node_element(inputs_group)

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
                traits={Options(choices=["gen4_turbo", "gen_3_alpha", "gen_2", "gen_1"])} # Add more as they become available
            )
            Parameter(
                name="ratio",
                input_types=["str"],
                output_type="str",
                type="str",
                default_value=DEFAULT_API_RATIO, # Use the new default
                tooltip="Aspect ratio for the output video (e.g., 1280:720). Must be one of the specific values supported by RunwayML API.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=RUNWAY_API_RATIO_VALUES)} # choices is now list[str]
            )
            Parameter(
                name="seed",
                input_types=["int"],
                output_type="int",
                type="int",
                default_value=0, # API default is random if not specified, 0 is a common way to say "random" or let API decide
                tooltip="Seed for generation. 0 for random.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            Parameter(
                name="motion_score",
                input_types=["int"],
                output_type="int",
                type="int",
                default_value=10, # Example value, API docs might specify range/default
                tooltip="Controls the amount of motion. (Range and effect may vary by model)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"min":0, "max": 20} # Placeholder range
            )
            Parameter(
                name="upscale",
                input_types=["bool"],
                output_type="bool",
                type="bool",
                default_value=False,
                tooltip="Whether to upscale the generated video.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )

        self.add_node_element(gen_settings_group)

        # Output Parameter
        self.add_parameter(
            Parameter(
                name="video_output",
                output_type="VideoUrlArtifact", # Custom artifact for clarity
                type="VideoUrlArtifact",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Output URL of the generated video.",
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

    def _get_image_data_uri(self, param_name: str) -> str | None:
        image_input = self.get_parameter_value(param_name)
        
        if not image_input:
            return None

        if isinstance(image_input, ImageArtifact): # Already Base64
            media_type = image_input.media_type or "image/png" 
            if not image_input.base64.startswith(f"data:{media_type};base64,"):
                 return f"data:{media_type};base64,{image_input.base64}"
            return image_input.base64
        elif isinstance(image_input, ImageUrlArtifact):
            url_value = image_input.value
            if url_value.startswith("data:image"):
                return url_value # Already a data URI
            
            parsed_url = urlparse(url_value)
            if parsed_url.scheme == "http" and (parsed_url.hostname == "localhost" or parsed_url.hostname == "127.0.0.1"):
                logger.info(f"RunwayML I2V: Converting local HTTP URL to base64 data URI: {url_value}")
                try:
                    import requests # Local import for this specific conversion path
                    response = requests.get(url_value, timeout=10)
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "image/png")
                    base64_data = base64.b64encode(response.content).decode("utf-8")
                    return f"data:{content_type};base64,{base64_data}"
                except Exception as e:
                    logger.error(f"RunwayML I2V: Failed to convert local URL {url_value} to base64: {e}")
                    return None # Or raise an error, depending on desired handling
            elif parsed_url.scheme == "https":
                logger.info(f"RunwayML I2V: Using public HTTPS URL for image: {url_value}")
                return url_value
            else:
                logger.warning(f"RunwayML I2V: ImageUrlArtifact with non-HTTPS/non-local-HTTP URL provided: {url_value}. Attempting to send as is.")
                return url_value # Send as is, API will likely reject if not data URI or HTTPS

        elif isinstance(image_input, str):
            if image_input.strip().startswith("data:image"):
                return image_input.strip() # Already a data URI
            
            parsed_url = urlparse(image_input.strip())
            if parsed_url.scheme == "http" and (parsed_url.hostname == "localhost" or parsed_url.hostname == "127.0.0.1"):
                logger.info(f"RunwayML I2V: Converting local HTTP URL string to base64 data URI: {image_input.strip()}")
                try:
                    import requests
                    response = requests.get(image_input.strip(), timeout=10)
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "image/png")
                    base64_data = base64.b64encode(response.content).decode("utf-8")
                    return f"data:{content_type};base64,{base64_data}"
                except Exception as e:
                    logger.error(f"RunwayML I2V: Failed to convert local URL string {image_input.strip()} to base64: {e}")
                    return None
            elif parsed_url.scheme == "https":
                logger.info(f"RunwayML I2V: Using public HTTPS URL string for image: {image_input.strip()}")
                return image_input.strip()
            else:
                logger.warning(f"RunwayML I2V: String input is not a data URI, HTTPS URL, or local HTTP URL: {image_input.strip()}. Attempting to send as is.")
                return image_input.strip() # Send as is, API will likely reject

        elif isinstance(image_input, dict): 
            logger.info(f"RunwayML I2V: received dict for {param_name}: {image_input}")
            input_type = image_input.get("type")
            url_from_dict = image_input.get("value")
            base64_from_dict = image_input.get("base64")
            media_type_from_dict = image_input.get("media_type", "image/png")


            if input_type == "ImageUrlArtifact" and url_from_dict:
                if str(url_from_dict).startswith("data:image"):
                    return str(url_from_dict)
                return str(url_from_dict) # Public URL
            elif input_type == "ImageArtifact" and base64_from_dict:
                if not str(base64_from_dict).startswith(f"data:{media_type_from_dict};base64,"):
                    return f"data:{media_type_from_dict};base64,{base64_from_dict}"
                return str(base64_from_dict)
            
            logger.warning(f"RunwayML I2V: received unhandled dict structure for {param_name}: {image_input}")
            return None
            
        logger.warning(f"RunwayML I2V: Unhandled image input type for {param_name}: {type(image_input)}")
        return None

    def validate_node(self) -> list[Exception] | None:
        errors = []
        api_key = self.get_config_value(service=SERVICE, value=API_KEY_ENV_VAR)

        if not api_key:
            errors.append(ValueError(f"RunwayML API key not found. Set {API_KEY_ENV_VAR} in environment variables or Griptape Cloud."))

        image_data = self._get_image_data_uri("image")
        if not image_data:
            errors.append(ValueError("Image input ('image') is required and must be a valid ImageArtifact, ImageUrlArtifact, public URL, or base64 data URI."))
        
        prompt_val = self.get_parameter_value("prompt")
        if not prompt_val or not str(prompt_val).strip():
             errors.append(ValueError("Text prompt ('prompt') cannot be empty."))

        return errors if errors else None

    def process(self) -> AsyncResult:
        validation_errors = self.validate_node()
        if validation_errors:
            error_message = "; ".join(str(e) for e in validation_errors)
            logger.error(f"RunwayML I2V validation failed: {error_message}")
            # Publish error artifact if possible (though process won't yield it directly here)
            self.publish_update_to_parameter("video_output", ErrorArtifact(error_message))
            raise ValueError(f"Validation failed: {error_message}")
        
        # Get parameter values OUTSIDE the async function to ensure idempotency
        # Create fresh copies to avoid any state modification issues
        prompt_text = str(self.get_parameter_value("prompt") or "").strip()
        model_name = str(self.get_parameter_value("model") or DEFAULT_MODEL)
        
        ratio_input = self.get_parameter_value("ratio")
        if isinstance(ratio_input, tuple) and len(ratio_input) == 2:
            ratio_val = str(ratio_input[1])
        else:
            ratio_val = str(ratio_input or DEFAULT_API_RATIO)
        
        seed_val = self.get_parameter_value("seed") or 0
        motion_score_val = self.get_parameter_value("motion_score") or 10
        upscale_val = bool(self.get_parameter_value("upscale") or False)
        
        # Get image data outside async function
        image_data_uri = self._get_image_data_uri("image")
        if not image_data_uri:
            error_msg = "Failed to process image input."
            self.publish_update_to_parameter("video_output", ErrorArtifact(error_msg))
            raise ValueError(error_msg)
            
        def generate_video_async() -> VideoUrlArtifact | ErrorArtifact:
            try:
                # API key is automatically picked up by SDK from RUNWAYML_API_SECRET env var
                # but ensure it's set for explicit client creation if needed.
                # For now, rely on SDK's default behavior.
                client = runwayml.RunwayML()

                # Use the pre-fetched values to ensure idempotency
                task_payload = {
                    "model": model_name,
                    "prompt_image": image_data_uri, 
                    "prompt_text": prompt_text,   
                    "ratio": ratio_val, 
                }

                # Add optional parameters if they have non-default values
                if seed_val and seed_val != 0:
                    task_payload["seed"] = seed_val
                
                if motion_score_val != 10:  # Only add if different from default
                    task_payload["motion_score"] = motion_score_val
                
                if upscale_val:
                    task_payload["upscale"] = upscale_val

                logger.info(f"RunwayML I2V: Creating task with payload keys: {list(task_payload.keys())}")
                logger.info(f"RunwayML I2V: Prompt text: '{prompt_text}'")
                
                # Create a new image-to-video task
                image_to_video_task = client.image_to_video.create(**task_payload) # type: ignore[arg-type]
                task_id = image_to_video_task.id
                self.publish_update_to_parameter("task_id_output", task_id)
                logger.info(f"RunwayML I2V: Task created with ID: {task_id}")

                # Poll the task until it's complete
                max_retries = 120  # 120 retries * 10 seconds = 20 minutes timeout
                retry_delay = 10  # seconds (as per RunwayML docs example)

                for attempt in range(max_retries):
                    time.sleep(retry_delay)
                    task_status = client.tasks.retrieve(task_id)
                    status = task_status.status
                    
                    logger.info(f"RunwayML I2V generation status (Task ID: {task_id}): {status} (Attempt {attempt + 1}/{max_retries})")

                    if status == 'SUCCEEDED':
                        video_url = None
                        if task_status.output:
                            if isinstance(task_status.output, list) and len(task_status.output) > 0:
                                output_item = task_status.output[0]
                                if hasattr(output_item, 'url') and isinstance(getattr(output_item, 'url', None), str):
                                    video_url = getattr(output_item, 'url')
                                elif isinstance(output_item, str) and output_item.startswith(('http://', 'https://')):
                                    video_url = output_item # output_item is the URL string itself
                            elif hasattr(task_status.output, 'url') and isinstance(getattr(task_status.output, 'url', None), str):
                                video_url = getattr(task_status.output, 'url') # output is an object with a url attribute

                        if video_url:
                            logger.info(f"RunwayML I2V generation succeeded: {video_url}")
                            video_artifact = VideoUrlArtifact(url=video_url, name="runwayml_video")
                            self.publish_update_to_parameter("video_output", video_artifact)
                            return video_artifact
                        else:
                            logger.error(f"RunwayML I2V task SUCCEEDED but no output URL found. Output structure: {task_status.output}")
                            err_msg = "RunwayML I2V task SUCCEEDED but no output URL found."
                            self.publish_update_to_parameter("video_output", ErrorArtifact(err_msg))
                            return ErrorArtifact(err_msg)
                    
                    elif status == 'FAILED':
                        error_msg = f"RunwayML I2V generation failed (Task ID: {task_id})."
                        if task_status.error:
                            error_msg += f" Reason: {task_status.error}"
                        logger.error(error_msg)
                        self.publish_update_to_parameter("video_output", ErrorArtifact(error_msg))
                        return ErrorArtifact(error_msg)
                    
                    # Other statuses like 'PENDING', 'RUNNING' mean continue polling

                timeout_msg = f"RunwayML I2V task (ID: {task_id}) timed out after {max_retries * retry_delay} seconds."
                logger.error(timeout_msg)
                self.publish_update_to_parameter("video_output", ErrorArtifact(timeout_msg))
                return ErrorArtifact(timeout_msg)

            except Exception as e:
                error_message = f"RunwayML I2V unexpected error: {type(e).__name__} - {e}"
                # Attempt to get more details if it's a RunwayML-like error based on common patterns
                if hasattr(e, 'status') and hasattr(e, 'reason') and hasattr(e, 'body'):
                    error_message = f"RunwayML API Error: Status {getattr(e, 'status', 'N/A')} - Reason: {getattr(e, 'reason', 'N/A')} - Body: {getattr(e, 'body', 'N/A')}"
                
                logger.exception(error_message) # Log full traceback for unexpected errors
                self.publish_update_to_parameter("video_output", ErrorArtifact(error_message))
                return ErrorArtifact(error_message)

        yield generate_video_async 