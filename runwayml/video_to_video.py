import time
import base64
import json
import os
import subprocess
import tempfile
import requests
from urllib.parse import urlparse
from typing import Any

from griptape.artifacts import TextArtifact, UrlArtifact, ImageUrlArtifact, ErrorArtifact
from griptape_nodes.traits.options import Options

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterGroup
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.retained_mode.griptape_nodes import logger, GriptapeNodes

SERVICE = "RunwayML"
API_KEY_ENV_VAR = "RUNWAYML_API_SECRET"
DEFAULT_MODEL = "gen4_aleph"

# Allowed ratio values from RunwayML API docs
RATIOS = [
    "1280:720", "720:1280", "1104:832", "832:1104", "960:960", "1584:672"
]
DEFAULT_ASPECT_RATIO = "1280:720"


class VideoUrlArtifact(ImageUrlArtifact):
    """
    Artifact that contains a URL to a video.
    """

    def __init__(self, url: str, name: str | None = None):
        super().__init__(value=url, name=name or self.__class__.__name__)


class RunwayML_VideoToVideo(ControlNode):
    # Class variable to track last used seed across instances (ComfyUI-style)
    _last_used_seed: int = 12345
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "AI/RunwayML"
        self.description = "Generates a video from an input video and prompt using RunwayML."
        self.metadata["author"] = "Griptape"
        self.metadata["dependencies"] = {"pip_dependencies": ["runwayml", "requests"]}
        self.metadata["description"] = "We strongly recommend that you install ffmpeg in order to enable video transcoding features in this node."

        # Individual parameters
        # Video Parameter
        self.add_parameter(
            Parameter(
                name="video",
                input_types=["VideoArtifact", "VideoUrlArtifact"],
                type="VideoArtifact", 
                tooltip="Input video (required). Accepts VideoUrlArtifact.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
            )
        )
        # Prompt Parameter
        self.add_parameter(
            Parameter(
                name="prompt",
                input_types=["str", "TextArtifact"],
                output_type="str",
                type="str",
                default_value="",
                tooltip="Text prompt describing the desired video content.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY, ParameterMode.OUTPUT},
                ui_options={"multiline": True, "placeholder_text": "e.g., a cinematic shot of a car driving down a road"},
            )
        )
        # Reference Image Parameter
        self.add_parameter(
            Parameter(
                name="reference_image",
                input_types=["ImageArtifact", "ImageUrlArtifact", "str"],
                type="ImageArtifact", 
                tooltip="Optional reference image for the generation.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
            )
        )

        # Settings Group
        with ParameterGroup(name="Generation Settings") as settings_group:
            Parameter(
                name="model",
                input_types=["str"],
                output_type="str",
                type="str",
                default_value=DEFAULT_MODEL,
                tooltip="RunwayML model to use for generation.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["gen4_aleph"])}
            )
        
            Parameter(
                name="ratio",
                input_types=["str"],
                output_type="str",
                type="str",
                default_value=DEFAULT_ASPECT_RATIO,
                tooltip="Aspect ratio for the output video.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=RATIOS)}
            )

            Parameter(
                name="seed",
                input_types=["int"],
                output_type="int",
                type="int",
                default_value=RunwayML_VideoToVideo._last_used_seed,
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
                name="public_figure_threshold",
                input_types=["str"],
                output_type="str",
                type="str",
                default_value="auto",
                tooltip="Public figure threshold for content moderation.",
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


    def _transcode_video_file(self, video_file_path: str) -> str | None:
        """
        Transcodes a video file to a format compatible with RunwayML.
        Returns the path to the transcoded file or None if transcoding failed.
        """
        try:
            logger.info("RunwayML V2V: Attempting to transcode video to ensure compatibility...")
            transcoded_file = video_file_path + ".transcoded.mp4"
            
            # Use ffmpeg to transcode to a known good format
            cmd = ["ffmpeg", "-i", video_file_path, "-c:v", "libx264", "-preset", "fast", 
                   "-pix_fmt", "yuv420p", "-c:a", "aac", "-strict", "experimental", transcoded_file]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if os.path.exists(transcoded_file) and os.path.getsize(transcoded_file) > 0:
                logger.info(f"RunwayML V2V: Successfully transcoded video to H.264: {transcoded_file}")
                return transcoded_file
            else:
                logger.warning(f"RunwayML V2V: Failed to transcode video. Error: {result.stderr}")
                return None
        except (FileNotFoundError, subprocess.SubprocessError) as e:
            logger.info(f"RunwayML V2V: ffmpeg not available, skipping video transcoding: {str(e)}")
            return None
        except Exception as e:
            logger.warning(f"RunwayML V2V: Error during video transcoding: {str(e)}")
            return None
    
    def _get_video_data_uri(self, param_name: str) -> str | None:
        """
        Gets a video URL or converts to data URI if needed.
        If ffmpeg is available, will transcode video to ensure compatibility.
        """
        video_input = self.get_parameter_value(param_name)
        
        if not video_input:
            return None

        # Handle URL artifacts
        if isinstance(video_input, (VideoUrlArtifact, ImageUrlArtifact, UrlArtifact)):
            url_value = video_input.value
            if url_value.startswith("data:video"):
                return url_value
            
            parsed_url = urlparse(url_value)
            if parsed_url.scheme == "http" and (parsed_url.hostname == "localhost" or parsed_url.hostname == "127.0.0.1"):
                logger.info(f"RunwayML V2V: Converting local HTTP URL to base64 data URI: {url_value}")
                try:
                    response = requests.get(url_value, timeout=30)  # Longer timeout for video
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "video/mp4")
                    
                    # Write to temporary file for potential transcoding
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                        temp_file.write(response.content)
                        temp_file_path = temp_file.name
                        
                    # Try to transcode the video if ffmpeg is available
                    transcoded_path = self._transcode_video_file(temp_file_path)
                    
                    # Use transcoded file if available, otherwise use original
                    file_to_encode = transcoded_path if transcoded_path else temp_file_path
                    
                    # Convert to base64
                    with open(file_to_encode, "rb") as f:
                        file_content = f.read()
                        base64_data = base64.b64encode(file_content).decode("utf-8")
                    
                    # Clean up
                    try:
                        os.unlink(temp_file_path)
                        if transcoded_path:
                            os.unlink(transcoded_path)
                    except:
                        pass
                    
                    return f"data:{content_type};base64,{base64_data}"
                except Exception as e:
                    logger.error(f"RunwayML V2V: Failed to convert local URL {url_value} to base64: {e}")
                    return None
            elif parsed_url.scheme == "https":
                logger.info(f"RunwayML V2V: Using public HTTPS URL for video: {url_value}")
                return url_value
            else:
                logger.warning(f"RunwayML V2V: URL artifact with non-HTTPS/non-local-HTTP URL provided: {url_value}. Attempting to send as is.")
                return url_value

        # Handle string input (URL or data URI)
        elif isinstance(video_input, str):
            if video_input.strip().startswith("data:video"):
                return video_input.strip()
            
            parsed_url = urlparse(video_input.strip())
            if parsed_url.scheme == "http" and (parsed_url.hostname == "localhost" or parsed_url.hostname == "127.0.0.1"):
                logger.info(f"RunwayML V2V: Converting local HTTP URL string to base64 data URI: {video_input.strip()}")
                try:
                    response = requests.get(video_input.strip(), timeout=30)
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "video/mp4")
                    
                    # Write to temporary file for potential transcoding
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                        temp_file.write(response.content)
                        temp_file_path = temp_file.name
                        
                    # Try to transcode the video if ffmpeg is available
                    transcoded_path = self._transcode_video_file(temp_file_path)
                    
                    # Use transcoded file if available, otherwise use original
                    file_to_encode = transcoded_path if transcoded_path else temp_file_path
                    
                    # Convert to base64
                    with open(file_to_encode, "rb") as f:
                        file_content = f.read()
                        base64_data = base64.b64encode(file_content).decode("utf-8")
                    
                    # Clean up
                    try:
                        os.unlink(temp_file_path)
                        if transcoded_path:
                            os.unlink(transcoded_path)
                    except:
                        pass
                    
                    return f"data:{content_type};base64,{base64_data}"
                except Exception as e:
                    logger.error(f"RunwayML V2V: Failed to convert local URL string {video_input.strip()} to base64: {e}")
                    return None
            elif parsed_url.scheme == "https":
                logger.info(f"RunwayML V2V: Using public HTTPS URL string for video: {video_input.strip()}")
                return video_input.strip()
            else:
                logger.warning(f"RunwayML V2V: String input is not a data URI, HTTPS URL, or local HTTP URL: {video_input.strip()}. Attempting to send as is.")
                return video_input.strip()

        # Handle dictionary input
        elif isinstance(video_input, dict): 
            logger.info(f"RunwayML V2V: received dict for {param_name}: {video_input}")
            input_type = video_input.get("type")
            url_from_dict = video_input.get("value")
            base64_from_dict = video_input.get("base64")
            media_type_from_dict = video_input.get("media_type", "video/mp4")

            if input_type in ["VideoUrlArtifact", "ImageUrlArtifact", "UrlArtifact"] and url_from_dict:
                if str(url_from_dict).startswith("data:video"):
                    return str(url_from_dict)
                return str(url_from_dict)
            elif base64_from_dict:
                if not str(base64_from_dict).startswith(f"data:{media_type_from_dict};base64,"):
                    return f"data:{media_type_from_dict};base64,{base64_from_dict}"
                return str(base64_from_dict)
            
            logger.warning(f"RunwayML V2V: received unhandled dict structure for {param_name}: {video_input}")
            return None
            
        logger.warning(f"RunwayML V2V: Unhandled video input type for {param_name}: {type(video_input)}")
        return None

    def _get_image_data_uri(self, param_name: str) -> str | None:
        """
        Gets an image URL or converts to data URI if needed.
        Validates that the image format is supported (JPEG, PNG, WebP).
        Raises ValueError for unsupported formats.
        """
        # Supported image formats for RunwayML API
        SUPPORTED_FORMATS = {
            "image/jpeg", "image/jpg", "image/png", "image/webp"
        }
        
        image_input = self.get_parameter_value(param_name)
        
        if not image_input:
            return None

        # Handle ImageArtifact
        if hasattr(image_input, 'base64'):
            # Convert format to media type
            format_to_media_type = {
                "JPEG": "image/jpeg",
                "PNG": "image/png",
                "WEBP": "image/webp",
                "GIF": "image/gif"  # Not supported, will be caught below
            }
            image_format = getattr(image_input, 'format', '')
            media_type = format_to_media_type.get(image_format, "image/png")
            
            # Check if format is supported
            if media_type not in SUPPORTED_FORMATS:
                error_msg = f"Unsupported reference image format: {image_format}. Supported formats are JPEG, PNG, and WebP."
                logger.error(f"RunwayML V2V: {error_msg}")
                raise ValueError(error_msg)
                
            if not image_input.base64.startswith(f"data:{media_type};base64,"):
                return f"data:{media_type};base64,{image_input.base64}"
            return image_input.base64
            
        # Handle URL artifacts
        elif hasattr(image_input, 'value'):
            url_value = image_input.value
            if url_value.startswith("data:image"):
                # Check data URI format
                content_type = url_value.split(':')[1].split(';')[0]
                if content_type not in SUPPORTED_FORMATS:
                    error_msg = f"Unsupported reference image format: {content_type}. Supported formats are image/jpeg, image/png, and image/webp."
                    logger.error(f"RunwayML V2V: {error_msg}")
                    raise ValueError(error_msg)
                return url_value
            
            parsed_url = urlparse(url_value)
            if parsed_url.scheme in ["http", "https"]:
                logger.info(f"RunwayML V2V: Converting URL to base64 data URI: {url_value}")
                try:
                    response = requests.get(url_value, timeout=30)
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "image/png")
                    
                    # Check if content type is supported
                    if content_type not in SUPPORTED_FORMATS:
                        error_msg = f"Unsupported reference image format: {content_type}. Supported formats are image/jpeg, image/png, and image/webp."
                        logger.error(f"RunwayML V2V: {error_msg}")
                        raise ValueError(error_msg)
                    
                    base64_data = base64.b64encode(response.content).decode("utf-8")
                    return f"data:{content_type};base64,{base64_data}"
                except ValueError as ve:
                    # Re-raise validation errors
                    raise ve
                except Exception as e:
                    logger.error(f"RunwayML V2V: Failed to convert URL {url_value} to base64: {e}")
                    return None
            else:
                logger.warning(f"RunwayML V2V: URL artifact with non-HTTP/HTTPS URL provided: {url_value}. Cannot process.")
                return None
                
        # Handle string input (URL or data URI)
        elif isinstance(image_input, str):
            if image_input.strip().startswith("data:image"):
                # Check data URI format
                content_type = image_input.strip().split(':')[1].split(';')[0]
                if content_type not in SUPPORTED_FORMATS:
                    error_msg = f"Unsupported reference image format: {content_type}. Supported formats are image/jpeg, image/png, and image/webp."
                    logger.error(f"RunwayML V2V: {error_msg}")
                    raise ValueError(error_msg)
                return image_input.strip()
            
            parsed_url = urlparse(image_input.strip())
            if parsed_url.scheme in ["http", "https"]:
                logger.info(f"RunwayML V2V: Converting URL string to base64 data URI: {image_input.strip()}")
                try:
                    response = requests.get(image_input.strip(), timeout=30)
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "image/png")
                    
                    # Check if content type is supported
                    if content_type not in SUPPORTED_FORMATS:
                        error_msg = f"Unsupported reference image format: {content_type}. Supported formats are image/jpeg, image/png, and image/webp."
                        logger.error(f"RunwayML V2V: {error_msg}")
                        raise ValueError(error_msg)
                        
                    base64_data = base64.b64encode(response.content).decode("utf-8")
                    return f"data:{content_type};base64,{base64_data}"
                except ValueError as ve:
                    # Re-raise validation errors
                    raise ve
                except Exception as e:
                    logger.error(f"RunwayML V2V: Failed to convert URL string {image_input.strip()} to base64: {e}")
                    return None
            else:
                logger.warning(f"RunwayML V2V: String input for {param_name} is not a data URI or valid URL: {image_input.strip()}. Cannot process.")
                return None
                
        # Handle dictionary input
        elif isinstance(image_input, dict):
            logger.info(f"RunwayML V2V: received dict for {param_name}: {image_input}")
            input_type = image_input.get("type")
            url_from_dict = image_input.get("value")
            base64_from_dict = image_input.get("base64")
            media_type_from_dict = image_input.get("media_type", "image/png")
            
            # Check if media type is supported
            if media_type_from_dict not in SUPPORTED_FORMATS:
                error_msg = f"Unsupported reference image format: {media_type_from_dict}. Supported formats are image/jpeg, image/png, and image/webp."
                logger.error(f"RunwayML V2V: {error_msg}")
                raise ValueError(error_msg)
            
            if (input_type in ["ImageUrlArtifact"]) and url_from_dict:
                if str(url_from_dict).startswith("data:image"):
                    # Check data URI format for URL
                    data_uri = str(url_from_dict)
                    content_type = data_uri.split(':')[1].split(';')[0]
                    if content_type not in SUPPORTED_FORMATS:
                        error_msg = f"Unsupported reference image format: {content_type}. Supported formats are image/jpeg, image/png, and image/webp."
                        logger.error(f"RunwayML V2V: {error_msg}")
                        raise ValueError(error_msg)
                    return data_uri
                
                parsed_url = urlparse(str(url_from_dict))
                if parsed_url.scheme in ["http", "https"]:
                    logger.info(f"RunwayML V2V: Converting dict URL to base64 data URI: {str(url_from_dict)[:50]}...")
                    try:
                        response = requests.get(str(url_from_dict), timeout=30)
                        response.raise_for_status()
                        content_type = response.headers.get("Content-Type", "image/png")
                        
                        # Check if content type is supported
                        if content_type not in SUPPORTED_FORMATS:
                            error_msg = f"Unsupported reference image format: {content_type}. Supported formats are image/jpeg, image/png, and image/webp."
                            logger.error(f"RunwayML V2V: {error_msg}")
                            raise ValueError(error_msg)
                            
                        base64_data = base64.b64encode(response.content).decode("utf-8")
                        return f"data:{content_type};base64,{base64_data}"
                    except ValueError as ve:
                        # Re-raise validation errors
                        raise ve
                    except Exception as e:
                        logger.error(f"RunwayML V2V: Failed to convert dict URL to base64: {e}")
                        return None
                else:
                    logger.warning(f"RunwayML V2V: Dict URL with non-HTTP/HTTPS URL provided: {str(url_from_dict)[:50]}... Cannot process.")
                    return None
            elif input_type == "ImageArtifact" and base64_from_dict:
                # For base64 data, use provided media type (already validated above)
                if not str(base64_from_dict).startswith(f"data:{media_type_from_dict};base64,"):
                    return f"data:{media_type_from_dict};base64,{base64_from_dict}"
                return str(base64_from_dict)
            
            logger.warning(f"RunwayML V2V: received unhandled dict structure for {param_name}: {image_input}")
            return None
            
        logger.warning(f"RunwayML V2V: Unhandled input type for {param_name}: {type(image_input)}")
        return None

    def validate_node(self) -> list[Exception] | None:
        errors = []
        api_key = GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR)

        if not api_key:
            errors.append(ValueError(f"RunwayML API key not found. Set {API_KEY_ENV_VAR} in environment variables or Griptape Cloud."))

        # Check required video input
        video_data = self._get_video_data_uri("video")
        if not video_data:
            errors.append(ValueError("Video input ('video') is required and must be a valid URL or data URI."))
        
        # Check prompt
        prompt_val = self.get_parameter_value("prompt")
        if not prompt_val or not str(prompt_val).strip():
            errors.append(ValueError("Text prompt ('prompt') cannot be empty."))
        
        # Check model
        model_val = self.get_parameter_value("model")
        if model_val != "gen4_aleph":
            errors.append(ValueError("Only 'gen4_aleph' model is supported for video-to-video."))

        return errors if errors else None

    def process(self) -> AsyncResult:
        validation_errors = self.validate_node()
        if validation_errors:
            error_message = "; ".join(str(e) for e in validation_errors)
            logger.error(f"RunwayML V2V validation failed: {error_message}")
            self.publish_update_to_parameter("video_output", ErrorArtifact(error_message))
            raise ValueError(f"Validation failed: {error_message}")
        
        # Get parameter values
        prompt_text = str(self.get_parameter_value("prompt") or "").strip()
        model_name = str(self.get_parameter_value("model") or DEFAULT_MODEL)
        ratio_val = str(self.get_parameter_value("ratio") or DEFAULT_ASPECT_RATIO)
        public_figure_threshold = str(self.get_parameter_value("public_figure_threshold") or "auto")
        
        # Handle seed control (ComfyUI-style)
        seed_value = int(self.get_parameter_value("seed") or RunwayML_VideoToVideo._last_used_seed)
        seed_control = self.get_parameter_value("seed_control") or "randomize"
        
        if seed_control == "fixed":
            actual_seed = seed_value
        elif seed_control == "increment":
            actual_seed = RunwayML_VideoToVideo._last_used_seed + 1
        elif seed_control == "decrement":
            actual_seed = RunwayML_VideoToVideo._last_used_seed - 1
        elif seed_control == "randomize":
            import random
            actual_seed = random.randint(0, 2**32 - 1)
        else:
            actual_seed = seed_value  # fallback
        
        # Ensure seed is in valid range for API (0 to 4294967295)
        actual_seed = max(0, min(actual_seed, 4294967295))
        
        # Update last used seed for next run
        RunwayML_VideoToVideo._last_used_seed = actual_seed
        
        # Get video data
        video_uri = self._get_video_data_uri("video")
        if not video_uri:
            error_msg = "Failed to process video input."
            self.publish_update_to_parameter("video_output", ErrorArtifact(error_msg))
            raise ValueError(error_msg)
        
        # Get reference image if provided
        reference_image_uri = self._get_image_data_uri("reference_image")

        def _download_and_store_video(video_url: str, task_id: str | None = None) -> VideoUrlArtifact:
            try:
                logger.info(f"RunwayML V2V: Downloading video from {video_url}")
                response = requests.get(video_url, timeout=60)
                response.raise_for_status()

                content_type = response.headers.get("Content-Type", "video/mp4").lower()
                if "quicktime" in content_type or content_type.endswith("/mov"):
                    extension = "mov"
                elif "webm" in content_type:
                    extension = "webm"
                elif "ogg" in content_type:
                    extension = "ogv"
                elif "h264" in content_type or "mp4" in content_type or "mpeg4" in content_type:
                    extension = "mp4"
                else:
                    extension = "mp4"

                if task_id:
                    filename = f"runwayml_video_to_video_{task_id}.{extension}"
                else:
                    filename = f"runwayml_video_to_video_{int(time.time() * 1000)}.{extension}"

                logger.info(f"RunwayML V2V: Saving video bytes to static storage as {filename}...")
                static_url = GriptapeNodes.StaticFilesManager().save_static_file(response.content, filename)
                logger.info(f"RunwayML V2V: ✅ Video saved. URL: {static_url}")
                return VideoUrlArtifact(url=static_url, name="runwayml_video_to_video")
            except Exception as e:
                logger.error(f"RunwayML V2V: Failed to download and store video: {e}")
                return VideoUrlArtifact(url=video_url, name="runwayml_video_to_video")
            
        def generate_video_async() -> VideoUrlArtifact | ErrorArtifact:
            try:
                api_key = GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR)

                # Build the payload according to the API format
                task_payload = {
                    "videoUri": video_uri,
                    "promptText": prompt_text,
                    "model": model_name,
                    "ratio": ratio_val,
                    "contentModeration": {
                        "publicFigureThreshold": public_figure_threshold
                    }
                }
                
                # Add optional seed if non-zero
                if actual_seed != 0:
                    task_payload["seed"] = actual_seed
                
                # Add reference image if provided
                if reference_image_uri:
                    task_payload["references"] = [
                        {
                            "type": "image",
                            "uri": reference_image_uri
                        }
                    ]

                logger.info(f"RunwayML V2V: Creating task with payload keys: {list(task_payload.keys())}")
                logger.info(f"RunwayML V2V: Prompt text: '{prompt_text}'")
                
                # Make direct API call
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-Runway-Version": "2024-11-06"  # Use latest API version
                }
                
                response = requests.post(
                    "https://api.dev.runwayml.com/v1/video_to_video",
                    json=task_payload,
                    headers=headers,
                    timeout=60
                )
                
                if response.status_code != 200:
                    error_body = response.text
                    logger.error(f"RunwayML V2V: API returned {response.status_code}: {error_body}")
                    raise ValueError(f"RunwayML API Error ({response.status_code}): {error_body}")
                
                response_data = response.json()
                task_id = response_data.get("id")
                if not task_id:
                    raise ValueError("No task ID returned from RunwayML API")
                
                self.publish_update_to_parameter("task_id_output", task_id)
                logger.info(f"RunwayML V2V: Task created with ID: {task_id}")

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
                    
                    logger.info(f"RunwayML V2V generation status (Task ID: {task_id}): {status} (Attempt {attempt + 1}/{max_retries})")

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
                            logger.info(f"RunwayML V2V generation succeeded: {video_url}")
                            video_artifact = _download_and_store_video(video_url, task_id)
                            self.publish_update_to_parameter("video_output", video_artifact)
                            self.publish_update_to_parameter("seed", actual_seed)
                            return video_artifact
                        else:
                            logger.error(f"RunwayML V2V task SUCCEEDED but no output URL found. Output structure: {output}")
                            err_msg = "RunwayML V2V task SUCCEEDED but no output URL found."
                            self.publish_update_to_parameter("video_output", ErrorArtifact(err_msg))
                            self.publish_update_to_parameter("seed", actual_seed)
                            return ErrorArtifact(err_msg)
                    
                    elif status == 'FAILED':
                        error_msg = f"RunwayML V2V generation failed (Task ID: {task_id})."
                        error_detail = task_status.get("error")
                        if error_detail:
                            error_msg += f" Reason: {error_detail}"
                        logger.error(error_msg)
                        self.publish_update_to_parameter("video_output", ErrorArtifact(error_msg))
                        self.publish_update_to_parameter("seed", actual_seed)
                        return ErrorArtifact(error_msg)

                timeout_msg = f"RunwayML V2V task (ID: {task_id}) timed out after {max_retries * retry_delay} seconds."
                logger.error(timeout_msg)
                self.publish_update_to_parameter("video_output", ErrorArtifact(timeout_msg))
                self.publish_update_to_parameter("seed", actual_seed)
                return ErrorArtifact(timeout_msg)

            except Exception as e:
                error_message = f"RunwayML V2V unexpected error: {type(e).__name__} - {e}"
                
                # Handle specific API errors
                if hasattr(e, 'status_code') and e.status_code == 413:
                    error_message = "Media too large! RunwayML has a limit for data URIs. Try using smaller files or HTTPS URLs instead of local files."
                elif hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 413:
                    error_message = "Media too large! RunwayML has a limit for data URIs. Try using smaller files or HTTPS URLs instead of local files."
                elif "413" in str(e) or "Request Entity Too Large" in str(e):
                    error_message = "Media too large! RunwayML has a limit for data URIs. Try using smaller files or HTTPS URLs instead of local files."
                
                logger.exception(error_message)
                self.publish_update_to_parameter("video_output", ErrorArtifact(error_message))
                self.publish_update_to_parameter("seed", actual_seed if 'actual_seed' in locals() else RunwayML_VideoToVideo._last_used_seed)
                return ErrorArtifact(error_message)

        yield generate_video_async