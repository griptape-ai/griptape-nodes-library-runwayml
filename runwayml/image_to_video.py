import time
import base64
import requests
from urllib.parse import urlparse
from typing import Any

from griptape.artifacts import TextArtifact, ImageUrlArtifact, ErrorArtifact
from griptape_nodes.traits.options import Options

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterGroup
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode, BaseNode
from griptape_nodes.retained_mode.griptape_nodes import logger, GriptapeNodes

SERVICE = "RunwayML"
API_KEY_ENV_VAR = "RUNWAYML_API_SECRET"
DEFAULT_MODEL = "gen4_turbo"

# Allowed ratio values by model from RunwayML API docs
GEN4_TURBO_RATIOS = ["1280:720", "720:1280", "1104:832", "832:1104", "960:960", "1584:672"]
GEN3A_TURBO_RATIOS = ["1280:768", "768:1280"]
GEN4_DEFAULT_ASPECT_RATIO = "1280:720"
GEN3A_DEFAULT_ASPECT_RATIO = "1280:768"


class VideoUrlArtifact(ImageUrlArtifact):
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
        self.metadata["dependencies"] = {"pip_dependencies": ["requests"]}

        # Individual parameters (following Kling pattern)
        self.add_parameter(
            Parameter(
                name="image",
                input_types=["ImageUrlArtifact", "str"],
                type="ImageUrlArtifact",
                tooltip="Input image (required). Accepts ImageUrlArtifact, a public URL string, or a base64 data URI string.",
                allowed_modes={ParameterMode.INPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="prompt",
                input_types=["str", "TextArtifact"],
                output_type="str",
                type="str",
                default_value="",
                tooltip="Text prompt describing the desired video content.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "multiline": True,
                    "placeholder_text": "e.g., a cinematic shot of a car driving down a road",
                },
            )
        )
        self.add_parameter(
            Parameter(
                name="model",
                input_types=["str"],
                output_type="str",
                type="str",
                default_value=DEFAULT_MODEL,
                tooltip="RunwayML model to use for generation.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["gen4_turbo", "gen3a_turbo"])},
            )
        )
        self.add_parameter(
            Parameter(
                name="ratio",
                input_types=["str"],
                output_type="str",
                type="str",
                default_value=GEN4_DEFAULT_ASPECT_RATIO,
                tooltip="Aspect ratio for the output video. Available ratios depend on selected model.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=GEN4_TURBO_RATIOS)},
            )
        )

        self.add_parameter(
            Parameter(
                name="duration",
                input_types=["int"],
                output_type="int",
                type="int",
                default_value=10,
                tooltip="Duration of output video in seconds.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=[5, 10])},
            )
        )
        self.add_parameter(
            Parameter(
                name="seed",
                input_types=["int"],
                output_type="int",
                type="int",
                default_value=0,
                tooltip="Seed for generation. 0 for random.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        # Output Parameters
        self.add_parameter(
            Parameter(
                name="video_output",
                output_type="VideoUrlArtifact",
                type="VideoUrlArtifact",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Output URL of the generated video.",
                ui_options={"placeholder_text": "", "is_full_width": True, "pulse_on_run": True},
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
                ui_options={"placeholder_text": ""},
            )
        )

    def _get_image_data_uri(self, param_name: str) -> str | None:
        image_input = self.get_parameter_value(param_name)

        if not image_input:
            return None

        if isinstance(image_input, ImageUrlArtifact):
            url_value = image_input.value
            if url_value.startswith("data:image"):
                return url_value

            parsed_url = urlparse(url_value)
            if parsed_url.scheme == "http" and (
                parsed_url.hostname == "localhost" or parsed_url.hostname == "127.0.0.1"
            ):
                logger.info(f"RunwayML I2V: Converting local HTTP URL to base64 data URI: {url_value}")
                try:
                    import requests

                    response = requests.get(url_value, timeout=10)
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "image/png")
                    base64_data = base64.b64encode(response.content).decode("utf-8")
                    return f"data:{content_type};base64,{base64_data}"
                except Exception as e:
                    logger.error(f"RunwayML I2V: Failed to convert local URL {url_value} to base64: {e}")
                    return None
            elif parsed_url.scheme == "https":
                logger.info(f"RunwayML I2V: Using public HTTPS URL for image: {url_value}")
                return url_value
            else:
                logger.warning(
                    f"RunwayML I2V: ImageUrlArtifact with non-HTTPS/non-local-HTTP URL provided: {url_value}. Attempting to send as is."
                )
                return url_value

        elif isinstance(image_input, str):
            if image_input.strip().startswith("data:image"):
                return image_input.strip()

            parsed_url = urlparse(image_input.strip())
            if parsed_url.scheme == "http" and (
                parsed_url.hostname == "localhost" or parsed_url.hostname == "127.0.0.1"
            ):
                logger.info(f"RunwayML I2V: Converting local HTTP URL string to base64 data URI: {image_input.strip()}")
                try:
                    import requests

                    response = requests.get(image_input.strip(), timeout=10)
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "image/png")
                    base64_data = base64.b64encode(response.content).decode("utf-8")
                    return f"data:{content_type};base64,{base64_data}"
                except Exception as e:
                    logger.error(
                        f"RunwayML I2V: Failed to convert local URL string {image_input.strip()} to base64: {e}"
                    )
                    return None
            elif parsed_url.scheme == "https":
                logger.info(f"RunwayML I2V: Using public HTTPS URL string for image: {image_input.strip()}")
                return image_input.strip()
            else:
                logger.warning(
                    f"RunwayML I2V: String input is not a data URI, HTTPS URL, or local HTTP URL: {image_input.strip()}. Attempting to send as is."
                )
                return image_input.strip()

        elif isinstance(image_input, dict):
            logger.info(f"RunwayML I2V: received dict for {param_name}: {image_input}")
            input_type = image_input.get("type")
            url_from_dict = image_input.get("value")
            base64_from_dict = image_input.get("base64")
            media_type_from_dict = image_input.get("media_type", "image/png")

            if input_type == "ImageUrlArtifact" and url_from_dict:
                if str(url_from_dict).startswith("data:image"):
                    return str(url_from_dict)
                return str(url_from_dict)
            elif base64_from_dict:
                if not str(base64_from_dict).startswith(f"data:{media_type_from_dict};base64,"):
                    return f"data:{media_type_from_dict};base64,{base64_from_dict}"
                return str(base64_from_dict)

            logger.warning(f"RunwayML I2V: received unhandled dict structure for {param_name}: {image_input}")
            return None

        logger.warning(f"RunwayML I2V: Unhandled image input type for {param_name}: {type(image_input)}")
        return None

    def validate_node(self) -> list[Exception] | None:
        errors = []
        api_key = GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR)

        if not api_key:
            errors.append(
                ValueError(
                    f"RunwayML API key not found. Set {API_KEY_ENV_VAR} in environment variables or Griptape Cloud."
                )
            )

        image_data = self._get_image_data_uri("image")
        if not image_data:
            errors.append(
                ValueError(
                    "Image input ('image') is required and must be a valid ImageUrlArtifact, public URL, or base64 data URI."
                )
            )

        prompt_val = self.get_parameter_value("prompt")
        if not prompt_val or not str(prompt_val).strip():
            errors.append(ValueError("Text prompt ('prompt') cannot be empty."))

        return errors if errors else None

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "model":
            model_name = self.get_parameter_value("model")
            if model_name == "gen4_turbo":
                self._update_option_choices(param="ratio", choices=GEN4_TURBO_RATIOS, default=GEN4_DEFAULT_ASPECT_RATIO)
            elif model_name == "gen3a_turbo":
                self._update_option_choices(
                    param="ratio", choices=GEN3A_TURBO_RATIOS, default=GEN3A_DEFAULT_ASPECT_RATIO
                )

        return super().after_value_set(parameter, value)

    def _download_and_store_video(self, video_url: str, task_id: str | None = None) -> VideoUrlArtifact:
        try:
            logger.info(f"RunwayML I2V: Downloading video from {video_url}")
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
                filename = f"runwayml_image_to_video_{task_id}.{extension}"
            else:
                filename = f"runwayml_image_to_video_{int(time.time() * 1000)}.{extension}"

            logger.info(f"RunwayML I2V: Saving video bytes to static storage as {filename}...")
            static_url = GriptapeNodes.StaticFilesManager().save_static_file(response.content, filename)
            logger.info(f"RunwayML I2V: ✅ Video saved. URL: {static_url}")
            return VideoUrlArtifact(url=static_url, name="runwayml_video")
        except Exception as e:
            logger.error(f"RunwayML I2V: Failed to download and store video: {e}")
            return VideoUrlArtifact(url=video_url, name="runwayml_video")

    def process(self) -> AsyncResult:
        validation_errors = self.validate_node()
        if validation_errors:
            error_message = "; ".join(str(e) for e in validation_errors)
            logger.error(f"RunwayML I2V validation failed: {error_message}")
            self.publish_update_to_parameter("video_output", ErrorArtifact(error_message))
            raise ValueError(f"Validation failed: {error_message}")

        # Get parameter values
        prompt_text = str(self.get_parameter_value("prompt") or "").strip()
        model_name = str(self.get_parameter_value("model") or DEFAULT_MODEL)
        ratio_val = str(self.get_parameter_value("ratio") or GEN4_DEFAULT_ASPECT_RATIO)
        seed_val = self.get_parameter_value("seed") or 0
        duration_val = self.get_parameter_value("duration") or 10

        # Get image data
        image_data_uri = self._get_image_data_uri("image")
        if not image_data_uri:
            error_msg = "Failed to process image input."
            self.publish_update_to_parameter("video_output", ErrorArtifact(error_msg))
            raise ValueError(error_msg)

        def generate_video_async() -> VideoUrlArtifact | ErrorArtifact:
            try:
                api_key = GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR)

                task_payload = {
                    "model": model_name,
                    "promptImage": image_data_uri,
                    "promptText": prompt_text,
                    "ratio": ratio_val,
                    "duration": duration_val,
                }

                # Note: 'position' parameter removed as it's not supported by the RunwayML Python SDK

                # Add optional parameters if they have non-default values
                if seed_val and seed_val != 0:
                    task_payload["seed"] = seed_val

                logger.info(f"RunwayML I2V: Creating task with payload keys: {list(task_payload.keys())}")
                logger.info(f"RunwayML I2V: Prompt text: '{prompt_text}'")

                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-Runway-Version": "2024-11-06",
                }

                # Create a new image-to-video task
                response = requests.post(
                    "https://api.dev.runwayml.com/v1/image_to_video", json=task_payload, headers=headers, timeout=60
                )
                if response.status_code != 200:
                    error_body = response.text
                    logger.error(f"RunwayML I2V: API returned {response.status_code}: {error_body}")
                    raise ValueError(f"RunwayML API Error ({response.status_code}): {error_body}")

                task_response = response.json()
                task_id = task_response.get("id")
                if not task_id:
                    raise ValueError(f"No task ID returned from RunwayML API. Response: {task_response}")
                self.publish_update_to_parameter("task_id_output", task_id)
                logger.info(f"RunwayML I2V: Task created with ID: {task_id}")

                # Poll the task until it's complete
                max_retries = 120
                retry_delay = 10

                for attempt in range(max_retries):
                    time.sleep(retry_delay)
                    status_response = requests.get(
                        f"https://api.dev.runwayml.com/v1/tasks/{task_id}", headers=headers, timeout=30
                    )
                    status_response.raise_for_status()
                    task_status = status_response.json()
                    status = task_status.get("status")

                    logger.info(
                        f"RunwayML I2V generation status (Task ID: {task_id}): {status} (Attempt {attempt + 1}/{max_retries})"
                    )

                    if status == "SUCCEEDED":
                        video_url = None
                        output = task_status.get("output")
                        if output:
                            if isinstance(output, list) and len(output) > 0:
                                output_item = output[0]
                                if isinstance(output_item, dict) and "url" in output_item:
                                    video_url = output_item["url"]
                                elif isinstance(output_item, str) and output_item.startswith(("http://", "https://")):
                                    video_url = output_item
                            elif isinstance(output, dict) and "url" in output:
                                video_url = output["url"]
                            elif isinstance(output, str) and output.startswith(("http://", "https://")):
                                video_url = output

                        if video_url:
                            logger.info(f"RunwayML I2V generation succeeded: {video_url}")
                            video_artifact = self._download_and_store_video(video_url, task_id)
                            self.publish_update_to_parameter("video_output", video_artifact)
                            return video_artifact
                        else:
                            logger.error(
                                f"RunwayML I2V task SUCCEEDED but no output URL found. Output structure: {task_status.output}"
                            )
                            err_msg = "RunwayML I2V task SUCCEEDED but no output URL found."
                            self.publish_update_to_parameter("video_output", ErrorArtifact(err_msg))
                            return ErrorArtifact(err_msg)

                    elif status == "FAILED":
                        error_msg = f"RunwayML I2V generation failed (Task ID: {task_id})."
                        if task_status.error:
                            error_msg += f" Reason: {task_status.error}"
                        logger.error(error_msg)
                        self.publish_update_to_parameter("video_output", ErrorArtifact(error_msg))
                        return ErrorArtifact(error_msg)

                timeout_msg = f"RunwayML I2V task (ID: {task_id}) timed out after {max_retries * retry_delay} seconds."
                logger.error(timeout_msg)
                self.publish_update_to_parameter("video_output", ErrorArtifact(timeout_msg))
                return ErrorArtifact(timeout_msg)

            except Exception as e:
                error_message = f"RunwayML I2V unexpected error: {type(e).__name__} - {e}"
                if hasattr(e, "status") and hasattr(e, "reason") and hasattr(e, "body"):
                    error_message = f"RunwayML API Error: Status {getattr(e, 'status', 'N/A')} - Reason: {getattr(e, 'reason', 'N/A')} - Body: {getattr(e, 'body', 'N/A')}"

                logger.exception(error_message)
                self.publish_update_to_parameter("video_output", ErrorArtifact(error_message))
                return ErrorArtifact(error_message)

        yield generate_video_async
