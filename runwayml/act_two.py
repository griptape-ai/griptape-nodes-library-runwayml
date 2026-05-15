import time

import requests
from griptape.artifacts import ErrorArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.retained_mode.events.os_events import ExistingFilePolicy
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger
from griptape_nodes.traits.options import Options
from media import prepare_media_data_uri

SERVICE = "RunwayML"
API_KEY_ENV_VAR = "RUNWAYML_API_SECRET"
DEFAULT_MODEL = "act_two"

# Character type options
CHARACTER_TYPES = ["image", "video"]
DEFAULT_CHARACTER_TYPE = "image"

# Aspect ratios supported by Runway API
ASPECT_RATIOS = ["1280:720", "720:1280", "1104:832", "832:1104", "960:960", "1584:672"]
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
        self.metadata["dependencies"] = {"pip_dependencies": ["requests"]}

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
                traits={Options(choices=CHARACTER_TYPES)},
            )
        self.add_node_element(character_type_group)

        # Media Inputs

        self.add_parameter(
            Parameter(
                name="character_video",
                input_types=["VideoUrlArtifact", "VideoArtifact"],
                type="VideoUrlArtifact",
                tooltip="The video to process",
                ui_options={
                    "clickable_file_browser": True,
                    "expander": True,
                    "display_name": "Video or Path to Video",
                },
            )
        )

        self.add_parameter(
            Parameter(
                name="reference_video",
                input_types=["VideoUrlArtifact", "VideoArtifact"],
                type="VideoUrlArtifact",
                tooltip="The video to process",
                ui_options={
                    "clickable_file_browser": True,
                    "expander": True,
                    "display_name": "Video or Path to Video",
                },
            )
        )

        self.add_parameter(
            Parameter(
                name="character_image",
                input_types=["ImageArtifact", "ImageUrlArtifact", "str"],
                type="ImageUrlArtifact",
                tooltip="Input image of the character. Accepts ImageUrlArtifact, a public URL string, or a base64 data URI string.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
                ui_options={"clickable_file_browser": True},
            )
        )

        # Settings Group
        with ParameterGroup(name="Settings") as settings_group:
            Parameter(
                name="body_control",
                input_types=["bool"],
                output_type="bool",
                type="bool",
                default_value=True,
                tooltip="[REQUIRED] Whether to enable body control.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )

            Parameter(
                name="expression_intensity",
                input_types=["int"],
                output_type="int",
                type="int",
                default_value=DEFAULT_EXPRESSION_INTENSITY,
                tooltip="[REQUIRED] Expression intensity (1-5).",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=EXPRESSION_INTENSITY_OPTIONS)},
            )

            Parameter(
                name="ratio",
                input_types=["str"],
                output_type="str",
                type="str",
                default_value=DEFAULT_ASPECT_RATIO,
                tooltip="[REQUIRED] Aspect ratio for the output video.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=ASPECT_RATIOS)},
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
                traits={Options(choices=["fixed", "increment", "decrement", "randomize"])},
            )

            Parameter(
                name="model",
                input_types=["str"],
                output_type="str",
                type="str",
                default_value=DEFAULT_MODEL,
                tooltip="[REQUIRED] RunwayML model to use for generation.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["act_two"])},
            )

            Parameter(
                name="public_figure_threshold",
                input_types=["str"],
                output_type="str",
                type="str",
                default_value="auto",
                tooltip="[OPTIONAL] Public figure threshold for content moderation.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["auto", "low"])},
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
                ui_options={"placeholder_text": "", "is_full_width": False, "pulse_on_run": True},
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
        """Resolve an input parameter to a value the character_performance endpoint accepts.

        Returns an ``https://``, ``runway://``, or ``data:<kind>/...`` URI, or ``None``
        when the parameter is unset or could not be loaded.
        """
        kind = "image" if param_name == "character_image" else "video"
        return prepare_media_data_uri(
            self.get_parameter_value(param_name),
            kind=kind,
            node_name="RunwayML Act Two",
        )

    def validate_node(self) -> list[Exception] | None:
        errors = []
        api_key = GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR)

        if not api_key:
            errors.append(
                ValueError(
                    f"RunwayML API key not found. Set {API_KEY_ENV_VAR} in environment variables or Griptape Cloud."
                )
            )

        # Get character type and validate appropriate input
        character_type = self.get_parameter_value("character_type") or DEFAULT_CHARACTER_TYPE

        if character_type == "image":
            if not self._get_data_uri("character_image"):
                errors.append(ValueError("Character image is required when character type is 'image'."))
        elif character_type == "video":
            if not self._get_data_uri("character_video"):
                errors.append(ValueError("Character video is required when character type is 'video'."))
        else:
            errors.append(ValueError(f"Invalid character type: {character_type}. Must be 'image' or 'video'."))

        # Validate reference video
        if not self._get_data_uri("reference_video"):
            errors.append(ValueError("Reference video ('reference_video') is required."))

        # Validate aspect ratio
        ratio_val = self.get_parameter_value("ratio")
        if not ratio_val or str(ratio_val) not in ASPECT_RATIOS:
            errors.append(
                ValueError(f"Valid aspect ratio is required. Supported values are: {', '.join(ASPECT_RATIOS)}")
            )

        # Validate expression intensity
        expression_intensity = self.get_parameter_value("expression_intensity")
        if expression_intensity is None or expression_intensity not in EXPRESSION_INTENSITY_OPTIONS:
            errors.append(ValueError("Valid expression intensity (1-5) is required."))

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

        def _download_and_store_video(video_url: str, task_id: str | None = None) -> VideoUrlArtifact:
            try:
                logger.info(f"RunwayML Act Two: Downloading video from {video_url}")
                file_content = File(video_url).read()

                content_type = file_content.mime_type.lower() if file_content.mime_type else "video/mp4"
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
                    filename = f"runwayml_act_two_{task_id}.{extension}"
                else:
                    filename = f"runwayml_act_two_{int(time.time() * 1000)}.{extension}"

                logger.info(f"RunwayML Act Two: Saving video bytes to static storage as {filename}...")
                static_url = GriptapeNodes.StaticFilesManager().save_static_file(
                    file_content.content, filename, ExistingFilePolicy.CREATE_NEW
                )
                logger.info(f"RunwayML Act Two: ✅ Video saved. URL: {static_url}")
                return VideoUrlArtifact(url=static_url, name="runwayml_character_video")
            except FileLoadError as e:
                logger.error(f"RunwayML Act Two: Failed to download and store video: {e}")
                return VideoUrlArtifact(url=video_url, name="runwayml_character_video")

        def generate_character_performance_async() -> VideoUrlArtifact | ErrorArtifact:
            try:
                api_key = GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR)

                # Build the payload according to the API format
                task_payload = {
                    "character": {"type": character_type, "uri": character_image_uri or character_video_uri},
                    "reference": {"type": "video", "uri": reference_video_uri},
                    "bodyControl": body_control,
                    "expressionIntensity": expression_intensity,
                    "model": model_val,
                    "ratio": ratio_val,
                    "contentModeration": {"publicFigureThreshold": public_figure_threshold},
                }

                # Add optional seed if non-zero
                if actual_seed != 0:
                    task_payload["seed"] = actual_seed

                logger.info(f"RunwayML Act Two: Creating task with payload keys: {list(task_payload.keys())}")

                # Make direct API call
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-Runway-Version": "2024-11-06",  # Use latest API version
                }

                response = requests.post(
                    "https://api.dev.runwayml.com/v1/character_performance",
                    json=task_payload,
                    headers=headers,
                    timeout=60,
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
                        f"https://api.dev.runwayml.com/v1/tasks/{task_id}", headers=headers, timeout=30
                    )

                    if status_response.status_code != 200:
                        raise ValueError(
                            f"Failed to get task status: {status_response.status_code} - {status_response.text}"
                        )

                    task_status = status_response.json()
                    status = task_status.get("status")

                    logger.info(
                        f"RunwayML Act Two generation status (Task ID: {task_id}): {status} (Attempt {attempt + 1}/{max_retries})"
                    )

                    if status == "SUCCEEDED":
                        video_url = None
                        output = task_status.get("output")

                        # Extract the video URL from the response
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
                            logger.info(f"RunwayML Act Two generation succeeded: {video_url}")
                            video_artifact = _download_and_store_video(video_url, task_id)
                            self.publish_update_to_parameter("video_output", video_artifact)
                            self.publish_update_to_parameter("seed", actual_seed)
                            return video_artifact
                        else:
                            logger.error(
                                f"RunwayML Act Two task SUCCEEDED but no output URL found. Output structure: {output}"
                            )
                            err_msg = "RunwayML Act Two task SUCCEEDED but no output URL found."
                            self.publish_update_to_parameter("video_output", ErrorArtifact(err_msg))
                            self.publish_update_to_parameter("seed", actual_seed)
                            return ErrorArtifact(err_msg)

                    elif status == "FAILED":
                        error_msg = f"RunwayML Act Two generation failed (Task ID: {task_id})."
                        error_detail = task_status.get("error")
                        if error_detail:
                            error_msg += f" Reason: {error_detail}"
                        logger.error(error_msg)
                        self.publish_update_to_parameter("video_output", ErrorArtifact(error_msg))
                        self.publish_update_to_parameter("seed", actual_seed)
                        return ErrorArtifact(error_msg)

                timeout_msg = (
                    f"RunwayML Act Two task (ID: {task_id}) timed out after {max_retries * retry_delay} seconds."
                )
                logger.error(timeout_msg)
                self.publish_update_to_parameter("video_output", ErrorArtifact(timeout_msg))
                self.publish_update_to_parameter("seed", actual_seed)
                return ErrorArtifact(timeout_msg)

            except Exception as e:
                error_message = f"RunwayML Act Two unexpected error: {type(e).__name__} - {e}"

                # Handle specific API errors
                if hasattr(e, "status_code") and e.status_code == 413:
                    error_message = "Media too large! RunwayML has a 5MB limit for data URIs. Try using smaller files or HTTPS URLs instead of local files."
                elif hasattr(e, "response") and hasattr(e.response, "status_code") and e.response.status_code == 413:
                    error_message = "Media too large! RunwayML has a 5MB limit for data URIs. Try using smaller files or HTTPS URLs instead of local files."
                elif "413" in str(e) or "Request Entity Too Large" in str(e):
                    error_message = "Media too large! RunwayML has a 5MB limit for data URIs. Try using smaller files or HTTPS URLs instead of local files."
                elif hasattr(e, "status") and hasattr(e, "reason") and hasattr(e, "body"):
                    error_message = f"RunwayML API Error: Status {getattr(e, 'status', 'N/A')} - Reason: {getattr(e, 'reason', 'N/A')} - Body: {getattr(e, 'body', 'N/A')}"

                logger.exception(error_message)
                self.publish_update_to_parameter("video_output", ErrorArtifact(error_message))
                self.publish_update_to_parameter(
                    "seed", actual_seed if "actual_seed" in locals() else RunwayML_ActTwo._last_used_seed
                )
                return ErrorArtifact(error_message)

        yield generate_character_performance_async
