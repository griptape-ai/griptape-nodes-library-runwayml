import os
import time

import requests
from griptape.artifacts import ErrorArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.retained_mode.events.os_events import ExistingFilePolicy
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger
from griptape_nodes.traits.options import Options


# Reuse the VideoUrlArtifact defined alongside ImageUrlArtifact in existing node
# Define a lightweight VideoUrlArtifact locally to avoid package import issues
class VideoUrlArtifact(ImageUrlArtifact):
    """Artifact that contains a URL to a video."""

    def __init__(self, url: str, name: str | None = None):
        super().__init__(value=url, name=name or self.__class__.__name__)


SERVICE = "RunwayML"
API_KEY_ENV_VAR = "RUNWAYML_API_SECRET"
DEFAULT_MODEL = "upscale_v1"

# Hardcoded polling (matches library pattern)
MAX_RETRIES = 120  # 20 minutes @ 10s
RETRY_DELAY_SECONDS = 10


class RunwayML_VideoUpscale(ControlNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "AI/RunwayML"
        self.description = "Upscales a video by ~4× (capped at 4096px) using RunwayML."
        self.metadata["author"] = "Griptape"
        self.metadata["dependencies"] = {"pip_dependencies": ["requests"]}

        # Input video. Wrapped with PublicArtifactUrlParameter so RunwayML receives a
        # public HTTPS URL it can fetch directly, sidestepping the data-URI body cap.
        self._public_video = PublicArtifactUrlParameter(
            node=self,
            artifact_url_parameter=ParameterVideo(
                name="video",
                tooltip=(
                    "Input video. Allowed content-types: video/mp4, video/webm, video/quicktime, "
                    "video/mov, video/ogg, video/h264. Max 16MB; max duration 40s."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            ),
            disclaimer_message="RunwayML uses this URL to fetch the video for upscaling.",
        )
        self._public_video.add_input_parameters()

        self.add_parameter(
            Parameter(
                name="model",
                input_types=["str"],
                output_type="str",
                type="str",
                default_value=DEFAULT_MODEL,
                tooltip="RunwayML model variant to use for upscaling.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=[DEFAULT_MODEL])},
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
                tooltip="URL of the upscaled video (saved to static files).",
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
                tooltip="The Task ID of the RunwayML job.",
                ui_options={"placeholder_text": ""},
            )
        )

    # --- Helpers ---
    def _get_video_uri(self) -> str | None:
        """Resolve the ``video`` input to a public HTTPS URL via Griptape Cloud upload.

        Public URLs pass through unchanged; macro paths and local files are uploaded.
        Returns ``None`` when the parameter is unset.
        """
        if not self.get_parameter_value("video"):
            return None
        return self._public_video.get_public_url_for_parameter()

    def _download_and_store_video(self, video_url: str, task_id: str | None = None) -> VideoUrlArtifact:
        try:
            logger.info(f"RunwayML VideoUpscale: Downloading video from {video_url}")
            file_content = File(video_url).read()

            content_type = file_content.mime_type.lower() if file_content.mime_type else "video/mp4"
            # Basic mapping for common content-types
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
                filename = f"runwayml_upscaled_video_{task_id}.{extension}"
            else:
                filename = f"runwayml_upscaled_video_{int(time.time() * 1000)}.{extension}"

            static_url = GriptapeNodes.StaticFilesManager().save_static_file(
                file_content.content, filename, ExistingFilePolicy.CREATE_NEW
            )
            return VideoUrlArtifact(url=static_url, name="runwayml_upscaled_video")
        except FileLoadError as e:
            logger.error(f"RunwayML VideoUpscale: Failed to download and store video: {e}")
            # Fallback to original URL if we can't save
            return VideoUrlArtifact(url=video_url, name="runwayml_video")

    def _log_storage_env_hints(self) -> None:
        try:
            sm = GriptapeNodes.StaticFilesManager()
            logger.info("RunwayML VideoUpscale: StaticFilesManager instance: %s", sm.__class__.__name__)
            # Attempt to log likely backend attribute names if present (without secrets)
            backend_attr_names = [
                n for n in dir(sm) if any(k in n.lower() for k in ["backend", "storage", "bucket", "client"])
            ]
            logger.info("RunwayML VideoUpscale: StaticFilesManager attrs (subset): %s", backend_attr_names)

            # Environment hints (keys only + masked preview)
            prefixes = [
                "GT_",
                "GRIPTAPE_",
                "STATIC_",
                "STORAGE_",
                "AWS_",
                "AZURE_",
                "GCP_",
                "GOOGLE_",
                "GCLOUD_",
                "S3_",
                "R2_",
                "DO_SPACES_",
                "SUPABASE_",
                "MINIO_",
                "BUCKET_",
                "BACKBLAZE_",
                "B2_",
                "CLOUDFLARE_",
            ]

            def _mask(val: str) -> str:
                s = str(val)
                if len(s) <= 8:
                    return "***"
                return s[:3] + "***" + s[-2:]

            matched = []
            for k, v in os.environ.items():
                if any(k.startswith(p) for p in prefixes):
                    matched.append((k, _mask(v)))
            if matched:
                logger.info("RunwayML VideoUpscale: Detected env keys: %s", [k for k, _ in matched])
            else:
                logger.info("RunwayML VideoUpscale: No storage-related env keys detected")
        except Exception as e:
            logger.warning(f"RunwayML VideoUpscale: Failed to log storage env hints: {e}")

    # --- Execution ---
    def validate_node(self) -> list[Exception] | None:
        errors: list[Exception] = []
        api_key = GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR)
        if not api_key:
            errors.append(
                ValueError(
                    f"RunwayML API key not found. Set {API_KEY_ENV_VAR} in environment variables or Griptape Cloud."
                )
            )

        if not self.get_parameter_value("video"):
            errors.append(ValueError("Video input ('video') is required."))

        return errors if errors else None

    def process(self) -> AsyncResult:
        validation_errors = self.validate_node()
        if validation_errors:
            error_message = "; ".join(str(e) for e in validation_errors)
            logger.error(f"RunwayML VideoUpscale validation failed: {error_message}")
            self.publish_update_to_parameter("video_output", ErrorArtifact(error_message))
            raise ValueError(f"Validation failed: {error_message}")

        # Log storage/backend hints once per run
        self._log_storage_env_hints()

        model_name = str(self.get_parameter_value("model") or DEFAULT_MODEL)

        def upscale_async() -> VideoUrlArtifact | ErrorArtifact:
            try:
                api_key = GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR)

                # Resolve to a public HTTPS URL (uploads to Griptape Cloud if needed).
                video_uri = self._public_video.get_public_url_for_parameter()

                payload = {"model": model_name, "videoUri": video_uri}
                logger.info(f"RunwayML VideoUpscale: Creating task with payload keys: {list(payload.keys())}")

                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-Runway-Version": "2024-11-06",
                }

                response = requests.post(
                    "https://api.dev.runwayml.com/v1/video_upscale",
                    json=payload,
                    headers=headers,
                    timeout=60,
                )
                if response.status_code != 200:
                    error_body = response.text
                    logger.error(f"RunwayML VideoUpscale: API returned {response.status_code}: {error_body}")
                    raise ValueError(f"RunwayML API Error ({response.status_code}): {error_body}")
                task_resp = response.json()
                task_id = task_resp.get("id")
                if task_id:
                    self.publish_update_to_parameter("task_id_output", task_id)
                logger.info(f"RunwayML VideoUpscale: Task created with ID: {task_id}")

                for attempt in range(MAX_RETRIES):
                    time.sleep(RETRY_DELAY_SECONDS)
                    status_response = requests.get(
                        f"https://api.dev.runwayml.com/v1/tasks/{task_id}",
                        headers=headers,
                        timeout=30,
                    )
                    status_response.raise_for_status()
                    status_resp = status_response.json()
                    status = status_resp.get("status")
                    logger.info(
                        f"RunwayML VideoUpscale status (Task ID: {task_id}): {status} (Attempt {attempt + 1}/{MAX_RETRIES})"
                    )

                    if status == "SUCCEEDED":
                        output_url: str | None = None
                        out = status_resp.get("output", None)
                        if out:
                            if isinstance(out, list) and len(out) > 0:
                                item = out[0]
                                if isinstance(item, dict) and "url" in item:
                                    output_url = item.get("url")
                                elif isinstance(item, str) and item.startswith(("http://", "https://")):
                                    output_url = item
                            elif isinstance(out, dict) and "url" in out:
                                output_url = out.get("url")
                            elif isinstance(out, str) and out.startswith(("http://", "https://")):
                                output_url = out

                        if output_url:
                            logger.info(f"RunwayML VideoUpscale succeeded: {output_url}")
                            # Always save to static files (per request)
                            artifact = self._download_and_store_video(output_url, task_id)
                            self.publish_update_to_parameter("video_output", artifact)
                            return artifact
                        else:
                            err_msg = "RunwayML VideoUpscale task SUCCEEDED but no output URL found."
                            logger.error(err_msg)
                            self.publish_update_to_parameter("video_output", ErrorArtifact(err_msg))
                            return ErrorArtifact(err_msg)

                    if status == "FAILED":
                        error_msg = f"RunwayML VideoUpscale failed (Task ID: {task_id})."
                        if getattr(status_resp, "error", None):
                            error_msg += f" Reason: {getattr(status_resp, 'error', None)}"
                        logger.error(error_msg)
                        self.publish_update_to_parameter("video_output", ErrorArtifact(error_msg))
                        return ErrorArtifact(error_msg)

                timeout_msg = (
                    f"RunwayML VideoUpscale task (ID: {task_id}) timed out after "
                    f"{MAX_RETRIES * RETRY_DELAY_SECONDS} seconds."
                )
                logger.error(timeout_msg)
                self.publish_update_to_parameter("video_output", ErrorArtifact(timeout_msg))
                return ErrorArtifact(timeout_msg)

            except Exception as e:
                error_message = f"RunwayML VideoUpscale unexpected error: {type(e).__name__} - {e}"
                if hasattr(e, "status") and hasattr(e, "reason") and hasattr(e, "body"):
                    error_message = (
                        f"RunwayML API Error: Status {getattr(e, 'status', 'N/A')} - "
                        f"Reason: {getattr(e, 'reason', 'N/A')} - Body: {getattr(e, 'body', 'N/A')}"
                    )
                logger.exception(error_message)
                self.publish_update_to_parameter("video_output", ErrorArtifact(error_message))
                return ErrorArtifact(error_message)
            finally:
                # Clean up any artifact uploaded to Griptape Cloud during this run.
                self._public_video.delete_uploaded_artifact()

        yield upscale_async
