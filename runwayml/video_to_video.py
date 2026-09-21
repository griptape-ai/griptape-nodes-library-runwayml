import base64
import mimetypes
import os
import subprocess
import tempfile
from typing import Any

from api_surface import (
    ENDPOINT_VIDEO_TO_VIDEO,
    get_model,
    model_choices,
    single_file_output_formats,
)
from griptape.artifacts import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.retained_mode.griptape_nodes import logger
from griptape_nodes.traits.options import Options
from media import coerce_media_url_or_data_uri, prepare_media_data_uri
from runway_node import RunwayTaskNode

DEFAULT_MODEL = "aleph2"

# Aleph 2 letterboxes the input to reach a target ratio (expand/outpaint) rather than taking
# an output resolution, and it deprecated the older `ratio` field. This sentinel keeps
# `targetAspectRatio` out of the payload so the source framing is preserved untouched.
MATCH_INPUT_ASPECT_RATIO = "Match input"
DEFAULT_OUTPUT_FORMAT = "mp4"

# Where a single guidance image is placed in the input video.
KEYFRAME_START_SECONDS = 0

SUPPORTED_REFERENCE_IMAGE_FORMATS = frozenset({"image/jpeg", "image/jpg", "image/png", "image/webp"})

# Containers RunwayML ingests directly, so they are uploaded byte-for-byte.
RUNWAY_ACCEPTED_VIDEO_TYPES = frozenset(
    {"video/mp4", "video/webm", "video/quicktime", "video/mov", "video/ogg", "video/h264"}
)
DEFAULT_VIDEO_CONTENT_TYPE = "video/mp4"


class RunwayML_VideoToVideo(RunwayTaskNode):
    endpoint = ENDPOINT_VIDEO_TO_VIDEO
    output_filename = "output.mp4"
    output_parameter_name = "video_output"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "AI/RunwayML"
        self.description = "Generates a video from an input video and prompt using RunwayML."

        default_spec = get_model(DEFAULT_MODEL, ENDPOINT_VIDEO_TO_VIDEO)

        self.add_parameter(
            Parameter(
                name="video",
                input_types=["VideoUrlArtifact", "VideoArtifact"],
                type="VideoUrlArtifact",
                tooltip=(
                    f"The video to edit. Must be between {default_spec.min_input_video_seconds} and "
                    f"{default_spec.max_input_video_seconds} seconds long."
                ),
                ui_options={
                    "clickable_file_browser": True,
                    "expander": True,
                    "display_name": "Video or Path to Video",
                },
            )
        )
        self.add_parameter(
            ParameterString(
                name="prompt",
                default_value="",
                tooltip=(
                    "Text prompt describing the desired edit. Optional: an edit can also be driven "
                    "by a reference image or a target aspect ratio alone."
                ),
                # OUTPUT as well as INPUT so the prompt can travel with the result: an asset
                # manager publishing the output wants the prompt that produced it.
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY, ParameterMode.OUTPUT},
                multiline=True,
                placeholder_text="e.g., make it look like a rainy night",
            )
        )
        self.add_parameter(
            Parameter(
                name="reference_image",
                input_types=["ImageArtifact", "ImageUrlArtifact", "str"],
                type="ImageArtifact",
                tooltip=("Optional guidance image applied at the start of the clip. JPEG, PNG, or WebP only."),
                # OUTPUT as well as INPUT, for the same provenance reason as `prompt`.
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY, ParameterMode.OUTPUT},
                ui_options={"clickable_file_browser": True},
            )
        )

        with ParameterGroup(name="Generation Settings") as settings:
            model = ParameterString(
                name="model",
                default_value=DEFAULT_MODEL,
                tooltip="RunwayML model to use for generation.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            model.add_trait(Options(choices=model_choices(ENDPOINT_VIDEO_TO_VIDEO)))

            target_aspect_ratio = ParameterString(
                name="target_aspect_ratio",
                default_value=MATCH_INPUT_ASPECT_RATIO,
                tooltip=(
                    "Expand the frame to this aspect ratio, letterboxing the input before "
                    "generation so the model outpaints the new edges. Leave on "
                    f"'{MATCH_INPUT_ASPECT_RATIO}' to keep the source framing."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            target_aspect_ratio.add_trait(
                Options(choices=[MATCH_INPUT_ASPECT_RATIO, *default_spec.target_aspect_ratios])
            )

            self._add_seed_parameters(settings)

            public_figure_threshold = ParameterString(
                name="public_figure_threshold",
                default_value="auto",
                tooltip="Public figure threshold for content moderation.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            public_figure_threshold.add_trait(Options(choices=["auto", "low"]))
        self.add_node_element(settings)

        # ProRes is encoded by RunwayML and downloaded as a finished .mov. Nothing here
        # encodes ProRes locally.
        with ParameterGroup(name="Professional Output", collapsed=True) as pro_output:
            output_format = ParameterString(
                name="output_format",
                default_value=DEFAULT_OUTPUT_FORMAT,
                tooltip="Delivery container. Anything other than mp4 carries a per-second credit surcharge.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            output_format.add_trait(Options(choices=single_file_output_formats(default_spec.output_formats)))

            prores_profile = ParameterString(
                name="prores_profile",
                default_value="4444",
                tooltip="ProRes tier. Only applies when the output format is prores.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            prores_profile.add_trait(Options(choices=list(default_spec.prores_profiles)))
        self.add_node_element(pro_output)

        self.add_parameter(
            Parameter(
                name="video_output",
                output_type="VideoUrlArtifact",
                type="VideoUrlArtifact",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="The generated video, saved into project files.",
                ui_options={"placeholder_text": "", "is_full_width": True, "pulse_on_run": True},
            )
        )

        self._add_output_file_parameter()
        self._create_status_parameters(result_details_placeholder="Generation progress will appear here.")

        self._sync_prores_visibility(DEFAULT_OUTPUT_FORMAT)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "output_format":
            output_format = str(value or DEFAULT_OUTPUT_FORMAT)
            self._sync_output_extension(output_format)
            self._sync_prores_visibility(output_format)

        return super().after_value_set(parameter, value)

    def _transcode_video_file(self, video_file_path: str) -> str | None:
        """Transcode a video to H.264 so RunwayML receives a known-good format.

        Returns the transcoded path, or None when ffmpeg is unavailable or fails, in which
        case the caller sends the original bytes.
        """
        try:
            logger.info("RunwayML V2V: Attempting to transcode video to ensure compatibility...")
            transcoded_file = video_file_path + ".transcoded.mp4"

            cmd = [
                "ffmpeg",
                "-i",
                video_file_path,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-strict",
                "experimental",
                transcoded_file,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603

            if os.path.exists(transcoded_file) and os.path.getsize(transcoded_file) > 0:
                logger.info(f"RunwayML V2V: Successfully transcoded video to H.264: {transcoded_file}")
                return transcoded_file

            logger.warning(f"RunwayML V2V: Failed to transcode video. Error: {result.stderr}")
        except (FileNotFoundError, subprocess.SubprocessError) as e:
            logger.info(f"RunwayML V2V: ffmpeg not available, skipping video transcoding: {e!s}")
            return None
        return None

    def _read_to_data_uri(self, source: str) -> str:
        """Read `source` (local path or HTTP(S) URL), transcode if needed, and base64 it.

        Raises:
            ValueError: If the source cannot be read.
        """
        try:
            video_bytes = File(source).read_bytes()
        except FileLoadError as e:
            msg = f"RunwayML V2V: failed to read video from {source!r}: {e}"
            raise ValueError(msg) from e

        # Only re-encode a container RunwayML will not take. Transcoding unconditionally cost a
        # generation of quality on every local input, and this node can be asked to deliver
        # ProRes 4444 -- where handing the model an 8-bit 4:2:0 source defeats the request.
        # Only skip the transcode when the extension positively identifies an accepted
        # container. `guess_type` returns None for an extensionless path or a query-string URL,
        # and defaulting that to video/mp4 would send unknown bytes up mislabelled and
        # un-normalized -- the inputs that most need the transcode.
        guessed = mimetypes.guess_type(source)[0]
        if guessed in RUNWAY_ACCEPTED_VIDEO_TYPES:
            logger.info("RunwayML V2V: %s is a format RunwayML accepts; sending it unchanged", guessed)
            return f"data:{guessed};base64,{base64.b64encode(video_bytes).decode('utf-8')}"

        return self._transcode_to_data_uri(video_bytes, guessed or "an unrecognized container")

    def _transcode_to_data_uri(self, video_bytes: bytes, content_type: str) -> str:
        """Normalize bytes RunwayML would reject into an H.264 mp4 data URI.

        Falls back to the original bytes when ffmpeg is unavailable, so the request still gets
        made and RunwayML reports what it does not like.
        """
        logger.info("RunwayML V2V: %s needs normalizing for RunwayML", content_type)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(video_bytes)
            temp_file_path = temp_file.name

        transcoded_path = None
        try:
            transcoded_path = self._transcode_video_file(temp_file_path)
            file_to_encode = transcoded_path if transcoded_path else temp_file_path
            with open(file_to_encode, "rb") as f:
                base64_data = base64.b64encode(f.read()).decode("utf-8")
        finally:
            # In a finally block so a failed transcode or read does not leave temp files behind
            # on every attempt.
            for path in (temp_file_path, transcoded_path):
                if path:
                    try:
                        os.unlink(path)
                    except OSError as e:
                        logger.debug("RunwayML V2V: could not remove temp file %s: %s", path, e)

        return f"data:{DEFAULT_VIDEO_CONTENT_TYPE};base64,{base64_data}"

    def _get_video_data_uri(self, param_name: str) -> str | None:
        """Resolve a video input to a value the /v1/video_to_video endpoint accepts.

        ``https://`` URLs, ``runway://`` URIs, and ``data:video/...`` URIs pass through
        unchanged. Anything else (local paths, ``{inputs}/...`` macro paths,
        ``http://`` URLs, ``file://`` URIs) is read via ``File``, optionally transcoded
        with ffmpeg, and returned as a ``data:video/mp4;base64,...`` URI so RunwayML
        receives a known-good format.
        """
        media_url = coerce_media_url_or_data_uri(self.get_parameter_value(param_name), kind="video")
        if not media_url:
            return None

        if media_url.startswith(("data:video/", "https://", "runway://")):
            return media_url

        return self._read_to_data_uri(media_url)

    def _get_image_data_uri(self, param_name: str) -> str | None:
        """Resolve a reference-image input to a data URI in a RunwayML-supported format.

        RunwayML's ``/v1/video_to_video`` endpoint only accepts JPEG, PNG, or WebP
        reference images. HTTPS URLs are downloaded so the format can be inspected;
        data URIs are validated in place. Unsupported formats raise ``ValueError``.
        """
        # Force an HTTPS download via ``pass_through_schemes=()`` so we always end up with a
        # data URI whose content type we can inspect.
        data_uri = prepare_media_data_uri(
            self.get_parameter_value(param_name),
            kind="image",
            node_name="RunwayML V2V",
            pass_through_schemes=(),
        )
        if not data_uri:
            return None

        content_type = data_uri.split(":", 1)[1].split(";", 1)[0]
        if content_type not in SUPPORTED_REFERENCE_IMAGE_FORMATS:
            msg = (
                f"Unsupported reference image format: {content_type}. "
                "Supported formats are image/jpeg, image/png, and image/webp."
            )
            raise ValueError(msg)
        return data_uri

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = super().validate_before_node_run() or []

        # Presence only: the engine calls this synchronously on its event loop, so reading and
        # encoding media here would stall every other node. `build_payload` runs in a thread
        # and reports anything unreadable from there.
        if not self.get_parameter_value("video"):
            errors.append(
                ValueError(
                    f"Attempted to edit a video on '{self.name}'. Failed because no input video is set. "
                    "Connect a video or choose a file."
                )
            )

        try:
            get_model(str(self.get_parameter_value("model") or ""), ENDPOINT_VIDEO_TO_VIDEO)
        except ValueError as e:
            errors.append(e)

        # Aleph 2 treats promptText as optional, so an empty prompt is not an error here, and
        # prompt length is RunwayML's call rather than ours.
        return errors or None

    def build_payload(self) -> dict[str, Any]:
        model_name = str(self.get_parameter_value("model") or DEFAULT_MODEL)
        spec = get_model(model_name, ENDPOINT_VIDEO_TO_VIDEO)

        video_uri = self._get_video_data_uri("video")
        if not video_uri:
            msg = f"Attempted to edit a video on '{self.name}'. Failed because the input video could not be read."
            raise ValueError(msg)

        payload: dict[str, Any] = {
            "model": model_name,
            "videoUri": video_uri,
            "seed": self._resolve_seed(),
            "contentModeration": self._content_moderation(),
        }

        # Runway rejects an empty promptText, so omit the key entirely rather than sending ""
        # when the edit is driven by a reference image or target ratio.
        prompt = str(self.get_parameter_value("prompt") or "").strip()
        if prompt:
            payload["promptText"] = prompt

        target_aspect_ratio = str(self.get_parameter_value("target_aspect_ratio") or MATCH_INPUT_ASPECT_RATIO)
        if target_aspect_ratio != MATCH_INPUT_ASPECT_RATIO:
            payload["targetAspectRatio"] = target_aspect_ratio

        # `_get_image_data_uri` returns None both when nothing is connected and when the read
        # failed, so presence is checked separately. Collapsing the two would drop an unreadable
        # reference, bill a full edit that ignored it, and still report success.
        if self.get_parameter_value("reference_image"):
            reference_image_uri = self._get_image_data_uri("reference_image")
            if not reference_image_uri:
                msg = (
                    f"Attempted to edit a video on '{self.name}'. Failed because the reference image "
                    "could not be read. Check that the file still exists, or clear the input."
                )
                raise ValueError(msg)
            # `references` was gen4_aleph's field. aleph2 declares additionalProperties:false
            # and does not accept it, so sending it 400s the request outright; timed `keyframes`
            # replaced it. One guidance image applies at the start of the clip.
            payload["keyframes"] = [{"uri": reference_image_uri, "seconds": KEYFRAME_START_SECONDS}]

        self._attach_output_format(payload, spec, default_format=DEFAULT_OUTPUT_FORMAT)
        return payload

    def build_artifact(self, location: str) -> VideoUrlArtifact:
        return VideoUrlArtifact(value=location, name="runwayml_video_to_video")
