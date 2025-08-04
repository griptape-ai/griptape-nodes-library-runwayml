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

from griptape.artifacts import TextArtifact, UrlArtifact, ImageArtifact, ImageUrlArtifact, ErrorArtifact
from griptape_nodes.traits.options import Options

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterGroup
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode, BaseNode
from griptape_nodes.retained_mode.griptape_nodes import logger

SERVICE = "RunwayML"
API_KEY_ENV_VAR = "RUNWAYML_API_SECRET"
DEFAULT_MODEL = "act_two"

# Aspect ratios supported by Runway API
ASPECT_RATIOS = [
    "1280:720", "720:1280", "1104:832", "832:1104", "960:960", "1584:672"
]
DEFAULT_ASPECT_RATIO = "1280:720"

# Expression intensity options
EXPRESSION_INTENSITY_OPTIONS = [1, 2, 3, 4, 5]
DEFAULT_EXPRESSION_INTENSITY = 3


class VideoUrlArtifact(UrlArtifact):
    """
    Artifact that contains a URL to a video.
    """

    def __init__(self, url: str, name: str | None = None):
        super().__init__(value=url, name=name or self.__class__.__name__)


class RunwayML_CharacterPerformance(ControlNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "AI/RunwayML"
        self.description = "Generates a character performance video using RunwayML's Act Two API."
        self.metadata["author"] = "Griptape"
        self.metadata["dependencies"] = {"pip_dependencies": ["runwayml", "requests"]}
            
        # Media Inputs Group
        with ParameterGroup(name="Media Inputs") as media_inputs_group:
            Parameter(
                name="character_image",
                input_types=["ImageArtifact", "ImageUrlArtifact", "str"],
                type="ImageArtifact", 
                tooltip="[REQUIRED*] Input image of the character. Either character_image OR character_video must be provided. Accepts ImageArtifact, ImageUrlArtifact, a public URL string, or a base64 data URI string.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"clickable_file_browser": True}
            )
            
            Parameter(
                name="character_video",
                input_types=["VideoArtifact", "UrlArtifact", "VideoUrlArtifact", "str"],
                type="UrlArtifact", 
                tooltip="[REQUIRED*] Character video for the performance. Either character_image OR character_video must be provided. Accepts UrlArtifact, VideoUrlArtifact, a public URL string, or a base64 data URI string.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"clickable_file_browser": True}
            )
            
            Parameter(
                name="reference_video",
                input_types=["VideoArtifact", "UrlArtifact", "VideoUrlArtifact", "str"],
                type="UrlArtifact", 
                tooltip="[REQUIRED] Reference video for the character. Accepts UrlArtifact, VideoUrlArtifact, a public URL string, or a base64 data URI string.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"clickable_file_browser": True}
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
                default_value=0,
                tooltip="[OPTIONAL] Seed for generation. 0 for random.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
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

    def _get_data_uri(self, param_name: str) -> str | None:
        """
        Extracts and processes data URI from different input types.
        Works with images and videos. Always returns a data URI, never a URL.
        """
        input_value = self.get_parameter_value(param_name)
        
        if not input_value:
            return None

        # Get the content type prefix based on parameter name
        is_image = param_name == "character_image"
        content_type_prefix = "data:image" if is_image else "data:video"
        expected_media_type = "image/png" if is_image else "video/mp4"
        logger.info(f"RunwayML Act Two: Processing {param_name} with type {type(input_value).__name__}")

        # Handle ImageArtifact (shouldn't happen but just in case)
        if isinstance(input_value, ImageArtifact):
            # Convert format to media type
            format_to_media_type = {
                "JPEG": "image/jpeg",
                "PNG": "image/png",
                "WEBP": "image/webp",
                "GIF": "image/gif"
            }
            media_type = format_to_media_type.get(getattr(input_value, 'format', ''), "image/png")
            
            if not input_value.base64.startswith(f"data:{media_type};base64,"):
                return f"data:{media_type};base64,{input_value.base64}"
            return input_value.base64
            
        # Handle URL artifacts
        elif isinstance(input_value, (ImageUrlArtifact, UrlArtifact)):
            url_value = input_value.value
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
            logger.info(f"RunwayML Act Two: received dict for {param_name}: {input_value}")
            input_type = input_value.get("type")
            url_from_dict = input_value.get("value")
            base64_from_dict = input_value.get("base64")
            media_type_from_dict = input_value.get("media_type", "video/mp4")
            
            if (input_type in ["ImageUrlArtifact", "UrlArtifact", "VideoUrlArtifact"]) and url_from_dict:
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
            elif input_type == "ImageArtifact" and base64_from_dict:
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
        
        # Get media inputs
        character_image_uri = self._get_data_uri("character_image")
        character_video_uri = self._get_data_uri("character_video")
        reference_video_uri = self._get_data_uri("reference_video")
        
        # Validate required parameters - need either image OR video for character
        if not any([character_image_uri, character_video_uri]):
            errors.append(ValueError("Either 'character_image' OR 'character_video' must be provided."))
        
        # Should not provide both character image and video
        if character_image_uri and character_video_uri:
            errors.append(ValueError("Provide either 'character_image' OR 'character_video', not both."))
        
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
        character_image_uri = self._get_data_uri("character_image")
        character_video_uri = self._get_data_uri("character_video")
        reference_video_uri = self._get_data_uri("reference_video")
        ratio_val = str(self.get_parameter_value("ratio") or DEFAULT_ASPECT_RATIO)
        seed_val = self.get_parameter_value("seed") or 0
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
                        "type": "image" if character_image_uri else "video",
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
                if seed_val and seed_val != 0:
                    task_payload["seed"] = seed_val

                logger.info(f"RunwayML Act Two: Creating task with payload keys: {list(task_payload.keys())}")
                
                # Print detailed debug information about the payload
                
                # Validate and debug the data URIs
                def validate_data_uri(uri, param_name):
                    
                    if not uri or not uri.startswith("data:"):
                        print(f"WARNING: {param_name} is not a data URI: {uri[:50]}...")
                        return False, {}
                    
                    try:
                        # Get content type and base64 data
                        header, encoded = uri.split(",", 1)
                        content_type = header.split(":")[1].split(";")[0]
                        
                        # Decode the full data
                        decoded = base64.b64decode(encoded)
                        
                        # Create appropriate file extension based on content type
                        ext_map = {
                            "image/jpeg": ".jpg",
                            "image/png": ".png",
                            "image/webp": ".webp",
                            "image/gif": ".gif",
                            "video/mp4": ".mp4",
                            "video/webm": ".webm",
                            "video/ogg": ".ogv"
                        }
                        ext = ext_map.get(content_type, ".bin")
                        
                        # Write the decoded data to a temporary file
                        temp_dir = tempfile.gettempdir()
                        temp_file = os.path.join(temp_dir, f"runway_debug_{param_name}_{int(time.time())}{ext}")
                        
                        with open(temp_file, "wb") as f:
                            f.write(decoded)
                        
                        # Extract and analyze metadata for videos using ffprobe
                        metadata = {}
                        if content_type.startswith('video/'):
                            try:
                                # First try ffprobe (if available)
                                try:
                                    # Use ffprobe to get detailed video metadata
                                    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", temp_file]
                                    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                                    
                                    if result.returncode == 0:
                                        metadata = json.loads(result.stdout)
                                        print(f"\nVIDEO METADATA ANALYSIS for {param_name}:")
                                        print("=====================================")
                                        
                                        # Extract useful information
                                        if 'streams' in metadata:
                                            for idx, stream in enumerate(metadata['streams']):
                                                stream_type = stream.get('codec_type', 'unknown')
                                                print(f"Stream #{idx} ({stream_type}):")
                                                
                                                # Print important properties based on stream type
                                                if stream_type == 'video':
                                                    print(f"  - Codec: {stream.get('codec_name', 'unknown')}")
                                                    print(f"  - Resolution: {stream.get('width', '?')}x{stream.get('height', '?')}")
                                                    print(f"  - Framerate: {stream.get('r_frame_rate', 'unknown')}")
                                                    print(f"  - Pixel format: {stream.get('pix_fmt', 'unknown')}")
                                                    print(f"  - Duration: {stream.get('duration', 'unknown')} seconds")
                                                    print(f"  - Bitrate: {stream.get('bit_rate', 'unknown')} bits/s")
                                                elif stream_type == 'audio':
                                                    print(f"  - Codec: {stream.get('codec_name', 'unknown')}")
                                                    print(f"  - Sample rate: {stream.get('sample_rate', 'unknown')} Hz")
                                                    print(f"  - Channels: {stream.get('channels', 'unknown')}")
                                                    print(f"  - Duration: {stream.get('duration', 'unknown')} seconds")
                                                    
                                        if 'format' in metadata:
                                            fmt = metadata['format']
                                            print(f"\nFormat:")
                                            print(f"  - Format name: {fmt.get('format_name', 'unknown')}")
                                            print(f"  - Duration: {fmt.get('duration', 'unknown')} seconds")
                                            print(f"  - Size: {fmt.get('size', 'unknown')} bytes")
                                            print(f"  - Bit rate: {fmt.get('bit_rate', 'unknown')} bits/s")
                                        
                                        # RunwayML likely requirements - print warnings for potential issues
                                        print("\nPotential issues for RunwayML:")
                                        
                                        # Check common issues
                                        has_issues = False
                                        
                                        # Check for video stream
                                        has_video = any(s.get('codec_type') == 'video' for s in metadata.get('streams', []))
                                        if not has_video:
                                            print("  ❌ No video stream found!")
                                            has_issues = True
                                        
                                        # Check for common unsupported video codecs
                                        video_streams = [s for s in metadata.get('streams', []) if s.get('codec_type') == 'video']
                                        for vs in video_streams:
                                            codec = vs.get('codec_name', '').lower()
                                            if codec not in ['h264', 'avc1', 'mp4v']:
                                                print(f"  ⚠️ Potentially unsupported video codec: {codec} (RunwayML prefers H.264)")
                                                has_issues = True
                                            
                                        # Check for odd resolutions
                                        for vs in video_streams:
                                            width = vs.get('width', 0)
                                            height = vs.get('height', 0)
                                            if width % 2 != 0 or height % 2 != 0:
                                                print(f"  ⚠️ Non-even dimensions: {width}x{height} (dimensions should be even numbers)")
                                                has_issues = True
                                            
                                            # Check pixel format
                                            pix_fmt = vs.get('pix_fmt', '')
                                            if pix_fmt != 'yuv420p':
                                                print(f"  ⚠️ Non-standard pixel format: {pix_fmt} (RunwayML prefers yuv420p)")
                                                has_issues = True
                                        
                                        # Check if file might be too large
                                        file_size = int(metadata.get('format', {}).get('size', 0))
                                        if file_size > 50 * 1024 * 1024:  # 50MB
                                            print(f"  ⚠️ File may be too large: {file_size/1024/1024:.2f}MB")
                                            has_issues = True
                                        
                                        if not has_issues:
                                            print("  ✅ No obvious issues detected")
                                    else:
                                        print(f"Failed to extract video metadata with ffprobe: {result.stderr}")
                                except (FileNotFoundError, subprocess.SubprocessError) as e:
                                    print(f"Note: ffprobe not available, skipping detailed video analysis: {str(e)}")
                                except json.JSONDecodeError:
                                    print("Error parsing ffprobe output")
                            except Exception as meta_error:
                                print(f"Error during metadata analysis: {str(meta_error)}")
                        
                        
                        return True, metadata
                    except Exception as e:
                        print(f"ERROR: {param_name} data URI validation failed: {str(e)}")
                        return False, {}
                
                # Validate character URI
                character_uri = task_payload["character"]["uri"]
                character_type = task_payload["character"]["type"]
                character_valid, character_metadata = validate_data_uri(character_uri, f"character_{character_type}")
                
                # Only transcode character video, not image
                transcoded_character_uri = None
                if character_type == "video" and character_valid:
                    print("\n===== CHARACTER VIDEO VALIDATION =====")
                    
                    # Additional codec validation for character video
                    try:
                        # Make a clean temporary file for media validation
                        char_temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                        char_temp_file.close()
                        
                        # Extract content
                        header, encoded = character_uri.split(",", 1)
                        content_type = header.split(":")[1].split(";")[0]
                        decoded = base64.b64decode(encoded)
                        
                        with open(char_temp_file.name, "wb") as f:
                            f.write(decoded)
                            
                        print(f"Character video test file: {char_temp_file.name}")
                        
                        # Try to transcode the video to a format known to work with RunwayML
                        try:
                            print("Attempting to transcode character video to ensure compatibility...")
                            char_transcoded_file = char_temp_file.name + ".transcoded.mp4"
                            
                            # Use ffmpeg to transcode to a known good format
                            cmd = ["ffmpeg", "-i", char_temp_file.name, "-c:v", "libx264", "-preset", "fast", 
                                   "-pix_fmt", "yuv420p", "-c:a", "aac", "-strict", "experimental", char_transcoded_file]
                            result = subprocess.run(cmd, capture_output=True, text=True)
                            
                            if os.path.exists(char_transcoded_file) and os.path.getsize(char_transcoded_file) > 0:
                                print(f"✅ Successfully transcoded to H.264. Using this file instead: {char_transcoded_file}")
                                
                                # Convert transcoded file to data URI for API use
                                with open(char_transcoded_file, "rb") as f:
                                    transcoded_data = f.read()
                                    transcoded_b64 = base64.b64encode(transcoded_data).decode("utf-8")
                                    transcoded_character_uri = f"data:video/mp4;base64,{transcoded_b64}"
                                    print(f"Created data URI from transcoded file (length: {len(transcoded_character_uri)})")
                            else:
                                print(f"❌ Failed to transcode video. Error: {result.stderr}")
                                
                        except (FileNotFoundError, subprocess.SubprocessError) as e:
                            print(f"Note: ffmpeg not available, skipping character video transcoding: {str(e)}")
                    except Exception as e:
                        print(f"Error during character video validation: {str(e)}")
                    
                    # Use transcoded character video if available
                    if transcoded_character_uri:
                        print("Using transcoded character video for API request")
                        task_payload["character"]["uri"] = transcoded_character_uri
                
                # Validate reference URI
                reference_uri = task_payload["reference"]["uri"]
                reference_valid, reference_metadata = validate_data_uri(reference_uri, "reference_video")
                
                # Special debug for reference video since that's where we're getting the error
                transcoded_reference_uri = None  # Will store the transcoded data URI if successful
                
                if reference_valid:
                    print("\n===== REFERENCE VIDEO VALIDATION =====")
                    
                    # Additional codec validation
                    try:
                        # Make a clean temporary file specifically for media validation
                        debug_temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                        debug_temp_file.close()
                        
                        # Extract content
                        header, encoded = reference_uri.split(",", 1)
                        content_type = header.split(":")[1].split(";")[0]
                        decoded = base64.b64decode(encoded)
                        
                        with open(debug_temp_file.name, "wb") as f:
                            f.write(decoded)
                            
                        print(f"Reference video test file: {debug_temp_file.name}")
                        
                        # Try to transcode the video to a format known to work with RunwayML
                        try:
                            print("Attempting to transcode reference video to ensure compatibility...")
                            transcoded_file = debug_temp_file.name + ".transcoded.mp4"
                            
                            # Use ffmpeg to transcode to a known good format
                            cmd = ["ffmpeg", "-i", debug_temp_file.name, "-c:v", "libx264", "-preset", "fast", 
                                   "-pix_fmt", "yuv420p", "-c:a", "aac", "-strict", "experimental", transcoded_file]
                            result = subprocess.run(cmd, capture_output=True, text=True)
                            
                            if os.path.exists(transcoded_file) and os.path.getsize(transcoded_file) > 0:
                                print(f"✅ Successfully transcoded to H.264. Using this file instead: {transcoded_file}")
                                
                                # Convert transcoded file to data URI for API use
                                with open(transcoded_file, "rb") as f:
                                    transcoded_data = f.read()
                                    transcoded_b64 = base64.b64encode(transcoded_data).decode("utf-8")
                                    transcoded_reference_uri = f"data:video/mp4;base64,{transcoded_b64}"
                                    print(f"Created data URI from transcoded file (length: {len(transcoded_reference_uri)})")
                                
                            else:
                                print(f"❌ Failed to transcode video. Error: {result.stderr}")
                                
                        except (FileNotFoundError, subprocess.SubprocessError) as e:
                            print(f"Note: ffmpeg not available, skipping transcoding: {str(e)}")
                    except Exception as e:
                        print(f"Error during reference video validation: {str(e)}")
                
                # Use transcoded reference video if available
                if transcoded_reference_uri:
                    print("Using transcoded reference video for API request")
                    task_payload["reference"]["uri"] = transcoded_reference_uri

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
                            return video_artifact
                        else:
                            logger.error(f"RunwayML Act Two task SUCCEEDED but no output URL found. Output structure: {output}")
                            err_msg = "RunwayML Act Two task SUCCEEDED but no output URL found."
                            self.publish_update_to_parameter("video_output", ErrorArtifact(err_msg))
                            return ErrorArtifact(err_msg)
                    
                    elif status == 'FAILED':
                        error_msg = f"RunwayML Act Two generation failed (Task ID: {task_id})."
                        error_detail = task_status.get("error")
                        if error_detail:
                            error_msg += f" Reason: {error_detail}"
                        logger.error(error_msg)
                        self.publish_update_to_parameter("video_output", ErrorArtifact(error_msg))
                        return ErrorArtifact(error_msg)

                timeout_msg = f"RunwayML Act Two task (ID: {task_id}) timed out after {max_retries * retry_delay} seconds."
                logger.error(timeout_msg)
                self.publish_update_to_parameter("video_output", ErrorArtifact(timeout_msg))
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
                return ErrorArtifact(error_message)

        yield generate_character_performance_async