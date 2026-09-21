"""Shared base class for the RunwayML generation nodes.

Every generation node needs the same scaffolding: seed handling, a status readout, the
submit/poll/download round trip, and success or failure routing. Keeping it here means a
node module only describes its own parameters and payload.
"""

from __future__ import annotations

import asyncio
import random
from pathlib import PurePosixPath
from typing import Any

from api_surface import (
    OUTPUT_FORMAT_EXTENSIONS,
    PRORES_OUTPUT_FORMATS,
    SEED_MAX,
    SEED_MIN,
    ModelSpec,
    single_file_output_formats,
)
from griptape.artifacts import UrlArtifact
from griptape_nodes.exe_types.core_types import ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File, FileLoadError, FileWriteError
from griptape_nodes.retained_mode.griptape_nodes import logger
from griptape_nodes.traits.options import Options
from runway_client import RunwayClient, RunwayError, get_api_key

SEED_CONTROL_FIXED = "fixed"
SEED_CONTROL_INCREMENT = "increment"
SEED_CONTROL_DECREMENT = "decrement"
SEED_CONTROL_RANDOMIZE = "randomize"
SEED_CONTROL_MODES = [
    SEED_CONTROL_FIXED,
    SEED_CONTROL_INCREMENT,
    SEED_CONTROL_DECREMENT,
    SEED_CONTROL_RANDOMIZE,
]

DEFAULT_SEED = 12345


class RunwayTaskNode(SuccessFailureNode):
    """Base for nodes that submit one RunwayML task and save its output.

    Subclasses set `endpoint`, `output_filename`, and `output_parameter_name`, then
    implement `build_payload`. `aprocess` handles the rest.
    """

    endpoint: str = ""
    output_filename: str = "output.mp4"
    output_parameter_name: str = "video_output"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.metadata["author"] = "Griptape"

        # Declared here rather than in each subclass because `aprocess` publishes to it, and
        # `publish_update_to_parameter` raises on a parameter that was never added.
        self.add_parameter(
            ParameterString(
                name="task_id_output",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="The Task ID of the RunwayML job, for cross-referencing with RunwayML's dashboard.",
            )
        )

    # --- Scaffolding subclasses opt into ---

    def _add_seed_parameters(self, group: ParameterGroup | None = None) -> None:
        """Add the `seed` / `seed_control` pair, inside `group` when one is given."""
        seed = ParameterInt(
            name="seed",
            default_value=DEFAULT_SEED,
            min_val=SEED_MIN,
            max_val=SEED_MAX,
            tooltip="Seed for reproducible generation. Reusing a seed with identical settings repeats a result.",
            allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT, ParameterMode.OUTPUT},
        )
        seed_control = ParameterString(
            name="seed_control",
            default_value=SEED_CONTROL_RANDOMIZE,
            tooltip=(
                "How the seed changes between runs: fixed keeps it, increment adds 1, "
                "decrement subtracts 1, randomize picks a new one."
            ),
            allowed_modes={ParameterMode.PROPERTY},
        )
        seed_control.add_trait(Options(choices=SEED_CONTROL_MODES))

        if group is None:
            self.add_parameter(seed)
            self.add_parameter(seed_control)
        else:
            group.add_child(seed)
            group.add_child(seed_control)

    def _add_output_file_parameter(self) -> None:
        self._output_file = ProjectFileParameter(node=self, name="output_file", default_filename=self.output_filename)
        self._output_file.add_parameter()

    def _sync_output_extension(self, output_format: str) -> None:
        """Point the output filename at the container the chosen format actually delivers.

        Runway returns ProRes and the mastering profiles as `.mov`, and frame sequences as a
        `.zip`. The user's base name is kept and only the extension is rewritten, so the
        change is visible in the node rather than silently applied at write time.
        """
        extension = OUTPUT_FORMAT_EXTENSIONS.get(output_format)
        if extension is None:
            return

        current = str(self.get_parameter_value("output_file") or self.output_filename)
        updated = str(PurePosixPath(current).with_suffix(extension))
        if updated != current:
            self.set_parameter_value("output_file", updated)

    def _content_moderation(self) -> dict[str, str]:
        """Build the moderation fragment.

        `contentModeration` declares `additionalProperties: false` upstream with the single
        property `publicFigureThreshold`, so the key name is spelled once here rather than in
        every node that sends it.
        """
        # The nodes shipped with two names for this control, and settling on one would reset the
        # stored value in every saved workflow that set it. Resolved by which parameter the node
        # declares, not by an `or` that would silently prefer one.
        for name in ("content_moderation", "public_figure_threshold"):
            if self.get_parameter_by_name(name) is not None:
                return {"publicFigureThreshold": str(self.get_parameter_value(name) or "auto")}

        msg = f"Node '{self.name}' sends contentModeration but declares no moderation parameter."
        raise ValueError(msg)

    def _sync_prores_visibility(self, output_format: str) -> None:
        """Show the ProRes tier only when the chosen container carries a ProRes stream.

        `proresProfile` is ignored by RunwayML for every other container, so leaving the control
        visible offers a setting that cannot do anything.
        """
        if output_format in PRORES_OUTPUT_FORMATS:
            self.show_parameter_by_name("prores_profile")
        else:
            self.hide_parameter_by_name("prores_profile")

    def _attach_output_format(
        self,
        payload: dict[str, Any],
        spec: ModelSpec,
        *,
        default_format: str,
        always_send: bool = False,
    ) -> None:
        """Attach `outputFormat`, and `proresProfile` when the container carries a ProRes stream.

        Lives on the base because all four nodes that offer a delivery format need the same four
        steps, and when each had its own copy they drifted -- different error wording, a redundant
        model lookup, and the frame-sequence rejection written four times.

        `always_send` is for a node whose default is not mp4 (Video-to-HDR defaults to `hdr10` and
        has no plain-mp4 escape), so the field is never omitted.

        Raises:
            ValueError: If the requested format is not one this model can deliver as a single
                file. This one check stays local because it is a limit of this library rather
                than of the API -- RunwayML will happily return a zip of frames, and saving that
                as a video is worse than refusing it. `output_format` accepts an incoming
                connection, so the dropdown's filtering is not a gate on its own.
        """
        output_format = str(self.get_parameter_value("output_format") or default_format)
        if output_format == default_format and not always_send:
            return

        deliverable = single_file_output_formats(spec.output_formats)
        if output_format not in deliverable:
            offered = ", ".join(deliverable) if deliverable else "only mp4"
            msg = (
                f"Attempted to deliver '{output_format}' from '{self.name}'. Failed because "
                f"{spec.model_id} cannot deliver that format. It supports: {offered}."
            )
            raise ValueError(msg)

        payload["outputFormat"] = output_format
        if output_format in PRORES_OUTPUT_FORMATS:
            # Not checked against the tier the container serves. RunwayML rejects an unavailable
            # pair itself, before billing, and names its own constraint better than we can -- and
            # a local copy of that rule would start blocking valid requests the moment RunwayML
            # relaxed it. Omitted when empty rather than sent blank: `proresProfile` is optional
            # upstream and RunwayML picks a tier when it is absent, so sending "" would
            # manufacture a rejection that omitting the key avoids.
            profile = str(self.get_parameter_value("prores_profile") or "")
            if profile:
                payload["proresProfile"] = profile

    def _resolve_seed(self) -> int:
        """Apply `seed_control` and return the seed to send.

        The resolved value is written back to the `seed` parameter so the next run can step
        from it and so the OUTPUT reports the seed that was actually used.
        """
        # The `seed` parameter is the only record of the last seed used, because the resolved
        # value is written back to it below. Keeping a separate counter alongside it would give
        # two answers to "what did the last run use", and the parameter is the one that
        # survives a save/reload.
        stored = self.get_parameter_value("seed")
        # `is None`, not falsy: 0 is a seed RunwayML accepts (SEED_MIN). Treating it as unset
        # would send a different seed than the node displays, and because the resolved value is
        # written back below, it would overwrite the user's 0 permanently.
        requested = DEFAULT_SEED if stored is None else int(stored)
        mode = str(self.get_parameter_value("seed_control") or SEED_CONTROL_RANDOMIZE)

        # Literals, not the SEED_CONTROL_* constants: a bare name in a `case` is a capture
        # pattern that binds and always matches, so substituting the constants here would
        # silently make the first branch swallow every mode.
        match mode:
            case "fixed":
                seed = requested
            case "increment":
                seed = requested + 1
            case "decrement":
                seed = requested - 1
            case "randomize":
                seed = random.randint(SEED_MIN, SEED_MAX)  # noqa: S311
            case _:
                msg = f"Unknown seed control mode: {mode!r}. Expected one of {SEED_CONTROL_MODES}."
                raise ValueError(msg)

        seed = max(SEED_MIN, min(seed, SEED_MAX))

        # `seed` is an OUTPUT, and the default control is `randomize`. Without writing the
        # resolved value back, wiring the seed downstream to reproduce a result hands over the
        # pre-run number instead of the one that produced it. Setting the parameter (not just
        # the output) also makes `increment`/`decrement` chain from the seed actually used.
        self.set_parameter_value("seed", seed)
        self.parameter_output_values["seed"] = seed
        self.publish_update_to_parameter("seed", seed)
        return seed

    # --- Validation ---

    def validate_before_workflow_run(self) -> list[Exception] | None:
        """Fail the whole run up front when no API key is configured.

        This belongs at workflow scope rather than node scope: a missing key affects every
        RunwayML node in the graph, and finding out before anything executes beats
        discovering it after an earlier node has already spent credits.
        """
        errors = super().validate_before_workflow_run() or []
        try:
            get_api_key()
        except RunwayError as e:
            errors.append(e)

        # Checked here rather than at download time: `_save_output` is the first thing to touch
        # `_output_file`, and by then the generation has already been paid for.
        if getattr(self, "_output_file", None) is None:
            errors.append(
                ValueError(
                    f"Node '{self.name}' is missing its output file parameter. Its class must call "
                    "_add_output_file_parameter() during __init__."
                )
            )
        if not self.endpoint:
            errors.append(ValueError(f"Node '{self.name}' does not declare which RunwayML endpoint it calls."))

        return errors or None

    # --- Execution ---

    def build_payload(self) -> dict[str, Any]:
        """Return the JSON body for this node's endpoint."""
        raise NotImplementedError

    def build_artifact(self, location: str) -> UrlArtifact:
        """Wrap a saved output file in the artifact type this node outputs."""
        raise NotImplementedError

    async def aprocess(self) -> None:
        self._clear_execution_status()
        output_param = self.output_parameter_name

        try:
            # `build_payload` reads media off disk, base64-encodes it, and for video-to-video
            # shells out to ffmpeg. `aprocess` runs on the engine's shared event loop, so doing
            # that inline would stall every other node, progress event, and cancel request for
            # the duration. The previous `process()`/`yield` form ran on a worker thread and
            # did not have this constraint.
            payload = await asyncio.to_thread(self.build_payload)

            async with RunwayClient(node_name=self.name) as client:
                task_id = await client.submit(self.endpoint, payload)
                self.publish_update_to_parameter("task_id_output", task_id)
                self.append_value_to_parameter("result_details", f"Task {task_id} submitted.\n")

                urls = await client.await_output(
                    task_id,
                    on_status=lambda status: self.append_value_to_parameter("result_details", f"{status}\n"),
                )

            if len(urls) > 1:
                logger.warning(
                    "%s: RunwayML returned %d outputs for task %s; saving the first and ignoring the rest",
                    self.name,
                    len(urls),
                    task_id,
                )
            artifact = await self._save_output(urls[0])
            self.parameter_output_values[output_param] = artifact
            self.publish_update_to_parameter(output_param, artifact)
            self._set_status_results(was_successful=True, result_details=f"Saved output to {artifact.value}")

        except (RunwayError, ValueError, OSError, FileLoadError, FileWriteError) as e:
            # The output parameter is left untouched on failure: a media-typed parameter must
            # never carry an error object. Failures travel down the Failed control output.
            self._set_status_results(was_successful=False, result_details=str(e))
            self._handle_failure_exception(e)

    async def _save_output(self, url: str) -> UrlArtifact:
        """Download a generated file into project storage.

        Runway's own output URLs expire, so the file is copied locally rather than handed
        downstream as a link that will later 404.

        Raises:
            FileLoadError: If the output cannot be downloaded.
            FileWriteError: If the output cannot be written.
        """
        logger.info("%s: downloading output from %s", self.name, url)
        payload = await File(url).aread_bytes()

        # The artifact has to carry the location `awrite_bytes` returns, not the destination's.
        # Before the write, the destination holds the situation template verbatim -- complete
        # with `{file_name_base}` and `{###}` -- which nothing downstream can resolve. The
        # returned File is the portable, collision-resolved path, and it also reflects any
        # extension the engine coerced to match the bytes actually written.
        saved = await self._output_file.build_file().awrite_bytes(payload)
        logger.info("%s: saved output to %s", self.name, saved.location)

        return self.build_artifact(saved.location)
