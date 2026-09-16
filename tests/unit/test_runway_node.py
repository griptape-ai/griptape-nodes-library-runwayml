"""Tests for the shared RunwayML node base class.

Covers the two behaviours most likely to regress silently: seed bookkeeping, which used
to be shared across every node of a type, and failure routing, which used to hand
downstream nodes an ErrorArtifact through a media-typed parameter.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from griptape.artifacts import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.files.file import FileWriteError
from griptape_nodes.retained_mode.events.os_events import FileIOFailureReason
from runway_client import RunwayTaskFailedError
from runway_node import DEFAULT_SEED, RunwayTaskNode


class StubRunwayNode(RunwayTaskNode):
    """Minimal concrete node so the base class can be exercised on its own."""

    endpoint = "image_to_video"
    output_parameter_name = "video_output"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._add_seed_parameters()
        self.add_parameter(
            Parameter(
                name="video_output",
                output_type="VideoUrlArtifact",
                type="VideoUrlArtifact",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="out",
            )
        )
        self._add_output_file_parameter()
        self._create_status_parameters()

    def build_payload(self) -> dict[str, Any]:
        return {"model": "gen4_turbo"}

    def build_artifact(self, location: str) -> VideoUrlArtifact:
        return VideoUrlArtifact(value=location)


@pytest.fixture
def node() -> StubRunwayNode:
    with patch("runway_node.get_api_key", return_value="k"):
        return StubRunwayNode(name="stub")


class TestSeedResolution:
    def test_fixed_uses_the_requested_seed(self, node: StubRunwayNode) -> None:
        node.set_parameter_value("seed", 999)
        node.set_parameter_value("seed_control", "fixed")
        assert node._resolve_seed() == 999

    def test_increment_walks_up_across_runs(self, node: StubRunwayNode) -> None:
        node.set_parameter_value("seed", DEFAULT_SEED)
        node.set_parameter_value("seed_control", "increment")
        assert node._resolve_seed() == DEFAULT_SEED + 1
        assert node._resolve_seed() == DEFAULT_SEED + 2

    def test_decrement_walks_down_across_runs(self, node: StubRunwayNode) -> None:
        node.set_parameter_value("seed_control", "decrement")
        assert node._resolve_seed() == DEFAULT_SEED - 1
        assert node._resolve_seed() == DEFAULT_SEED - 2

    def test_randomize_stays_in_runways_accepted_range(self, node: StubRunwayNode) -> None:
        node.set_parameter_value("seed_control", "randomize")
        for _ in range(50):
            assert 0 <= node._resolve_seed() <= 4294967295

    def test_unknown_mode_raises_rather_than_guessing(self, node: StubRunwayNode) -> None:
        node.set_parameter_value("seed_control", "fixed")
        node.parameter_values["seed_control"] = "nonsense"
        with pytest.raises(ValueError, match="Unknown seed control mode"):
            node._resolve_seed()

    def test_seed_state_is_per_instance(self) -> None:
        """Two nodes of the same type must not share seed state."""
        with patch("runway_node.get_api_key", return_value="k"):
            first = StubRunwayNode(name="a")
            second = StubRunwayNode(name="b")

        first.set_parameter_value("seed_control", "increment")
        second.set_parameter_value("seed_control", "increment")

        first._resolve_seed()
        first._resolve_seed()

        assert second._resolve_seed() == DEFAULT_SEED + 1


class TestValidation:
    def test_missing_api_key_fails_the_workflow_before_it_runs(self, node: StubRunwayNode) -> None:
        with patch("runway_node.get_api_key", side_effect=RunwayTaskFailedError("no key")):
            errors = node.validate_before_workflow_run()
        assert errors is not None
        assert any("no key" in str(e) for e in errors)

    def test_configured_api_key_passes(self, node: StubRunwayNode) -> None:
        with patch("runway_node.get_api_key", return_value="k"):
            assert node.validate_before_workflow_run() is None


class TestSeedIsPublished:
    """The `seed` parameter is an OUTPUT and the default control is `randomize`."""

    def test_resolved_seed_is_written_back_and_published(self, node: StubRunwayNode) -> None:
        node.set_parameter_value("seed_control", "randomize")
        used = node._resolve_seed()

        assert node.parameter_output_values["seed"] == used
        assert node.get_parameter_value("seed") == used

    def test_increment_chains_from_the_seed_actually_used(self, node: StubRunwayNode) -> None:
        """A saved workflow's seed must be the starting point, not the class default."""
        node.set_parameter_value("seed", 5000)
        node.set_parameter_value("seed_control", "increment")

        assert node._resolve_seed() == 5001
        assert node._resolve_seed() == 5002


class TestSaveOutput:
    @pytest.mark.asyncio
    async def test_artifact_carries_the_written_location_not_the_template(self, node: StubRunwayNode) -> None:
        """Before the write, the destination holds an unresolved macro template.

        Using it would hand downstream `{outputs}/..._v{###}.{file_extension}`, which nothing
        can resolve, while the node reported success.
        """
        written = SimpleNamespace(location="{outputs}/output_v001.mp4")
        destination = SimpleNamespace(
            location="{outputs}/{file_name_base}_v{###}.{file_extension}",
            awrite_bytes=AsyncMock(return_value=written),
        )

        with (
            patch("runway_node.File") as mock_file,
            patch.object(node._output_file, "build_file", return_value=destination),
        ):
            mock_file.return_value.aread_bytes = AsyncMock(return_value=b"bytes")
            artifact = await node._save_output("https://runway.example/out.mp4")

        assert artifact.value == "{outputs}/output_v001.mp4"
        destination.awrite_bytes.assert_awaited_once_with(b"bytes")

    @pytest.mark.asyncio
    async def test_a_write_failure_routes_to_the_failed_branch(self, node: StubRunwayNode) -> None:
        """FileWriteError is not a FileLoadError, so it needs catching in its own right.

        Runway has already been paid at this point; the user must get their Failed path, not a
        bare traceback.
        """
        with (
            patch("runway_node.RunwayClient") as mock_client,
            patch.object(
                node,
                "_save_output",
                side_effect=FileWriteError(FileIOFailureReason.DISK_FULL, "no space left on device"),
            ),
            patch.object(node, "_has_outgoing_connections", return_value=True),
        ):
            runway = mock_client.return_value.__aenter__.return_value
            runway.submit.return_value = "task-1"
            runway.await_output.return_value = ["https://runway.example/out.mp4"]
            await node.aprocess()

        assert node.get_parameter_value("was_successful") is False
        assert "no space left on device" in str(node.get_parameter_value("result_details"))
        assert node.get_next_control_output() is node.failure_output


class TestOutputFileContract:
    def test_a_node_that_never_added_its_output_file_is_caught_before_running(self) -> None:
        """Otherwise this surfaces as an AttributeError after the generation is billed."""

        class MissingOutputFile(StubRunwayNode):
            def __init__(self, **kwargs) -> None:
                RunwayTaskNode.__init__(self, **kwargs)
                self._create_status_parameters()

        with patch("runway_node.get_api_key", return_value="k"):
            node = MissingOutputFile(name="broken")
            errors = node.validate_before_workflow_run()

        assert errors is not None
        assert any("output file parameter" in str(e) for e in errors)


class TestFailureRouting:
    @pytest.mark.asyncio
    async def test_failure_leaves_the_media_output_unset(self, node: StubRunwayNode) -> None:
        """A failure must not populate a VideoUrlArtifact parameter with an error object."""
        with (
            patch("runway_node.RunwayClient") as mock_client,
            patch.object(node, "_handle_failure_exception") as mock_handle,
        ):
            mock_client.return_value.__aenter__.return_value.submit.side_effect = RunwayTaskFailedError("boom")
            await node.aprocess()

        assert "video_output" not in node.parameter_output_values
        assert node.get_parameter_value("was_successful") is False
        assert "boom" in str(node.get_parameter_value("result_details"))
        mock_handle.assert_called_once()

    @pytest.mark.asyncio
    async def test_failure_routes_to_the_failed_control_output(self, node: StubRunwayNode) -> None:
        with (
            patch("runway_node.RunwayClient") as mock_client,
            patch.object(node, "_has_outgoing_connections", return_value=True),
        ):
            mock_client.return_value.__aenter__.return_value.submit.side_effect = RunwayTaskFailedError("boom")
            await node.aprocess()

        assert node.get_next_control_output() is node.failure_output

    @pytest.mark.asyncio
    async def test_success_sets_the_output_and_success_branch(self, node: StubRunwayNode) -> None:
        with (
            patch("runway_node.RunwayClient") as mock_client,
            patch.object(node, "_save_output", return_value=VideoUrlArtifact(value="/tmp/out.mp4")),
        ):
            runway = mock_client.return_value.__aenter__.return_value
            runway.submit.return_value = "task-1"
            runway.await_output.return_value = ["https://runway.example/out.mp4"]
            await node.aprocess()

        assert node.parameter_output_values["video_output"].value == "/tmp/out.mp4"
        assert node.get_parameter_value("was_successful") is True
        assert node.get_next_control_output() is node.control_parameter_out
