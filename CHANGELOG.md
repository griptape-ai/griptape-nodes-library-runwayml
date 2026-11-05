# Changelog

All notable changes to the Griptape Nodes RunwayML Library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/), and each version is
the `library_version` in `runwayml/griptape_nodes_library.json`.
**Breaking** marks a change that can stop a saved workflow from working without edits.

## [Unreleased]

<!-- Entries go in changelog.d/, one file each. See changelog.d/README.md. -->

## [0.3.0] - 2026-11-06

### Changed

- **Breaking:** The RunwayML nodes no longer check the API key when a flow starts. A missing
  `RUNWAYML_API_SECRET` is now reported when the node runs, on the node with a link to Settings, and
  a flow with no key starts instead of refusing. A wired Failed output is not followed for a missing
  key. Nothing needs changing in a saved workflow unless it relied on the flow refusing to start.
  [#38](https://github.com/griptape-ai/griptape-nodes-library-runwayml/issues/38)
- Errors from the RunwayML nodes show Runway's own message in the editor, with the task ID,
  status code, request ID and Runway's response under More info. A missing or refused
  `RUNWAYML_API_SECRET` links to Settings and to Runway's key page. Input problems name the
  parameter as the editor labels it, and the "Attempted to ... Failed because" wording and the
  node name are gone from messages.
  [#38](https://github.com/griptape-ai/griptape-nodes-library-runwayml/issues/38)

[Unreleased]: https://github.com/griptape-ai/griptape-nodes-library-runwayml/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/griptape-ai/griptape-nodes-library-runwayml/compare/7ab323cdc5d8a2ceee642d0d96a14b4d326bfb17...v0.3.0
