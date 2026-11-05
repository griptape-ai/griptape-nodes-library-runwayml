- **Breaking:** The RunwayML nodes no longer check the API key and inputs when a flow starts. A
  missing `RUNWAYML_API_SECRET` or an empty input now stops the node when it runs, with the error
  on the node. A flow with an empty input now starts and fails at that node. A wired Failed output
  is not followed for a missing key. Nothing needs changing in a saved workflow unless it relied on
  the flow refusing to start.
  [#38](https://github.com/griptape-ai/griptape-nodes-library-runwayml/issues/38)
