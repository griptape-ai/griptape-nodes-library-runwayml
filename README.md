# RunwayML Video Nodes

This library provides Griptape Nodes for interacting with the RunwayML video generation services. You can use these nodes to generate videos from images and text prompts.

**IMPORTANT:** To use these nodes, you will need an API key from RunwayML. Please visit the [RunwayML website](https://runwayml.com/) and their [API documentation](https://docs.dev.runwayml.com/guides/using-the-api/) for more information on how to obtain your key. You will need to set the `RUNWAYML_API_SECRET` environment variable, or configure it in your Griptape Nodes Engine settings for the "RunwayML" service.

To configure your key within the Griptape Nodes IDE:
1. Open the **Settings** menu.
2. Navigate to the **API Keys & Secrets** panel.
3. Add a new secret configuration for the service named `RunwayML`.
4. Enter your `RUNWAYML_API_SECRET` in the respective field.

Below is a description of the node and its parameters.

### RunwayML Image to Video (`RunwayML_ImageToVideo`)

Generates a video from a reference image and a text prompt using the RunwayML API.

![Example RunwayML Image to Video Flow](./images/example_runway.png)

| Parameter        | Type                          | Description                                                                                                                  | Default Value   |
|------------------|-------------------------------|------------------------------------------------------------------------------------------------------------------------------|-----------------|
| `image`          | `ImageArtifact` / `str`       | Input image (required). Accepts `ImageArtifact`, `ImageUrlArtifact`, a public HTTPS URL string, or a base64 data URI string. Local HTTP URLs will be converted to data URIs. |                 |
| `prompt`         | `str` / `TextArtifact`        | Text prompt describing the desired video content.                                                                            | `""`            |
| `model`          | `str`                         | RunwayML model to use for generation.                                                                                        | `gen4_turbo`    |
| `ratio`          | `str`                         | Aspect ratio for the output video. Must be one of the specific values supported by RunwayML API (e.g., "1280:720").         | `1280:720`      |
| `seed`           | `int`                         | Seed for generation. 0 for random. (Note: May not be supported by all models or the current API version for this endpoint). | `0`             |
| `motion_score`   | `int`                         | Controls the amount of motion. (Note: May not be supported by all models or the current API version for this endpoint).       | `10`            |
| `upscale`        | `bool`                        | Whether to upscale the generated video. (Note: May not be supported by all models or the current API version for this endpoint). | `False`         |
| `video_output`   | `VideoUrlArtifact`            | **Output:** URL of the generated video.                                                                                      | `None`          |
| `task_id_output` | `str`                         | **Output:** The Task ID of the generation job from RunwayML.                                                                 | `None`          |

*Note: `Inputs` and `Generation Settings` parameters are grouped and may be collapsed by default in the UI. The `seed`, `motion_score`, and `upscale` parameters are included for potential future API support or use with other models but are not currently sent to the `image_to_video` endpoint based on observed API behavior.*

## Add your library to your installed Engine! 

If you haven't already installed your Griptape Nodes engine, follow the installation steps [HERE](https://github.com/griptape-ai/griptape-nodes).
After you've completed those and you have your engine up and running: 


1. Copy the path to your `griptape-nodes-library.json` file within this `runwayml` directory. Right click on the file, and `Copy Path` (Not `Copy Relative Path`).
2. Start up the engine! 
3. Navigate to settings.
4. Open your settings and go to the App Events tab. Add an item in **Libraries to Register**.
5. Paste your copied `griptape-nodes-library.json` path from earlier into the new item.
6. Exit out of Settings. It will save automatically! 
7. Open up the **Libraries** dropdown on the left sidebar.
8. Your newly registered library should appear! Drag and drop nodes to use them! 