# Test plan

Per-node coverage for the RunwayML library, in a real engine with a real key.

The offline suite (257 tests) constructs nodes by hand and drives `aprocess` against mocks,
and `make api/drift` only compares constants to the published spec. Neither touches the
engine. So everything below is about the half that has never run: the engine loading the
library, dispatching a node, and turning a finished generation into a usable artifact.

Work top to bottom. Prerequisites first, then shared behaviour once, then each node.

---

## 0. Prerequisites

### 0.1 The library loads

Register the library (`runwayml/griptape_nodes_library.json`) and restart the engine.

- [ ] All 8 nodes appear, under `Video/RunwayML` and `Image/RunwayML`
- [ ] No load error in the engine log

Watching for three things this change introduced:

- The four new sibling modules (`api_surface`, `artifacts`, `runway_client`, `runway_node`)
  resolve. Nodes import them as top-level modules because the loader puts the manifest's
  directory on `sys.path`; the offline suite reproduces that by hand, so this is the first
  real test of it.
- `httpx` and `pillow` install from `pip_dependencies`. Both are new and have only ever been
  satisfied by the dev environment.
- `engine_version` is `0.101.0`, raised for `SuccessFailureNode` / `aprocess` /
  `exe_types.param_types`. If the engine rejects the library on version, that number is wrong.

### 0.2 Key is configured

- [ ] `uv run python scripts/live_smoke.py` passes (auth, API version, tier model availability)
- [ ] All 8 models report as enabled for the account

---

## 1. Shared behaviour — test once, on any node

These live in `RunwayTaskNode` / `RunwayClient`, so they do not need repeating per node. Use
Text-to-Image; it is the fastest to iterate on.

### 1.1 Validation fires in the editor

- [ ] Key unset → the run fails before anything executes, naming the missing key
- [ ] Required input missing → error in the editor, not at run time
- [ ] A retired model driven in **over a connection** (string node → `model`, so the `Options`
      trait cannot snap it) → editor error naming the model

### 1.2 Failure routes to Failed

- [ ] Connect **Failed** to something visible; force a failure
- [ ] **Failed** fires, **Succeeded** does not
- [ ] The media output stays **empty** — not populated with an error object
- [ ] `result_details` carries RunwayML's own field-level message

### 1.3 Success path and the output artifact

- [ ] **Succeeded** fires
- [ ] The media output previews in the GUI. **This is the important one:** `_save_output` used
      to hand downstream the unresolved macro template
      (`{outputs}/..._v{###}.{file_extension}`) while reporting success. A preview that renders
      is the proof it now carries the written location.
- [ ] The file exists on disk and opens
- [ ] `task_id_output` is populated and matches the job in RunwayML's dashboard

### 1.4 Seed

- [ ] `seed_control: randomize` → the `seed` output reports the seed **used**, not the
      pre-run property value
- [ ] `seed_control: fixed`, same seed, same settings → same result twice
- [ ] `seed_control: increment` → next run sends the previous seed + 1
- [ ] Two of the same node in one graph do not share seed state
- [ ] `seed: 0` is respected as 0, not treated as unset

### 1.5 Cancellation

- [ ] Start a long generation, hit Stop mid-poll
- [ ] The task is cancelled in RunwayML's dashboard rather than left running

### 1.6 Progress and concurrency

- [ ] `result_details` streams status while polling
- [ ] Two RunwayML nodes running in parallel both progress; neither blocks the other. (Media
      reads, base64 and ffmpeg were moved off the event loop; this is what confirms it.)

### 1.7 Oversized inline media

- [ ] Feed a video comfortably over 5MB → refused **before** the upload, naming the size
      limit, rather than after a long transfer

---

## 2. Runway Text-to-Image

| Run | Model | Set | Verify |
|---|---|---|---|
| 2.1 | `gen4_image` | prompt only | image returned and previews |
| 2.2 | `gen4_image` | 3 reference images, prompt using every `@tag` | each reference visibly influences the result |
| 2.3 | `gen4_image_turbo` | prompt, **no** references | editor error: this model requires one |
| 2.4 | `gen4_image_turbo` | prompt + 1 reference | succeeds |
| 2.5 | `gen4_image` | 4 reference images | RunwayML rejects it, naming `referenceImages` (the cap is deliberately not enforced locally) |
| 2.6 | `gen4_image` | each of the 16 `ratio` values, spot-checked | output dimensions match the ratio |

- [ ] Switching model does not silently reset `ratio`
- [ ] `output_file` lands as `.png`

---

## 3. Runway Create Reference Image

No API call; it builds the artifact Text-to-Image consumes.

- [ ] Each declared input shape works: `ImageUrlArtifact`, a bare path via the file browser,
      a `{inputs}/...` macro path, and a plain URL string
- [ ] Empty `tag` → a generated tag, stable across an **engine restart** (it is a digest, not
      `hash()`, precisely so a prompt written against it keeps matching)
- [ ] A user tag round-trips into `referenceImages[].tag` unchanged, including CamelCase —
      RunwayML matches tags case-insensitively, so `EiffelTower` must not be rejected
- [ ] A panorama (aspect ratio outside 0.5–2.0) is rejected **here**, naming this node rather
      than the downstream generation node
- [ ] An unmeasurable image passes through rather than being dropped

---

## 4. Runway Text-to-Video

| Run | Set | Verify |
|---|---|---|
| 4.1 | prompt, defaults | 5s 1280:720 video returned and previews |
| 4.2 | `ratio: 720:1280` | portrait output |
| 4.3 | `duration: 2` and `duration: 10` | both accepted; length matches |
| 4.4 | empty prompt | editor error |

- [ ] `duration` outside 2–10 is caught by the slider bounds

---

## 5. Runway Image-to-Video

| Run | Model | Set | Verify |
|---|---|---|---|
| 5.1 | `gen4_turbo` | image + prompt | video starts from the supplied frame |
| 5.2 | `gen4.5` | image + prompt | succeeds; **Professional Output** group becomes visible |
| 5.3 | `gen4_turbo` | — | **Professional Output** group is hidden |
| 5.4 | `gen4.5` → `gen4_turbo` | `output_format: prores` set first | on switching, `output_format` resets to `mp4` and `output_file` back to `.mp4`, so the request is not silently downgraded |
| 5.5 | either | `ratio: 720:1280`, then switch model | the chosen ratio survives the switch |

- [ ] Each input shape resolves: artifact, file browser, macro path, public URL, data URI

---

## 6. Runway Video-to-Video

The node that was fully broken before this change, and the one with the most new surface.

| Run | Set | Verify |
|---|---|---|
| 6.1 | video + prompt | edit returned and previews |
| 6.2 | video, **no** prompt, `target_aspect_ratio: 16:9` | succeeds — `promptText` is optional for `aleph2`, and the edit is driven by the ratio alone |
| 6.3 | video + reference image | succeeds. Sends a `keyframes` entry at `seconds: 0`, which replaced `gen4_aleph`'s `references` field; the shape is verified live but has never produced an actual edit |
| 6.4 | video + a reference image that no longer exists on disk | fails naming the **reference image**, rather than silently dropping it and billing an edit that ignored it |
| 6.5 | a 1-second clip | rejected: `aleph2` requires 2–30s |
| 6.6 | a 31-second clip | rejected on duration |
| 6.7 | an `.mp4` local file | uploaded byte-for-byte, **not** recompressed (check the engine log for the "sending it unchanged" line) |
| 6.8 | a container RunwayML rejects, e.g. `.avi` | normalized through ffmpeg first |
| 6.9 | an extensionless file holding video bytes | normalized, not assumed to be mp4 |

- [ ] `prompt` and `reference_image` both expose **OUTPUT**, so they can be passed downstream
      for provenance

---

## 7. Runway Video Upscale

Every parameter here is new; none has ever been sent.

| Run | Set | Verify |
|---|---|---|
| 7.1 | video, defaults | upscaled video returned; dimensions larger than the source |
| 7.2 | `resolution` at each of `720p`, `1k`, `2k`, `4k` | output dimensions match |
| 7.3 | `flavor: vivid` vs `natural` | visibly different grade |
| 7.4 | `creativity: 0` vs `100` | 0 stays faithful; 100 invents detail |
| 7.5 | `sharpen` and `smart_grain` at 0 and 100 | visible difference |
| 7.6 | `fps_boost: true` | output frame rate higher than the source |
| 7.7 | a 31-second clip | rejected on duration |

- [ ] `model` set to `upscale_v1` over a connection → editor error naming the replacement

---

## 8. Runway Video-to-HDR

| Run | Set | Verify |
|---|---|---|
| 8.1 | SDR video, `hdr10` | HDR10 output; `ffprobe` shows BT.2020 + PQ, 10-bit |
| 8.2 | `hlg` | HLG transfer characteristic |
| 8.3 | `hdr_prores` + `422 HQ` | a real ProRes stream in a `.mov` — check with `ffprobe`, not the extension |
| 8.4 | `hdr_prores` + a tier RunwayML does not serve for it | rejected by RunwayML naming the pairing (not gated locally) |
| 8.5 | a video already HDR-tagged | rejected: the input must be genuine SDR |
| 8.6 | a source over 4096px per side | rejected |

- [ ] `prores_profile` is hidden unless `output_format` is `hdr_prores`
- [ ] `output_file` extension follows the container: `.mp4` for hdr10/hlg, `.mov` for hdr_prores

---

## 9. Runway Act Two

| Run | Set | Verify |
|---|---|---|
| 9.1 | `character_type: image`, character image + reference video | performance video returned |
| 9.2 | `character_type: video`, character video + reference video | succeeds |
| 9.3 | `character_type: image` | only `character_image` is visible; `character_video` is hidden, and vice versa |
| 9.4 | `body_control` on vs off | body motion transfers only when on |
| 9.5 | `expression_intensity` at 1 and 5 | visibly different |
| 9.6 | each `ratio` value, spot-checked | output dimensions match |
| 9.7 | reference video missing | editor error naming the reference video |

- [ ] `character_image` exposes **OUTPUT**

---

## 10. Professional output formats

Run after §2–9 are clean, since this is the same path with a different container.

| Run | Node | Set | Verify |
|---|---|---|---|
| 10.1 | Text-to-Video | `prores` + `4444` | `ffprobe` confirms ProRes 4444 in a `.mov` |
| 10.2 | Text-to-Video | `hdr10` | 10-bit HEVC, BT.2020 + PQ |
| 10.3 | Text-to-Video | `sdr_rec709_10bit` | 10-bit Rec.709 |
| 10.4 | Image-to-Video on `gen4.5` | `hdr_prores` | ProRes in a `.mov` |
| 10.5 | Video-to-Video | `prores` | ProRes, and the **source was not recompressed** first (§6.7) |
| 10.6 | any | `prores` with `prores_profile` cleared | the key is omitted and RunwayML picks a tier, rather than being sent empty and rejected |

**The thing most worth knowing from this section:** the engine's `WRITE_VIDEO_CODEC`
authorization checkpoint ffprobes written bytes and **fails closed** on a codec its policy
disallows. With no policy installed it permits everything, but an app build that denies
`prores` writes would fail every paid ProRes generation *at save time*, after the credits are
spent. Nothing short of a real save against a real app build will tell us.

---

## Not covered, deliberately

- **Frame-sequence formats** (`png_sequence`, `hdr_exr_sequence`, ACEScg EXR) — not offered by
  any node and rejected in payload validation, so there is nothing to exercise until they are
  unpacked into a `Sequence` for the OpenEXR and OpenColorIO nodes.
- **`/v1/uploads`** — media is still inlined as base64, hence §1.7 rather than an upload test.
- **Full `aleph2` keyframes** (up to 5, timed) — §6.3 covers the single-image case the node
  actually sends.
- **Third-party models RunwayML proxies** — they belong to their vendors.

## One property that changes once the account is funded

Payload acceptance was verified at zero cost because RunwayML validates the request body
*before* checking the credit balance, so an out-of-credit rejection proved the payload was
well-formed. That property disappears on a funded account: a valid payload now starts a real
job. `scripts/live_smoke.py --probe` submits and cancels immediately, which used to be free
and no longer is.
