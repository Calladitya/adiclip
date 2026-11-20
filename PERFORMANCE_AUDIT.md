# Performance audit

## Current implementation

Snapshot reviewed on 2026-11-20. There is no Git repository in this workspace, so code history and yesterday's exact diff are unavailable. The current code includes changes made through approximately 02:53 on Nov 20.

### Existing performance and resource controls

- Whisper runs on CPU with `int8`, a cached model per process, up to 8 CPU threads, and beam size selected by performance mode (1 fast, 2 balanced, 5 high quality).
- Extracted audio is cached by source file identity and audio format. Transcript results are cached by source identity and transcription settings. Clip selection is cached by transcript and analysis settings.
- Face/reframe plans are cached by source identity, clip interval, sampling settings, and speech timing signature. Face inference is downscaled; BALANCED mode samples every 4 seconds at up to 320 pixels wide.
- Encoder selection probes supported hardware encoders and prefers AMD AMF on this AMD Windows machine when the probe succeeds. It falls back to NVENC/QSV if available, then software x264; a failed hardware render retries with x264.
- The app uses one background pipeline task per Streamlit server and supports cancellation. Clip rendering uses a bounded worker pool; BALANCED and FAST modes can select two workers when at least 8 logical CPUs and 4 GB available RAM are detected. Other configurations use one.
- OpenCV is limited to two threads. Performance modes trade sampling density, transcription effort, and encode quality/speed.

### Current implementation and known limits

- Active speaker selection is a visual heuristic: face tracking plus mouth-region pixel motion, scored near Whisper speech intervals. Whisper supplies speech timing, not speaker identity. This is not diarization and has not been validated against podcast interviews.
- Smart Dynamic can choose a single crop or a stacked two-face split. The split uses fixed horizontal crop centers from detected tracks; it does not independently follow both speakers over time. Layout choice and speaker attribution remain heuristic.
- Reframe plans are computed once per clip before rendering and then passed to render jobs. With the default settings the code can render two clips at once on a sufficiently provisioned machine.
- Caption events use Whisper word timestamps and are burned via FFmpeg/libass. No current visual or audio-sync regression results are available.
- The UI records stage timings and process resource snapshots, but process peak working set does not include FFmpeg child-process memory. System-wide peak RAM and CPU utilization require external sampling during a run.

## Historical measurements (not current validation)

The prior audit measured a single 30-second 4K-source segment scaled/cropped to 1080x1920, without audio or captions:

| Encoder | Time | Output size | Result |
| --- | ---: | ---: | --- |
| libx264, `faster`, CRF 22 | 102.01 s | 20.4 MB | Successful |
| h264_amf, speed, QP 22 | 72.92 s | 28.6 MB | Successful |

That one run found AMF 1.40x faster with a larger file; quality equivalence was not established. These are historical render-only figures, not an end-to-end baseline, and should not be treated as representative of the present pipeline or source. The earlier audit's statements that transcription, audio extraction, face plans, and clip selection were uncached, and that UI processing was synchronous without cancellation, are outdated as of the current code.

## Validation still required

- End-to-end timings for audio extraction, transcription, face analysis/reframing, selection, rendering, and total runtime.
- External peak RAM and CPU utilization, including FFmpeg child processes.
- Actual encoder used and any hardware-to-software fallback.
- Output review for 9:16 framing, active speaker visibility, transition smoothness, split-screen correctness, caption timing, frame continuity, audio sync, and visual quality.
- Multi-speaker podcast validation. The selected source is a single on-camera gaming commentary video, so it cannot establish interview speaker attribution or split-screen quality.

## Controlled run plan

Use the existing 22.5-minute `HOW_TO_PLAY_GTA_5_ON_MOBILE_3VdvM0GtqRM.mp4` source, BALANCED mode, one selected clip, and exactly one render worker. Save generated media under a dedicated validation output directory so existing user clips are preserved. This is a pipeline smoke/performance check on available media, not a representative podcast-quality test.

## Controlled run results

Run on 2026-11-20 using `HOW_TO_PLAY_GTA_5_ON_MOBILE_3VdvM0GtqRM.mp4` (3840x2160, 1352.49 seconds), BALANCED mode, CPU Whisper `base`/beam 2, one selected clip, Smart Dynamic layout, and one render worker. Generated assets were isolated in `output/validation_run_20261120`; logs and frame samples are in `temp/validation_run_20261120`. Existing output clips were not overwritten.

| Stage / measurement | Result |
| --- | ---: |
| Audio extraction | 8.98 s |
| Whisper model load | 4.46 s |
| Transcription | 768.99 s (12m 49s), 553 segments; language detector returned `hi` |
| Clip selection | 1.07 s |
| Face analysis / reframe plan | 24.89 s |
| Render wall time (face analysis + encoding) | 69.70 s |
| FFmpeg encode time reported by the clip renderer | 44.59 s |
| Total run, including packaging | 855.37 s (14m 15s) |
| Encoder | `h264_amf` (AMD AMF; no fallback) |
| CPU | 48.9% average of 16 logical processors, sampled across Python and FFmpeg process CPU time |
| Peak RAM evidence | Python peak working set 1452.3 MiB via Windows PSAPI. Two-second process-tree sampling observed 863.9 MiB maximum; it missed short peaks. A complete system-wide peak was not captured. |

### Output inspection

- The isolated export is 1080x1920 H.264 at 30 fps, 41.067 seconds, 1232 frames; AAC audio is 41.006 seconds. Both streams start at 0. The 60 ms stream-duration difference is under two output frames; this metadata check does not replace listening for sync.
- Sampled frames show the visible speaker centered in the vertical crop. Reframe points moved by at most 138 source pixels between 4-second samples, with interpolation between points. No abrupt crop jump was apparent in the reviewed frames.
- Word-highlight captions are present and readable in sampled Urdu-script frames. At sampled times they align with the nearby Whisper word intervals. Caption wording was not independently checked by a fluent speaker, and audio playback sync was not verified.
- No exact duplicate images appeared between 1-second samples. The whole clip was not checked frame-by-frame for duplicates or dropped frames.
- This one-face clip selected `single_speaker`; split-screen behavior was not exercised. It cannot validate multi-speaker podcast tracking.
- A large English title graphic in the source obscures much of the face in one sampled frame around 15 seconds. That graphic is baked into the source, not added by the crop/caption renderer.
- The generated clip title is blank (`clip_01__...mp4`): ASCII-only title cleanup strips Urdu-script words. This is a reproducible localization/filename issue to address separately.

### Failures and interpretation

- The first test harness launch failed before processing because its temporary script directory hid the project package from Python. The harness path was corrected; the subsequent single pipeline run completed successfully.
- The face detector emitted an OpenCV backend warning about unsupported target preference, but continued and produced the plan.
- This measures a single-speaker, Hindi/Urdu commentary video, not podcast interview footage. Split-screen and actual diarization remain untested. One clip and one source do not establish general performance or visual quality.
- The main bottleneck in this run was CPU transcription (about 90% of the total runtime). Audio extraction, face analysis, selection, and clip rendering were much shorter. RAM use was modest relative to the laptop's 16 GB, but combined system RAM peak is not precisely measured.

### Recommended next improvements

1. Improve CPU transcription throughput, with timing comparisons on representative speech and accuracy checks before changing Whisper settings.
2. Preserve Unicode in generated titles and filenames, then validate clip selection quality for non-English transcripts.
3. Add a repeatable audio/video sync and frame-continuity check; retain the present outputs as a baseline.
4. Validate active-speaker tracking and split-screen using a real two-person podcast sample before tuning their heuristics.
