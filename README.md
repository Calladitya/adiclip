# PodClip AI

PodClip AI is a Streamlit application that turns long-form podcast videos into vertical short clips. It combines speech transcription, highlight selection, speaker-aware framing, captions, and FFmpeg rendering.

## Features

- Accepts a YouTube URL or an uploaded MP4, MOV, MKV, or AVI file.
- Transcribes speech locally with faster-whisper and word timestamps.
- Selects candidate moments with a local heuristic engine or optional Google Gemini.
- Creates 9:16 H.264/AAC clips with animated captions.
- Uses OpenCV YuNet face detection, speaker activity cues, dynamic reframing, and two-speaker split layouts.
- Offers Fast, Balanced, and High Quality processing modes.
- Supports cancellation and individual or ZIP downloads.

## Technology

Python 3.12, Streamlit, faster-whisper/CTranslate2, FFmpeg, PyAV, OpenCV, YuNet, yt-dlp, and the optional Google Gen AI SDK.

## Project files

- `app.py` — Streamlit entry point and UI.
- `core/` — video ingest, transcription, selection, framing, captions, performance, and rendering.
- `models/face_detection_yunet_2023mar.onnx` — small YuNet model required for face tracking.
- `requirements.txt` — Python dependencies.
- `packages.txt` — Debian packages used by Streamlit Community Cloud.
- `Dockerfile` — Linux container configuration.
- `.env.example` — name of the optional Gemini environment variable; it contains no credential.

Personal downloads, generated clips, temporary media, local caches, and other model checkpoints are excluded from Git by `.gitignore`.

## Local development

Use Python 3.12 and install the dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The app uses the bundled `imageio-ffmpeg` binary on Windows. On Linux it prefers the system FFmpeg, which is installed by `packages.txt` or the Docker image. The YuNet model is included; faster-whisper downloads its selected speech model when first used.

The local heuristic selector works without a key. Gemini selection is optional. To set a key in PowerShell for the current terminal session:

```powershell
$env:GEMINI_API_KEY = "your-key"
streamlit run app.py
```

For Linux/macOS, use `export GEMINI_API_KEY="your-key"`. Do not commit `.env` files or put a key in source code. On a hosted platform, set `GEMINI_API_KEY` in that platform's secrets/environment settings.

## Run with Docker

```sh
docker build -t podclip-ai .
docker run --rm -p 7860:7860 \
  -e GEMINI_API_KEY \
  -v podclip-downloads:/app/downloads \
  -v podclip-output:/app/output \
  -v podclip-temp:/app/temp \
  podclip-ai
```

Then open `http://localhost:7860`. Named volumes preserve input, output, and cache data between container restarts. Keep enough disk space for source videos, rendered clips, and temporary audio. Back up or expire these files according to your retention needs.

## Deployment guidance

The Dockerfile is a starting point for Linux container hosting. For dependable processing of large videos, choose a host with persistent storage, at least 16 GB RAM, and enough CPU for software video encoding. This is a single-app Streamlit pipeline, not a multi-tenant job service.

Streamlit Community Cloud is useful for a small demo, but its default file upload limit is 200 MB and it does not publish a guaranteed CPU/RAM allocation. See [its dependency instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies) and [upload-size configuration](https://docs.streamlit.io/develop/api-reference/configuration/config.toml).

Hugging Face Spaces can run this app through its Docker SDK. Its documented CPU Basic runtime has 2 vCPU, 16 GB RAM, and 50 GB of non-persistent disk; files in the container can be lost when it restarts, and inactive free Spaces can sleep. See [Spaces hardware](https://huggingface.co/docs/hub/spaces-overview), [storage](https://huggingface.co/docs/hub/spaces-storage), and [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker).

Before exposing a deployment publicly, add access control and usage limits. The current app has no login, per-user quotas, or durable object storage. Uploaded videos and generated clips are stored on the server filesystem; this is unsuitable for confidential media on an unprotected public service. Processing also remains CPU intensive, and each instance handles a bounded number of pipeline jobs.

### Streamlit Community Cloud

1. Push this source tree to a GitHub repository, excluding local data with `.gitignore`.
2. Create an app from the repository with `app.py` as the entry point and Python 3.12.
3. Configure `GEMINI_API_KEY` in the app's Secrets settings only if using Gemini.
4. Confirm the app can install the Python dependencies and Debian packages in `requirements.txt` and `packages.txt`.

### Docker host

Build and run the supplied image, expose port `7860`, provide secrets through the host's secret manager, and mount persistent volumes for `downloads/`, `output/`, and `temp/`. Configure HTTPS and authentication in the hosting platform or reverse proxy before making the service public.

## Environment variables

| Name | Required | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | No | Enables Google Gemini moment selection. The local heuristic mode needs no key. |

## Uploads and storage

The Streamlit upload limit is configured to 200 MB. YouTube downloads, rendered clips, and intermediate/cache files can use much more disk space over time. The current app does not automatically purge user media, so production hosting needs a retention policy and persistent disk sized for the expected workload.

## Screenshots

No screenshots are included yet. Add reviewed, non-sensitive screenshots here when available.

## Tests

The repository contains `test_pipeline.py` for a local integration check. It creates a synthetic video and renders a clip, so run it only when local FFmpeg and the Python dependencies are installed and you are comfortable with generated test artifacts in the project folders.
