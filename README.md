# AI Video Translator

A local, end-to-end video translation and lip-sync pipeline. Given a video in one language, it produces a dubbed version in another language with realistic lip movements and optional voice cloning.

## Features

- **Speech-to-text** via OpenAI Whisper (auto language detection)
- **Translation** via Google Translate / Deep Translator (16 target languages)
- **Voice-cloning TTS** via Coqui XTTS v2 (or gTTS fallback)
- **Lip-sync** via Wav2Lip with optional GFPGAN face enhancement
- **Audio preprocessing** (denoise, normalize, bandpass, compress)
- **Advanced post-processing** (optical flow smoothing, color matching, temporal jitter removal)
- **Pipeline checkpoint/resume** (skip completed steps on re-run)
- **Segment-aligned timing** (preserves per-utterance pacing from the original)

## Prerequisites

- Python 3.11+ (3.11 required for Coqui XTTS on Windows; 3.12+ for gTTS-only mode)
- FFmpeg on PATH
- CUDA GPU recommended (auto-detected); CPU fallback available

## Installation

```bash
git clone <repo-url>
cd AI Video Translator
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Model checkpoints (Whisper, XTTS, Wav2Lip, GFPGAN) are auto-downloaded on first run. Place `wav2lip_gan.pth` in `models/`.

## Quick Start

```bash
python main.py \
  --input_video input.mp4 \
  --output_video output.mp4 \
  --checkpoint_path models/wav2lip_gan.pth \
  --audio_language en \
  --output_language french
```

## CLI Reference

### Core Arguments

| Argument | Default | Description |
|---|---|---|
| `--input_video` | (required) | Source video path |
| `--output_video` | (required) | Output video path |
| `--checkpoint_path` | (required) | Wav2Lip model checkpoint |
| `--wav2lip_root` | `Wav2Lip` | Path to Wav2Lip repository |
| `--whisper_model` | `small` | Whisper model size (tiny/base/small/medium/large) |

### Language

| Argument | Default | Description |
|---|---|---|
| `--audio_language` | `None` (falls back to `en`) | Source language code or `auto` for detection |
| `--output_language` | `None` (falls back to `es`) | Target language code or name |


### TTS

| Argument | Default | Description |
|---|---|---|
| `--tts_backend_policy` | `strict_clone` | `strict_clone`, `fallback_allowed`, or `fallback_only` |
| `--tts_model_name` | `tts_models/multilingual/multi-dataset/xtts_v2` | Coqui XTTS model |
| `--extract_sample_seconds` | `10.0` | Voice sample length for cloning |

### Speaker Diarization & Narrator Isolation

| Argument | Default | Description |
|---|---|---|
| `--diarize_speakers` | off | Enable speaker diarization (clustering) to clone and apply different speaker voices |
| `--num_speakers` | `None` | Number of speakers to detect/cluster (if None, auto-detected) |
| `--narrator_speaker_ids` | `None` | Comma-separated speaker IDs to treat as narrator/off-screen (no lip sync, e.g., `0` or `0,2`) |
| `--auto_detect_narrator` | off | Automatically detect off-screen narrator(s) based on face visibility |


### Timing & Device

| Argument | Default | Description |
|---|---|---|
| `--timing_mode` | `segment` | `segment` (per-utterance) or `global` (full stretch) |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |
| `--require_cuda` | off | Fail if CUDA unavailable |
| `--working_dir` | `temp` | Intermediate artifacts directory |

### Enhancement & Post-Processing

| Argument | Default | Description |
|---|---|---|
| `--enhance` | off | Apply GFPGAN face enhancement |
| `--enable_preprocessing` | off | Audio preprocessing before transcription |
| `--optimize_segments` | off | Optimize Whisper segments for TTS |
| `--enable_advanced_postprocessing` | off | Optical flow smoothing |
| `--enable_color_matching` | off | Match output colors to original |
| `--postprocess_denoise_strength` | `1.2` | Denoise strength |
| `--postprocess_sharpen_amount` | `0.6` | Sharpening amount |
| `--postprocess_contrast` | `1.02` | Contrast multiplier |
| `--postprocess_saturation` | `1.03` | Saturation multiplier |
| `--postprocess_crf` | `16` | x264 CRF (lower = higher quality) |
| `--postprocess_preset` | `slow` | x264 encoding preset |

### Wav2Lip Tuning

| Argument | Default | Description |
|---|---|---|
| `--wav2lip_pads` | `0 20 0 0` | Face padding (top bottom left right) |
| `--wav2lip_resize_factor` | `1` | Downscale factor for face detection |
| `--wav2lip_no_smooth` | off | Disable face detection smoothing |
| `--wav2lip_face_det_batch_size` | `16` | Face detection batch size |
| `--wav2lip_batch_size` | `128` | Inference batch size |
| `--wav2lip_rotate` | off | Rotate frames 90deg for phone videos |
| `--wav2lip_crop` | `0 -1 0 -1` | Optional crop region passed to Wav2Lip (top bottom left right) |
| `--wav2lip_box` | `None` | Optional fixed face box for Wav2Lip (top bottom left right) |


### Utilities

| Argument | Default | Description |
|---|---|---|
| `--list_languages` | off | Show supported target languages |
| `--xtts_healthcheck_only` | off | Check XTTS availability without running pipeline |
| `--resume` | off | Resume from last checkpoint |
| `--translated_text_path` | `None` | Optional path to save the translated segment text file |


## Supported Languages

Arabic, Chinese (Simplified), Czech, Dutch, English, French, German, Hindi, Hungarian, Italian, Japanese, Korean, Polish, Portuguese, Russian, Spanish, Turkish.


## Example Commands

```bash
# Full pipeline with all optimizations
python main.py \
  --input_video input.mp4 \
  --output_video output.mp4 \
  --checkpoint_path models/wav2lip_gan.pth \
  --audio_language en \
  --output_language hindi \
  --enhance \
  --enable_preprocessing \
  --optimize_segments \
  --enable_advanced_postprocessing \
  --enable_color_matching

# Minimal (auto-detect language)
python main.py \
  --input_video input.mp4 \
  --output_video output.mp4 \
  --checkpoint_path models/wav2lip_gan.pth \
  --audio_language auto \
  --output_language french

# Resume a failed run
python main.py \
  --input_video input.mp4 \
  --output_video output.mp4 \
  --checkpoint_path models/wav2lip_gan.pth \
  --audio_language auto \
  --output_language japanese \
  --resume

# Run with speaker diarization and auto narrator detection
python main.py \
  --input_video input.mp4 \
  --output_video output.mp4 \
  --checkpoint_path models/wav2lip_gan.pth \
  --audio_language auto \
  --output_language french \
  --diarize_speakers \
  --auto_detect_narrator

# XTTS healthcheck
python main.py --xtts_healthcheck_only --tts_backend_policy strict_clone
```

## Project Structure

```
AI Video Translator/
├── main.py                 # Pipeline orchestrator and CLI
├── requirements.txt        # Python dependencies
├── src/
│   ├── media_utils.py      # FFmpeg wrappers, audio extraction, muxing
│   ├── transcription.py    # Whisper transcription
│   ├── translation.py      # Translation backends
│   ├── tts.py              # TTS with XTTS voice cloning + gTTS fallback
│   ├── syncing.py          # Wav2Lip subprocess wrapper
│   ├── enhancement.py      # GFPGAN face enhancement
│   ├── languages.py        # Supported language definitions
│   ├── preprocessing.py    # Audio preprocessing pipeline
│   ├── postprocessing.py   # Optical flow, color matching, jitter removal
│   └── quality_analysis.py # Quality scoring and segment optimization
├── Wav2Lip/                # Vendored Wav2Lip repository
├── models/                 # Model checkpoints
├── gfpgan/weights/         # GFPGAN weights
├── input/                  # Input videos
├── output/                 # Generated outputs
└── temp/                   # Intermediate artifacts
```

## Troubleshooting

| Problem | Solution |
|---|---|
| XTTS unavailable on Windows | Use Python 3.11 or set `--tts_backend_policy fallback_allowed` |
| Audio sounds degraded | Reduce `--postprocess_denoise_strength` (try 0.8) |
| Lip-sync is off | Try `--timing_mode segment` or adjust `--wav2lip_pads` |
| Video looks jittery | Enable `--enable_advanced_postprocessing` |
| Colors differ from original | Enable `--enable_color_matching` |
| Slow processing | Disable `--enable_advanced_postprocessing` or use `--whisper_model tiny` |
