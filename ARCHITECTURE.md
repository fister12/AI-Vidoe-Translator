# Architecture

## Pipeline Overview

```
Input Video
    |
    v
[1] Audio Extraction ----------> [2] Audio Preprocessing (optional)
    |                                    |
    |                                    v
    |                            [3] Transcription (Whisper)
    |                                    |
    |                                    v
    |                            [3b] Segment Optimization (optional)
    |                                    |
    |                                    v
    |                            [4] Translation
    |                                    |
    v                                    v
[5] Voice Sample & -------> [5b] Speaker Diarization (optional)
    Speaker references                   |
    |                                    v
    +--------------------------> [5c] TTS Synthesis (XTTS / gTTS)
                                         |
                                         v
                                 [6] Wav2Lip Lip-sync (with Narrator exclusion)
                                         |
                                         v
                                 [7] GFPGAN Enhancement (optional)
                                         |
                                         v
                                 [8] Post-Processing (basic or optical flow)
                                         |
                                         v
                                 [9] Color Matching (optional)
                                         |
                                         v
                                 [10] Audio + Video Mux
                                         |
                                         v
                                   Output Video
```

The pipeline supports checkpoint/resume -- completed steps are saved to `temp/.pipeline_state.json` and skipped on re-run.


---

## Module Reference

### `main.py` -- Pipeline Orchestrator
- CLI argument parsing (30+ arguments)
- Pipeline state management (checkpoint/resume)
- Language resolution with backward-compatible aliases
- XTTS healthcheck utility

### `src/media_utils.py` -- Media Utilities
- FFmpeg availability check and CUDA detection
- Audio extraction from video (`extract_audio_from_video()`)
- Voice sample extraction for speaker cloning (`extract_voice_sample()`)
- Audio segment time-slice extraction (`extract_audio_segment()`)
- Audio stretching/padding (`stretch_audio_to_video_duration()`, `pad_or_trim_audio_to_video_duration()`)
- FFmpeg-based basic post-processing (`postprocess_video_quality()`)
- Video+audio muxing (`mux_video_with_audio()`)

### `src/diarization.py` -- Speaker Diarization
- Performs speaker clustering using segment conditioning latents from the XTTS model (`diarize_and_extract_speakers()`)
- Assigns unique speaker IDs and groups voice samples to clone multi-speaker conversations

### `src/transcription.py` -- Whisper Wrapper
- Loads Whisper model and transcribes audio (`transcribe_english_audio()`)
- Returns text, segments (with timestamps), and detected language
- Supports all Whisper model sizes (tiny through large)

### `src/translation.py` -- Translation
- Dual-backend: `googletrans` (primary) with `deep-translator` fallback
- `translate_text()` for single strings
- `translate_segments()` preserves timing fields from Whisper segments

### `src/tts.py` -- Text-to-Speech
- Custom exceptions: `XTTSUnavailableError`, `XTTSRuntimeSynthesisError`, `GTTSError`
- Model caching via `_CachedXTTSModel` -- loads XTTS once, reuses across segments
- Policy-driven TTS: `strict_clone`, `fallback_allowed`, `fallback_only`
- Segment-aligned synthesis with time-stretching and original-timestamp placement (`synthesize_aligned_audio_from_segments()`)
- Peak normalization to prevent clipping

### `src/syncing.py` -- Wav2Lip Wrapper
- Launches Wav2Lip as a subprocess (`run_wav2lip_inference()`)
- Parses stdout for percentage progress with tqdm display
- Validates input files before launching
- Integrates exclusion intervals (`exclude_intervals.json`) to skip off-screen narrators

### `src/enhancement.py` -- Face Enhancement
- Torchvision compatibility shim for `functional_tensor` module rename
- GFPGAN initialization with auto-download of model weights
- Frame-by-frame face enhancement with fallback to original (`enhance_faces_in_video()`)

### `src/languages.py` -- Language Support
- 17 supported target languages
- Extensive alias dictionary (handles "french", "fr", "francais", etc.)
- `normalize_target_language()` with validation

### `src/preprocessing.py` -- Audio Preprocessing
- LUFS-based loudness normalization
- Spectral subtraction denoising
- Butterworth bandpass filtering (80Hz-15kHz for speech)
- Dynamic range compression with envelope follower
- `preprocess_audio_for_transcription()` orchestrates all steps

### `src/postprocessing.py` -- Advanced Post-Processing
- FFmpeg-based optical flow smoothing (`advanced_postprocess_with_optical_flow()`)
- Histogram-based color matching in LAB color space (`apply_histogram_matching()`)
- Temporal jitter removal via median filtering (`remove_temporal_jitter()`)
- CLAHE adaptive contrast enhancement (`enhance_contrast_adaptive()`)
- Lip-sync accuracy verification via mouth motion (`verify_lip_sync_accuracy()`)

### `src/quality_analysis.py` -- Quality Analysis & Segment Optimization
- Segment optimization: confidence filtering, long-segment splitting, short-segment merging (`optimize_transcription_segments()`)
- Pronunciation difficulty analysis (`analyze_segment_pronunciation_difficulty()`)
- Optimal TTS parameter estimation (`estimate_optimal_tts_parameters()`)
- Comprehensive video quality assessment (`VideoQualityScorer.analyze_video()`)
- Sync quality estimation from segment characteristics (`estimate_sync_quality_from_segments()`)


---

## Integration Guide

### Adding Audio Preprocessing

```python
from src.preprocessing import preprocess_audio_for_transcription

# After extract_audio_from_video():
preprocessed = working_dir / "audio_preprocessed.wav"
preprocess_audio_for_transcription(
    extracted_audio_path,
    preprocessed,
    normalize=True,
    denoise=True,
    bandpass=True,
    compress=True,
)
# Use `preprocessed` for transcription instead of raw audio
```

### Adding Segment Optimization

```python
from src.quality_analysis import optimize_transcription_segments

if args.optimize_segments and transcript_segments:
    transcript_segments = optimize_transcription_segments(
        transcript_segments,
        min_confidence=0.80,
        max_segment_duration=5.0,
    )
```

### Adding Advanced Post-Processing

```python
from src.postprocessing import advanced_postprocess_with_optical_flow, apply_histogram_matching

# After Wav2Lip:
advanced_postprocess_with_optical_flow(
    video_for_postprocess,
    advanced_output,
    denoise_strength=1.5,
    sharpen_amount=0.8,
    enable_optical_flow_smoothing=True,
)

# Color matching:
apply_histogram_matching(
    advanced_output,
    reference_video=input_video,
    output_path=color_matched,
    preserve_luminance=True,
)
```

### Adding Quality Analysis

```python
from src.quality_analysis import VideoQualityScorer, estimate_sync_quality_from_segments

quality = VideoQualityScorer.analyze_video("video.mp4", sample_frame_count=50)
print(f"Sharpness: {quality['sharpness']['mean']:.1f}/100")

sync = estimate_sync_quality_from_segments(segments, target_language="es")
print(f"Sync difficulty: {sync['sync_difficulty']}")
```

### Adding Speaker Diarization & Narrator Isolation

```python
from src.diarization import diarize_and_extract_speakers

# Perform speaker diarization on translated segments:
num_speakers = diarize_and_extract_speakers(
    segments=translated_segments,
    working_dir=working_dir,
    tts_model_name=args.tts_model_name,
    device=device,
    num_speakers=args.num_speakers,
)

# In run_wav2lip_inference(), pass exclude_intervals_path to skip lip-syncing for narrators:
from src.syncing import run_wav2lip_inference
run_wav2lip_inference(
    checkpoint_path=args.checkpoint_path,
    face_video_path=input_video,
    audio_path=translated_audio_synced_path,
    output_video_path=wav2lip_output_path,
    exclude_intervals_path=exclude_intervals_path,
)
```


---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Segment-aligned TTS over global stretch | Preserves natural per-utterance timing; avoids rushed or dragged speech |
| XTTS model caching | Avoids reloading the ~1.8 GB model per segment (30-50% faster on repeat runs) |
| TTS backend policy system | Graceful degradation for environments where XTTS C++ deps can't compile |
| Pipeline checkpoint/resume | Handles interruptions and allows incremental re-runs |
| GFPGAN as optional pass | Adds processing time; only needed when Wav2Lip blur is noticeable |
| Dual translation backends | `googletrans` is rate-limited; `deep-translator` provides reliability fallback |

---

## Performance Impact

| Feature | Speed Cost | Quality Gain |
|---|---|---|
| Audio preprocessing | -5-10% | +5-15% transcription accuracy |
| Segment optimization | +2-5% (faster) | +10-15% lip-sync |
| Optical flow smoothing | -30-40% | +20-30% visual smoothness |
| Color matching | -10-15% | +15% visual consistency |
| GFPGAN enhancement | -20-30% | +15-20% face clarity |

---

## Dependencies

**Core:** Python 3.11+, FFmpeg, PyTorch + CUDA (recommended)

**AI Models:** OpenAI Whisper, Coqui XTTS v2, Wav2Lip, GFPGAN

**Audio:** librosa, soundfile, scipy, torchaudio

**Video:** opencv-python, moviepy

**Translation:** googletrans 4.0.0-rc1, deep-translator

**Optional:** textstat, epitran (pronunciation analysis), face-recognition (sync verification)
