from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --- torchaudio monkeypatch for soundfile fallback (WDAC / torchcodec DLL block bypass) ---
try:
    import torchaudio
    import soundfile as sf
    import torch

    def _soundfile_load(filepath, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, **kwargs):
        filepath_str = str(filepath)
        start = frame_offset
        frames = num_frames if num_frames > 0 else -1
        data, samplerate = sf.read(filepath_str, start=start, frames=frames, dtype='float32')
        tensor = torch.from_numpy(data)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0) if channels_first else tensor.unsqueeze(1)
        else:
            if channels_first:
                tensor = tensor.transpose(0, 1)
        return tensor, samplerate

    def _soundfile_save(uri, src, sample_rate, channels_first=True, **kwargs):
        data = src.detach().cpu().numpy()
        if data.ndim == 2 and channels_first:
            data = data.T
        sf.write(str(uri), data, sample_rate)

    # Apply monkeypatch
    torchaudio.load = _soundfile_load
    torchaudio.save = _soundfile_save
    print("  [PATCH] Applied soundfile-based monkeypatch to torchaudio.load and torchaudio.save to bypass torchcodec DLL block.")
except Exception as e:
    print(f"  [WARN] Failed to apply soundfile monkeypatch to torchaudio: {e}")

from src.media_utils import (
    ensure_ffmpeg_available,
    extract_audio_from_video,
    extract_audio_segment,
    extract_voice_sample,
    mux_video_with_audio,
    pad_or_trim_audio_to_video_duration,
    postprocess_video_quality,
    resolve_runtime_device,
    stretch_audio_to_video_duration,
)
from src.enhancement import enhance_faces_in_video
from src.languages import format_supported_target_languages, normalize_target_language
from src.syncing import run_wav2lip_inference
from src.transcription import transcribe_english_audio
from src.translation import translate_segments, translate_text
from src.tts import synthesize_aligned_audio_from_segments, synthesize_speech
from src.diarization import diarize_and_extract_speakers


# ---------------------------------------------------------------------------
# Pipeline checkpoint helpers
# ---------------------------------------------------------------------------

_PIPELINE_STEPS = [
    "extract_audio",
    "preprocess_audio",
    "transcribe",
    "translate",
    "tts",
    "wav2lip",
    "enhance",
    "postprocess",
    "color_match",
    "mux",
]


def _load_pipeline_state(state_path: Path) -> dict:
    """Load the pipeline checkpoint file, or return a blank state."""
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"completed_steps": []}


def _save_pipeline_state(state_path: Path, state: dict) -> None:
    """Persist the current pipeline state to disk."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _mark_step_done(state_path: Path, state: dict, step: str) -> None:
    """Mark a step as completed and save."""
    if step not in state["completed_steps"]:
        state["completed_steps"].append(step)
    _save_pipeline_state(state_path, state)


def _step_already_done(state: dict, step: str, resume: bool) -> bool:
    """Return True if the step was already completed and resume is enabled."""
    return resume and step in state.get("completed_steps", [])


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Language-To-Language dub and sync pipeline.")

    # --- Core I/O ---
    parser.add_argument("--input_video", required="--list_languages" not in sys.argv and "--xtts_healthcheck_only" not in sys.argv,
                        help="Path to the source language video.")
    parser.add_argument("--output_video", required="--list_languages" not in sys.argv and "--xtts_healthcheck_only" not in sys.argv,
                        help="Path for the final dubbed video.")
    parser.add_argument("--checkpoint_path", required="--list_languages" not in sys.argv and "--xtts_healthcheck_only" not in sys.argv,
                        help="Path to the Wav2Lip checkpoint file.")
    parser.add_argument("--wav2lip_root", default="Wav2Lip", help="Path to the Wav2Lip repository root.")
    parser.add_argument("--whisper_model", default="small",
                        help="Whisper model size, e.g. tiny, base, small, medium, large.")

    # --- Language arguments (new primary names) ---
    parser.add_argument("--audio_language", default=None,
                        help="Language code for the input audio (e.g. 'en', 'auto' for auto-detect). "
                             "Alias for --Initial_language.")
    parser.add_argument("--output_language", default=None,
                        help="Target language code or name (e.g. 'es', 'hindi', 'french'). "
                             "Alias for --target_language.")

    # --- Legacy language aliases (hidden, for backwards compat) ---
    parser.add_argument("--Initial_language", default="en",
                        help=argparse.SUPPRESS)
    parser.add_argument("--target_language", default="es",
                        help=argparse.SUPPRESS)

    # --- TTS ---
    parser.add_argument(
        "--tts_model_name",
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="Coqui XTTS model name.",
    )
    parser.add_argument(
        "--tts_backend_policy",
        default="strict_clone",
        choices=["strict_clone", "fallback_allowed", "fallback_only"],
        help=(
            "TTS backend behavior: strict_clone requires XTTS voice cloning; "
            "fallback_allowed uses gTTS if XTTS fails; fallback_only skips XTTS."
        ),
    )

    # --- Timing ---
    parser.add_argument("--working_dir", default="temp", help="Directory for intermediate artifacts.")
    parser.add_argument(
        "--timing_mode",
        choices=["segment", "global"],
        default="segment",
        help="Audio timing strategy: segment preserves per-utterance timing, global stretches one full track.",
    )
    parser.add_argument("--extract_sample_seconds", type=float, default=10.0,
                        help="Length of voice sample for cloning.")

    # --- Device ---
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for Whisper/XTTS: auto chooses CUDA when available, else CPU.",
    )
    parser.add_argument(
        "--require_cuda",
        action="store_true",
        help="Fail fast if CUDA is not available.",
    )

    # --- Enhancement ---
    parser.add_argument(
        "--enhance",
        action="store_true",
        help="Apply GFPGAN face enhancement pass after Wav2Lip.",
    )

    # --- Post-processing parameters ---
    parser.add_argument("--postprocess_denoise_strength", type=float, default=1.2,
                        help="Postprocess denoise strength (higher removes more compression noise).")
    parser.add_argument("--postprocess_sharpen_amount", type=float, default=0.6,
                        help="Postprocess sharpening amount.")
    parser.add_argument("--postprocess_contrast", type=float, default=1.02,
                        help="Postprocess contrast multiplier.")
    parser.add_argument("--postprocess_saturation", type=float, default=1.03,
                        help="Postprocess saturation multiplier.")
    parser.add_argument("--postprocess_crf", type=int, default=16,
                        help="x264 CRF for postprocessed video (lower is higher quality).")
    parser.add_argument(
        "--postprocess_preset",
        default="slow",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
        help="x264 preset used for postprocess encoding.",
    )

    # --- Wav2Lip tuning ---
    parser.add_argument(
        "--wav2lip_pads", nargs=4, type=int,
        metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"), default=[0, 20, 0, 0],
        help="Face padding passed to Wav2Lip (top bottom left right). Increase bottom to include chin.",
    )
    parser.add_argument("--wav2lip_resize_factor", type=int, default=1,
                        help="Downscale input frames by this factor before Wav2Lip face detection.")
    parser.add_argument("--wav2lip_no_smooth", action="store_true",
                        help="Disable smoothing of face detections (helps when mouth looks duplicated/dislocated).")
    parser.add_argument("--wav2lip_face_det_batch_size", type=int, default=16,
                        help="Face detection batch size for Wav2Lip.")
    parser.add_argument("--wav2lip_batch_size", type=int, default=128,
                        help="Inference batch size for Wav2Lip model.")
    parser.add_argument(
        "--wav2lip_crop", nargs=4, type=int,
        metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"), default=[0, -1, 0, -1],
        help="Optional crop region passed to Wav2Lip (top bottom left right).",
    )
    parser.add_argument("--wav2lip_rotate", action="store_true",
                        help="Rotate frames 90 degrees clockwise before Wav2Lip (for wrongly oriented phone videos).")
    parser.add_argument(
        "--wav2lip_box", nargs=4, type=int,
        metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"), default=None,
        help="Optional fixed face box for Wav2Lip (use only if detection is unstable).",
    )

    # --- NEW: Optimization toggles ---
    parser.add_argument(
        "--enable_preprocessing",
        action="store_true",
        help="Enable audio preprocessing (denoise, bandpass, compression, normalization) before transcription.",
    )
    parser.add_argument(
        "--optimize_segments",
        action="store_true",
        help="Auto-optimize transcription segments (filter low-confidence, split long, merge short) for better TTS.",
    )
    parser.add_argument(
        "--enable_advanced_postprocessing",
        action="store_true",
        help="Enable advanced post-processing with optical flow smoothing (replaces basic postprocess).",
    )
    parser.add_argument(
        "--diarize_speakers",
        action="store_true",
        help="Enable speaker diarization (clustering) to clone and apply different speaker voices.",
    )
    parser.add_argument(
        "--num_speakers",
        type=int,
        default=None,
        help="Number of speakers to detect/cluster (if None, auto-detected).",
    )
    parser.add_argument(
        "--narrator_speaker_ids",
        default=None,
        help="Comma-separated speaker IDs to treat as narrator/off-screen (no lip sync, e.g. '0' or '0,2').",
    )
    parser.add_argument(
        "--auto_detect_narrator",
        action="store_true",
        help="Automatically detect off-screen narrator(s) based on face visibility during their segments.",
    )
    parser.add_argument(
        "--enable_color_matching",
        action="store_true",
        help="Match output video colors to the original input video.",
    )

    # --- NEW: Utility flags ---
    parser.add_argument(
        "--translated_text_path",
        default=None,
        help="If set, save the translated text to this file path.",
    )
    parser.add_argument(
        "--list_languages",
        action="store_true",
        help="Print all supported target languages and exit.",
    )
    parser.add_argument(
        "--xtts_healthcheck_only",
        action="store_true",
        help="Test the TTS stack (XTTS / gTTS) and exit without processing a video.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous run: skip pipeline steps whose output files already exist in the working directory.",
    )

    return parser


# ---------------------------------------------------------------------------
# Language resolution
# ---------------------------------------------------------------------------

def _resolve_languages(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve source and target language codes, preferring the new flag names."""

    # Source language: --audio_language takes priority over legacy --Initial_language
    source_lang = args.audio_language if args.audio_language is not None else args.Initial_language
    source_lang = source_lang.strip().lower() if source_lang else "en"

    # Target language: --output_language takes priority over legacy --target_language
    target_lang = args.output_language if args.output_language is not None else args.target_language

    # Validate and normalize the target language using the languages module
    target_lang = normalize_target_language(target_lang)

    return source_lang, target_lang


# ---------------------------------------------------------------------------
# XTTS health check
# ---------------------------------------------------------------------------

def _run_xtts_healthcheck(policy: str) -> int:
    """Attempt to load the TTS stack and report status."""
    print("=" * 50)
    print("XTTS Health Check")
    print("=" * 50)

    if policy == "fallback_only":
        print("\nBackend policy is 'fallback_only' -- testing gTTS only.")
        try:
            from gtts import gTTS  # noqa: F401
            print("  [OK] gTTS is available.")
            return 0
        except ImportError:
            print("  [FAIL] gTTS is NOT installed.")
            return 1

    # Test XTTS
    print("\nTesting Coqui XTTS import ...")
    try:
        from src.tts import _prepare_windows_dll_search_paths
        _prepare_windows_dll_search_paths()
        from TTS.api import TTS  # noqa: F401
        print("  [OK] Coqui TTS imported successfully.")
    except Exception as exc:
        print(f"  [FAIL] Coqui TTS import failed: {exc}")
        if policy == "strict_clone":
            print("  -> With 'strict_clone' policy, pipeline will FAIL without XTTS.")
            return 1
        print("  -> With 'fallback_allowed' policy, pipeline will fall back to gTTS.")

    # Test gTTS as well
    print("\nTesting gTTS import ...")
    try:
        from gtts import gTTS  # noqa: F401
        print("  [OK] gTTS is available.")
    except ImportError:
        print("  [WARN] gTTS is NOT installed (fallback unavailable).")

    print("\nHealth check complete.")
    return 0


def _auto_detect_narrators(video_path: Path, segments: list[dict], threshold: float = 0.10) -> list[int]:
    """
    Sample frames from each speaker's active segments and check for face presence.
    If a speaker's segments contain a face in less than the threshold percentage of
    sampled frames, they are classified as an off-screen narrator.
    """
    import cv2
    import os
    
    # Locate OpenCV's Haar Cascade XML for frontal face detection
    cascade_dir = getattr(cv2, "data", None)
    if cascade_dir and hasattr(cascade_dir, "haarcascades"):
        cascade_path = os.path.join(cascade_dir.haarcascades, "haarcascade_frontalface_default.xml")
    else:
        cascade_path = "haarcascade_frontalface_default.xml"
        
    if not os.path.exists(cascade_path):
        print(f"  [WARN] Face detector XML not found at {cascade_path}. Skipping narrator auto-detection.")
        return []
        
    try:
        face_cascade = cv2.CascadeClassifier(cascade_path)
    except Exception as e:
        print(f"  [WARN] Failed to load CascadeClassifier: {e}. Skipping narrator auto-detection.")
        return []
        
    # Group segments by speaker_id
    speaker_segments = {}
    for seg in segments:
        spk_id = seg.get("speaker_id")
        if spk_id is None:
            continue
        try:
            spk_id = int(spk_id)
        except (ValueError, TypeError):
            continue
        speaker_segments.setdefault(spk_id, []).append(seg)
        
    if not speaker_segments:
        return []
        
    # Open the video to read frames
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Failed to open video: {video_path} for narrator detection.")
        return []
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        cap.release()
        return []
        
    print("\n[Narrator Detection] Running face visibility analysis on speaker segments...")
    narrator_ids = []
    
    for spk_id, segs in speaker_segments.items():
        # Collect target timestamps/frames to sample (e.g. 1 frame per second of speech)
        sample_frames = []
        for seg in segs:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            # Sample every 1.0 seconds
            t = start
            while t <= end:
                frame_idx = int(round(t * fps))
                if 0 <= frame_idx < total_frames:
                    sample_frames.append(frame_idx)
                t += 1.0
                
        # Deduplicate and sort
        sample_frames = sorted(list(set(sample_frames)))
        if not sample_frames:
            continue
            
        faces_detected_count = 0
        total_samples = len(sample_frames)
        
        for frame_idx in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
                
            # Resize frame to speed up face detection (e.g. max height/width 480)
            h, w = frame.shape[:2]
            scale = 1.0
            if max(h, w) > 480:
                scale = 480.0 / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=4,
                minSize=(25, 25)
            )
            if len(faces) > 0:
                faces_detected_count += 1
                
        presence_ratio = faces_detected_count / total_samples if total_samples > 0 else 0.0
        print(f"  Speaker {spk_id}: face visible in {faces_detected_count}/{total_samples} samples ({presence_ratio * 100:.1f}%)")
        
        # If face is visible in less than 10% of samples, classify as narrator
        if presence_ratio < threshold:
            narrator_ids.append(spk_id)
            print(f"    -> Speaker {spk_id} classified as NARRATOR (off-screen)")
        else:
            print(f"    -> Speaker {spk_id} classified as ON-SCREEN speaker")
            
    cap.release()
    return narrator_ids


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    args = build_arg_parser().parse_args()

    # --- Early-exit utility commands ---
    if args.list_languages:
        print(format_supported_target_languages())
        return 0

    if args.xtts_healthcheck_only:
        return _run_xtts_healthcheck(args.tts_backend_policy)

    # --- Resolve languages early (validates target language) ---
    source_lang, target_lang = _resolve_languages(args)
    print(f"Source language: {source_lang}")
    print(f"Target language: {target_lang}")

    ensure_ffmpeg_available()
    device = resolve_runtime_device(requested_device=args.device, require_cuda=args.require_cuda)
    print(f"Using runtime device: {device}")

    input_video = Path(args.input_video)
    output_video = Path(args.output_video)
    working_dir = Path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    if not input_video.exists():
        candidates: list[str] = []
        input_dir = Path("input")
        if input_dir.exists() and input_dir.is_dir():
            candidates = sorted(str(path) for path in input_dir.glob("*.mp4"))

        candidate_suffix = ""
        if candidates:
            candidate_suffix = "\nAvailable .mp4 files under input/:\n- " + "\n- ".join(candidates)

        raise FileNotFoundError(
            f"Input video not found: {input_video}. "
            "Update --input_video to an existing file path."
            + candidate_suffix
        )

    # --- Pipeline checkpoint state ---
    state_path = working_dir / ".pipeline_state.json"
    resume = args.resume
    state = _load_pipeline_state(state_path) if resume else {"completed_steps": []}

    # --- Intermediate file paths ---
    extracted_audio_path = working_dir / "extracted_audio.wav"
    preprocessed_audio_path = working_dir / "preprocessed_audio.wav"
    voice_sample_path = working_dir / "speaker_sample.wav"
    translated_audio_raw_path = working_dir / "translated_audio_raw.wav"
    translated_audio_synced_path = working_dir / "translated_audio_synced.wav"
    wav2lip_output_path = working_dir / "wav2lip_output.mp4"
    postprocessed_video_path = working_dir / "wav2lip_postprocessed.mp4"
    color_matched_video_path = working_dir / "wav2lip_color_matched.mp4"

    # =========================================================================
    # STEP 1: Extract audio
    # =========================================================================
    if _step_already_done(state, "extract_audio", resume) and extracted_audio_path.exists():
        print("[SKIP] Skipping audio extraction (resume).")
    else:
        print("\n[1/10] Extracting audio from video ...")
        extract_audio_from_video(input_video, extracted_audio_path)
        _mark_step_done(state_path, state, "extract_audio")

    # =========================================================================
    # STEP 2: Audio preprocessing (optional)
    # =========================================================================
    audio_for_transcription = extracted_audio_path

    if args.enable_preprocessing:
        if _step_already_done(state, "preprocess_audio", resume) and preprocessed_audio_path.exists():
            print("[SKIP] Skipping audio preprocessing (resume).")
            audio_for_transcription = preprocessed_audio_path
        else:
            print("\n[2/10] Preprocessing audio for better transcription ...")
            from src.preprocessing import preprocess_audio_for_transcription
            preprocess_audio_for_transcription(
                extracted_audio_path,
                preprocessed_audio_path,
                normalize=True,
                denoise=True,
                bandpass=True,
                compress=True,
            )
            audio_for_transcription = preprocessed_audio_path
            _mark_step_done(state_path, state, "preprocess_audio")
            print("  [OK] Audio preprocessing complete.")

    # =========================================================================
    # STEP 3: Transcribe
    # =========================================================================
    # Transcription result cannot be easily cached as a file — always re-run.
    print("\n[3/10] Transcribing audio ...")
    whisper_language = source_lang if source_lang != "auto" else None
    transcription_result = transcribe_english_audio(
        audio_for_transcription,
        model_size=args.whisper_model,
        device=device,
        language=whisper_language,
    )
    if len(transcription_result) == 3:
        transcript_text, transcript_segments, detected_language = transcription_result
        print(f"  Whisper detected language: {detected_language}")
    else:
        transcript_text, transcript_segments = transcription_result

    _mark_step_done(state_path, state, "transcribe")

    # =========================================================================
    # STEP 3b: Segment optimization (optional)
    # =========================================================================
    if args.optimize_segments and transcript_segments:
        print("\n[3b/10] Optimizing transcription segments ...")
        from src.quality_analysis import optimize_transcription_segments
        original_count = len(transcript_segments)
        transcript_segments = optimize_transcription_segments(
            transcript_segments,
            min_confidence=0.80,
            max_segment_duration=5.0,
            min_segment_duration=0.5,
        )
        print(f"  Segments: {original_count} -> {len(transcript_segments)}")

    # =========================================================================
    # STEP 4: Extract voice sample
    # =========================================================================
    extract_voice_sample(
        extracted_audio_path,
        voice_sample_path,
        sample_seconds=args.extract_sample_seconds,
    )

    # =========================================================================
    # STEP 5: Translate + TTS
    # =========================================================================
    if _step_already_done(state, "tts", resume) and translated_audio_synced_path.exists():
        print("[SKIP] Skipping translation + TTS (resume).")
    else:
        used_segment_timing = False
        if args.timing_mode == "segment" and transcript_segments:
            try:
                print("\n[5/10] Translating segments ...")
                translated_segments = translate_segments(
                    transcript_segments,
                    source_language=source_lang,
                    target_language=target_lang,
                )

                # Save translated text if requested
                if args.translated_text_path:
                    _save_translated_text(translated_segments, args.translated_text_path)

                segment_reference_dir = working_dir / "segment_references"
                segment_reference_dir.mkdir(parents=True, exist_ok=True)
                for index, segment in enumerate(translated_segments):
                    source_start = float(segment.get("start", 0.0))
                    source_end = float(segment.get("end", source_start))
                    segment_reference_path = segment_reference_dir / f"segment_{index:04d}.wav"
                    extract_audio_segment(
                        extracted_audio_path,
                        segment_reference_path,
                        start_seconds=source_start,
                        end_seconds=source_end,
                    )
                    segment["speaker_wav_path"] = segment_reference_path

                # Call speaker diarization if enabled
                if args.diarize_speakers:
                    num_detected = diarize_and_extract_speakers(
                        segments=translated_segments,
                        working_dir=working_dir,
                        tts_model_name=args.tts_model_name,
                        device=device,
                        num_speakers=args.num_speakers,
                    )
                    print(f"  [OK] Speaker diarization complete. {num_detected} speakers processed.")

                # Save translated segments metadata for subsequent steps / resumes
                class NumpyEncoder(json.JSONEncoder):
                    def default(self, obj):
                        try:
                            import numpy as np
                            if isinstance(obj, (np.integer, np.int64, np.int32)):
                                return int(obj)
                            if isinstance(obj, (np.floating, np.float64, np.float32)):
                                return float(obj)
                            if isinstance(obj, np.ndarray):
                                return obj.tolist()
                        except ImportError:
                            pass
                        return super(NumpyEncoder, self).default(obj)

                segments_json_path = working_dir / "translated_segments.json"
                serializable_segments = []
                for seg in translated_segments:
                    s_copy = seg.copy()
                    if "speaker_wav_path" in s_copy and isinstance(s_copy["speaker_wav_path"], Path):
                        s_copy["speaker_wav_path"] = str(s_copy["speaker_wav_path"])
                    serializable_segments.append(s_copy)
                segments_json_path.write_text(json.dumps(serializable_segments, cls=NumpyEncoder, indent=2, ensure_ascii=False), encoding="utf-8")

                print("\n[5b/10] Synthesizing aligned TTS audio ...")
                synthesize_aligned_audio_from_segments(
                    segments=translated_segments,
                    speaker_wav_path=voice_sample_path,
                    output_audio_path=translated_audio_raw_path,
                    model_name=args.tts_model_name,
                    language=target_lang,
                    gpu=device == "cuda",
                    backend_policy=args.tts_backend_policy,
                )
                pad_or_trim_audio_to_video_duration(
                    video_path=input_video,
                    audio_path=translated_audio_raw_path,
                    output_audio_path=translated_audio_synced_path,
                )
                used_segment_timing = True
                print("  Timing mode: segment-aligned")
            except Exception as exc:
                print(f"  [WARN] Segment timing failed ({exc}); falling back to global timing.")

        if not used_segment_timing:
            print("\n[5/10] Translating full text ...")
            translated_text = translate_text(
                transcript_text,
                source_language=source_lang,
                target_language=target_lang,
            )

            # Save translated text if requested
            if args.translated_text_path:
                Path(args.translated_text_path).parent.mkdir(parents=True, exist_ok=True)
                Path(args.translated_text_path).write_text(translated_text, encoding="utf-8")
                print(f"  Translated text saved to: {args.translated_text_path}")

            print("\n[5b/10] Synthesizing TTS audio ...")
            synthesize_speech(
                text=translated_text,
                speaker_wav_path=voice_sample_path,
                output_audio_path=translated_audio_raw_path,
                model_name=args.tts_model_name,
                language=target_lang,
                gpu=device == "cuda",
                backend_policy=args.tts_backend_policy,
            )

            stretch_audio_to_video_duration(
                video_path=input_video,
                audio_path=translated_audio_raw_path,
                output_audio_path=translated_audio_synced_path,
            )
            print("  Timing mode: global-stretch")

        _mark_step_done(state_path, state, "translate")
        _mark_step_done(state_path, state, "tts")

    # =========================================================================
    # STEP 6: Wav2Lip
    # =========================================================================
    if _step_already_done(state, "wav2lip", resume) and wav2lip_output_path.exists():
        print("[SKIP] Skipping Wav2Lip (resume).")
    else:
        print("\n[6/10] Running Wav2Lip lip-sync ...")
        # Check for narrator isolation and exclusion intervals
        exclude_intervals_path = None
        if args.diarize_speakers and (args.narrator_speaker_ids or args.auto_detect_narrator):
            segments_json_path = working_dir / "translated_segments.json"
            if segments_json_path.exists():
                try:
                    with open(segments_json_path, "r", encoding="utf-8") as f:
                        loaded_segments = json.load(f)

                    narrator_ids = []
                    if args.narrator_speaker_ids:
                        narrator_ids.extend([
                            int(x.strip())
                            for x in args.narrator_speaker_ids.split(",")
                            if x.strip().lstrip("-").isdigit()
                        ])

                    if args.auto_detect_narrator:
                        detected_ids = _auto_detect_narrators(input_video, loaded_segments, threshold=0.10)
                        for d_id in detected_ids:
                            if d_id not in narrator_ids:
                                narrator_ids.append(d_id)

                    if narrator_ids:
                        print(f"  Excluding speaker(s) {narrator_ids} (narrator) from lip-syncing.")
                        exclude_intervals = []
                        for seg in loaded_segments:
                            spk_id = seg.get("speaker_id")
                            if spk_id is not None and int(spk_id) in narrator_ids:
                                exclude_intervals.append([float(seg.get("start", 0.0)), float(seg.get("end", 0.0))])

                        if exclude_intervals:
                            exclude_intervals_path = working_dir / "exclude_intervals.json"
                            exclude_intervals_path.write_text(json.dumps(exclude_intervals, indent=2), encoding="utf-8")
                            print(f"  Saved exclusion intervals to: {exclude_intervals_path.name}")
                except Exception as e:
                    print(f"  [WARN] Failed to process narrator isolation: {e}")
            else:
                print("  [WARN] translated_segments.json not found. Cannot perform narrator isolation.")

        run_wav2lip_inference(
            checkpoint_path=args.checkpoint_path,
            face_video_path=input_video,
            audio_path=translated_audio_synced_path,
            output_video_path=wav2lip_output_path,
            wav2lip_root=args.wav2lip_root,
            pads=tuple(args.wav2lip_pads),
            resize_factor=args.wav2lip_resize_factor,
            no_smooth=args.wav2lip_no_smooth,
            face_det_batch_size=args.wav2lip_face_det_batch_size,
            wav2lip_batch_size=args.wav2lip_batch_size,
            crop=tuple(args.wav2lip_crop),
            rotate=args.wav2lip_rotate,
            box=tuple(args.wav2lip_box) if args.wav2lip_box is not None else None,
            exclude_intervals_path=exclude_intervals_path,
        )
        _mark_step_done(state_path, state, "wav2lip")

    # =========================================================================
    # STEP 7: GFPGAN face enhancement (optional)
    # =========================================================================
    video_for_postprocess = wav2lip_output_path
    if args.enhance:
        enhanced_temp_path = working_dir / "wav2lip_enhanced.mp4"
        if _step_already_done(state, "enhance", resume) and enhanced_temp_path.exists():
            print("[SKIP] Skipping GFPGAN enhancement (resume).")
            video_for_postprocess = enhanced_temp_path
        else:
            print("\n[7/10] Enhancing faces with GFPGAN ...")
            enhance_faces_in_video(wav2lip_output_path, enhanced_temp_path, device=device)
            video_for_postprocess = enhanced_temp_path
            _mark_step_done(state_path, state, "enhance")

    # =========================================================================
    # STEP 8: Post-processing
    # =========================================================================
    if _step_already_done(state, "postprocess", resume) and postprocessed_video_path.exists():
        print("[SKIP] Skipping post-processing (resume).")
    else:
        if args.enable_advanced_postprocessing:
            print("\n[8/10] Running advanced post-processing (optical flow smoothing) ...")
            from src.postprocessing import advanced_postprocess_with_optical_flow
            advanced_postprocess_with_optical_flow(
                video_path=video_for_postprocess,
                output_path=postprocessed_video_path,
                denoise_strength=args.postprocess_denoise_strength,
                sharpen_amount=args.postprocess_sharpen_amount,
                contrast=args.postprocess_contrast,
                saturation=args.postprocess_saturation,
                crf=args.postprocess_crf,
                preset=args.postprocess_preset,
                enable_optical_flow_smoothing=True,
            )
        else:
            print("\n[8/10] Running post-processing ...")
            postprocess_video_quality(
                video_path=video_for_postprocess,
                output_video_path=postprocessed_video_path,
                denoise_strength=args.postprocess_denoise_strength,
                sharpen_amount=args.postprocess_sharpen_amount,
                contrast=args.postprocess_contrast,
                saturation=args.postprocess_saturation,
                crf=args.postprocess_crf,
                preset=args.postprocess_preset,
            )
        _mark_step_done(state_path, state, "postprocess")

    # =========================================================================
    # STEP 9: Color matching (optional)
    # =========================================================================
    video_for_mux = postprocessed_video_path
    if args.enable_color_matching:
        if _step_already_done(state, "color_match", resume) and color_matched_video_path.exists():
            print("[SKIP] Skipping color matching (resume).")
            video_for_mux = color_matched_video_path
        else:
            print("\n[9/10] Matching output colors to original video ...")
            from src.postprocessing import apply_histogram_matching
            apply_histogram_matching(
                video_path=postprocessed_video_path,
                reference_video=input_video,
                output_path=color_matched_video_path,
                preserve_luminance=True,
            )
            video_for_mux = color_matched_video_path
            _mark_step_done(state_path, state, "color_match")

    # =========================================================================
    # STEP 10: Final mux
    # =========================================================================
    if _step_already_done(state, "mux", resume) and output_video.exists():
        print("[SKIP] Skipping final mux (resume).")
    else:
        print("\n[10/10] Muxing final video ...")
        mux_video_with_audio(
            video_path=video_for_mux,
            audio_path=translated_audio_synced_path,
            output_video_path=output_video,
        )
        _mark_step_done(state_path, state, "mux")

    print(f"\n[DONE] Final output written to: {output_video}")
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_translated_text(segments: list[dict], path: str) -> None:
    """Write translated segment text to a file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = str(seg.get("text", "")).strip()
        lines.append(f"[{start:.2f} - {end:.2f}]  {text}")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Translated text saved to: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
