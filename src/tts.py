from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Literal

import librosa
import numpy as np
import soundfile as sf


TTSBackendPolicy = Literal["strict_clone", "fallback_allowed", "fallback_only"]


class TTSBackendError(RuntimeError):
    """Base class for expected TTS backend failures."""


class XTTSUnavailableError(TTSBackendError):
    """Raised when XTTS import/model initialization fails."""


class XTTSRuntimeSynthesisError(TTSBackendError):
    """Raised when XTTS is available but synthesis fails."""


class GTTSError(TTSBackendError):
    """Raised when gTTS fallback is requested but unavailable/fails."""


def _synthesize_with_xtts(
    text: str,
    speaker_wav_path: Path,
    output_audio_path: Path,
    model_name: str,
    language: str,
    gpu: bool,
) -> Path:
    try:
        from TTS.api import TTS
    except Exception as exc:
        raise XTTSUnavailableError(
            "XTTS cannot be imported in this environment. "
            "Install/fix Coqui TTS dependencies or use --tts_backend_policy fallback_allowed."
        ) from exc

    try:
        tts = TTS(model_name=model_name, progress_bar=True, gpu=gpu)
    except Exception as exc:  # pragma: no cover - backend/model dependent
        raise XTTSUnavailableError(
            "XTTS model could not be initialized. "
            "Check model name, Python compatibility, and runtime dependencies."
        ) from exc

    try:
        tts.tts_to_file(
            text=text,
            speaker_wav=str(speaker_wav_path),
            language=language,
            file_path=str(output_audio_path),
        )
    except Exception as exc:  # pragma: no cover - backend/model dependent
        raise XTTSRuntimeSynthesisError(
            "XTTS failed during speech synthesis for the selected language/speaker sample."
        ) from exc

    return output_audio_path


def _synthesize_with_gtts(
    text: str,
    output_audio_path: Path,
    language: str,
) -> Path:
    try:
        from gtts import gTTS
    except ImportError as exc:
        raise GTTSError(
            "gTTS fallback is not available. Install gTTS or switch to an XTTS-compatible environment."
        ) from exc

    temp_mp3_path = output_audio_path.with_suffix(".tmp.mp3")
    try:
        gTTS(text=text, lang=language).save(str(temp_mp3_path))
    except Exception as exc:  # pragma: no cover - network/backend dependent
        raise GTTSError("gTTS synthesis failed for the selected language.") from exc

    try:
        try:
            from moviepy import AudioFileClip
        except ImportError:  # pragma: no cover - compatibility for older MoviePy versions
            from moviepy.editor import AudioFileClip

        clip = AudioFileClip(str(temp_mp3_path))
        try:
            write_kwargs = {
                "fps": 22050,
                "codec": "pcm_s16le",
                "verbose": False,
                "logger": None,
            }
            try:
                clip.write_audiofile(str(output_audio_path), **write_kwargs)
            except TypeError as exc:
                if "unexpected keyword argument" not in str(exc):
                    raise
                write_kwargs.pop("verbose", None)
                write_kwargs.pop("logger", None)
                clip.write_audiofile(str(output_audio_path), **write_kwargs)
        finally:
            clip.close()
    except Exception as exc:
        raise GTTSError("gTTS audio post-processing failed while converting MP3 to WAV.") from exc
    finally:
        if temp_mp3_path.exists():
            temp_mp3_path.unlink()

    return output_audio_path


def check_xtts_availability(model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2", gpu: bool = True) -> tuple[bool, str]:
    """Return whether XTTS can be imported and initialized in the current environment."""

    try:
        from TTS.api import TTS
    except Exception as exc:
        return False, f"XTTS import failed: {exc}"

    try:
        TTS(model_name=model_name, progress_bar=False, gpu=gpu)
    except Exception as exc:  # pragma: no cover - backend/model dependent
        return False, f"XTTS model initialization failed: {exc}"

    return True, "XTTS is available."


def synthesize_speech(
    text: str,
    speaker_wav_path: str | Path,
    output_audio_path: str | Path,
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    language: str = "es",
    gpu: bool = True,
    backend_policy: TTSBackendPolicy = "strict_clone",
) -> Path:
    """Generate speech with XTTS v2 voice cloning.

    Backend policy options:
    - strict_clone: fail if XTTS is unavailable/fails.
    - fallback_allowed: use XTTS first, then gTTS fallback if XTTS fails.
    - fallback_only: skip XTTS and always use gTTS.
    """

    text = text.strip()
    if not text:
        raise ValueError("Cannot synthesize empty text.")

    speaker_wav_path = Path(speaker_wav_path)
    output_audio_path = Path(output_audio_path)
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    if not speaker_wav_path.exists():
        raise FileNotFoundError(f"Speaker reference audio not found: {speaker_wav_path}")

    if backend_policy not in {"strict_clone", "fallback_allowed", "fallback_only"}:
        raise ValueError(f"Unsupported TTS backend policy: {backend_policy}")

    if backend_policy == "fallback_only":
        print("TTS backend policy=fallback_only: using gTTS (speaker cloning disabled).")
        return _synthesize_with_gtts(text=text, output_audio_path=output_audio_path, language=language)

    try:
        return _synthesize_with_xtts(
            text=text,
            speaker_wav_path=speaker_wav_path,
            output_audio_path=output_audio_path,
            model_name=model_name,
            language=language,
            gpu=gpu,
        )
    except TTSBackendError as exc:
        if backend_policy == "strict_clone":
            raise XTTSUnavailableError(
                "Speaker cloning is required, but XTTS is unavailable or failed. "
                "Install/fix Coqui XTTS, or rerun with --tts_backend_policy fallback_allowed."
            ) from exc

        print(
            "Warning: XTTS cloning unavailable; falling back to gTTS. "
            "Output speech will not match the original speaker voice."
        )
        return _synthesize_with_gtts(text=text, output_audio_path=output_audio_path, language=language)


def synthesize_spanish_audio(
    text: str,
    speaker_wav_path: str | Path,
    output_audio_path: str | Path,
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    language: str = "es",
    gpu: bool = True,
) -> Path:
    """Backwards-compatible wrapper for the previous function name."""

    return synthesize_speech(
        text=text,
        speaker_wav_path=speaker_wav_path,
        output_audio_path=output_audio_path,
        model_name=model_name,
        language=language,
        gpu=gpu,
    )


def synthesize_aligned_audio_from_segments(
    segments: list[dict],
    speaker_wav_path: str | Path,
    output_audio_path: str | Path,
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    language: str = "es",
    gpu: bool = True,
    backend_policy: TTSBackendPolicy = "strict_clone",
    sample_rate: int = 22050,
    max_stretch_ratio: float = 1.35,
) -> Path:
    """Synthesize translated speech segment-by-segment and place clips on original timestamps."""

    cleaned_segments: list[dict] = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        if not text or end <= start:
            continue
        cleaned_segments.append({"start": start, "end": end, "text": text})

    if not cleaned_segments:
        raise ValueError("No valid timed segments were provided for aligned TTS synthesis.")

    output_audio_path = Path(output_audio_path)
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    total_duration = max(float(segment["end"]) for segment in cleaned_segments) + 0.15
    total_samples = int(round(total_duration * sample_rate))
    combined_waveform = np.zeros(total_samples, dtype=np.float32)

    with tempfile.TemporaryDirectory(prefix="aligned_tts_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        for index, segment in enumerate(cleaned_segments):
            temp_segment_path = temp_dir_path / f"segment_{index:04d}.wav"
            synthesize_speech(
                text=segment["text"],
                speaker_wav_path=speaker_wav_path,
                output_audio_path=temp_segment_path,
                model_name=model_name,
                language=language,
                gpu=gpu,
                backend_policy=backend_policy,
            )

            segment_waveform, _segment_sr = librosa.load(str(temp_segment_path), sr=sample_rate, mono=True)
            if segment_waveform.size == 0:
                continue

            target_duration = max(0.08, float(segment["end"]) - float(segment["start"]))
            current_duration = segment_waveform.shape[0] / float(sample_rate)
            if current_duration > 0:
                stretch_ratio = current_duration / target_duration
                stretch_ratio = float(np.clip(stretch_ratio, 1.0 / max_stretch_ratio, max_stretch_ratio))
                segment_waveform = librosa.effects.time_stretch(segment_waveform, rate=stretch_ratio)

            target_samples = int(round(target_duration * sample_rate))
            if segment_waveform.shape[0] > target_samples:
                segment_waveform = segment_waveform[:target_samples]
            elif segment_waveform.shape[0] < target_samples:
                segment_waveform = np.pad(
                    segment_waveform,
                    (0, target_samples - segment_waveform.shape[0]),
                )

            start_index = int(round(float(segment["start"]) * sample_rate))
            end_index = min(start_index + segment_waveform.shape[0], combined_waveform.shape[0])
            if end_index <= start_index:
                continue

            combined_waveform[start_index:end_index] += segment_waveform[: end_index - start_index]

    peak = float(np.max(np.abs(combined_waveform))) if combined_waveform.size else 0.0
    if peak > 0.98:
        combined_waveform = combined_waveform * (0.98 / peak)

    sf.write(str(output_audio_path), combined_waveform, sample_rate)
    return output_audio_path
