from __future__ import annotations

import os
from pathlib import Path
import platform
import shutil
import tempfile

import librosa
import numpy as np
import soundfile as sf


class XTTSUnavailableError(RuntimeError):
    """Raised when XTTS is requested but unavailable in the runtime."""


class XTTSRuntimeSynthesisError(RuntimeError):
    """Raised when XTTS is installed but synthesis fails at runtime."""


class GTTSError(RuntimeError):
    """Raised when gTTS fallback synthesis fails."""


_WINDOWS_DLL_HANDLES: list[object] = []


def _prepare_windows_dll_search_paths() -> None:
    """Add likely FFmpeg/torch directories to Windows DLL search path."""

    if platform.system().lower() != "windows":
        return

    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        return

    candidate_dirs: set[str] = set()

    ffmpeg_env_dir = os.environ.get("TORCHCODEC_FFMPEG_DIR", "").strip()
    if ffmpeg_env_dir:
        candidate_dirs.add(ffmpeg_env_dir)

    ffmpeg_executable = shutil.which("ffmpeg")
    if ffmpeg_executable:
        candidate_dirs.add(str(Path(ffmpeg_executable).parent))

    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    for entry in path_entries:
        entry = entry.strip().strip('"')
        if not entry:
            continue
        entry_path = Path(entry)
        if not entry_path.exists() or not entry_path.is_dir():
            continue
        if any(entry_path.glob("avutil-*.dll")):
            candidate_dirs.add(str(entry_path))

    try:
        import torch

        torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib_dir.exists():
            candidate_dirs.add(str(torch_lib_dir))
    except Exception:
        pass

    for dll_dir in sorted(candidate_dirs):
        try:
            _WINDOWS_DLL_HANDLES.append(add_dll_directory(dll_dir))
        except Exception:
            continue


def _synthesize_with_xtts(
    text: str,
    speaker_wav_path: Path,
    output_audio_path: Path,
    model_name: str,
    language: str,
    gpu: bool,
) -> None:
    """Run Coqui XTTS voice cloning and write directly to output path."""

    try:
        _prepare_windows_dll_search_paths()
        from TTS.api import TTS
    except Exception as exc:  # pragma: no cover - depends on environment
        os_hint = ""
        if platform.system().lower() == "windows":
            os_hint = (
                " On Windows, installing Coqui TTS may require Microsoft Visual C++ Build Tools "
                "(MSVC 14+). If torchcodec cannot load, install FFmpeg shared binaries and set "
                "TORCHCODEC_FFMPEG_DIR to that bin folder."
            )
        raise XTTSUnavailableError(
            "Coqui XTTS is not available in this environment. Install 'TTS' to enable voice cloning. "
            f"Original import error: {exc}."
            + os_hint
        ) from exc

    try:
        tts = TTS(model_name=model_name, progress_bar=True, gpu=gpu)
        tts.tts_to_file(
            text=text,
            speaker_wav=str(speaker_wav_path),
            language=language,
            file_path=str(output_audio_path),
        )
    except Exception as exc:  # pragma: no cover - depends on model/runtime
        raise XTTSRuntimeSynthesisError(f"XTTS synthesis failed: {exc}") from exc


def _synthesize_with_gtts(text: str, output_audio_path: Path, language: str) -> None:
    """Fallback non-cloning TTS path for environments without XTTS."""

    try:
        from gtts import gTTS
    except Exception as exc:  # pragma: no cover - depends on environment
        raise GTTSError("gTTS is not installed or unavailable.") from exc

    temp_mp3_path = output_audio_path.with_suffix(".tmp.mp3")
    try:
        gTTS(text=text, lang=language).save(str(temp_mp3_path))
    except Exception as exc:  # pragma: no cover - depends on network/runtime
        raise GTTSError(f"gTTS synthesis failed: {exc}") from exc

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
    except Exception as exc:  # pragma: no cover - depends on codec/runtime
        raise GTTSError(f"Failed converting gTTS output to WAV: {exc}") from exc
    finally:
        if temp_mp3_path.exists():
            temp_mp3_path.unlink()


def synthesize_speech(
    text: str,
    speaker_wav_path: str | Path,
    output_audio_path: str | Path,
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    language: str = "es",
    gpu: bool = True,
    backend_policy: str = "strict_clone",
) -> Path:
    """Synthesize speech with policy-controlled backends.

    backend_policy options:
    - strict_clone: require XTTS voice cloning, fail if unavailable.
    - fallback_allowed: prefer XTTS, fall back to gTTS if XTTS fails.
    - fallback_only: skip XTTS and use gTTS directly.
    """

    text = text.strip()
    if not text:
        raise ValueError("Cannot synthesize empty text.")

    speaker_wav_path = Path(speaker_wav_path)
    output_audio_path = Path(output_audio_path)
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    if not speaker_wav_path.exists():
        raise FileNotFoundError(f"Speaker reference audio not found: {speaker_wav_path}")

    normalized_policy = backend_policy.strip().lower()
    if normalized_policy not in {"strict_clone", "fallback_allowed", "fallback_only"}:
        raise ValueError(
            f"Invalid backend_policy '{backend_policy}'. Use one of: strict_clone, fallback_allowed, fallback_only."
        )

    if normalized_policy == "fallback_only":
        _synthesize_with_gtts(text=text, output_audio_path=output_audio_path, language=language)
        return output_audio_path

    try:
        _synthesize_with_xtts(
            text=text,
            speaker_wav_path=speaker_wav_path,
            output_audio_path=output_audio_path,
            model_name=model_name,
            language=language,
            gpu=gpu,
        )
        return output_audio_path
    except (XTTSUnavailableError, XTTSRuntimeSynthesisError):
        if normalized_policy == "strict_clone":
            raise

    _synthesize_with_gtts(text=text, output_audio_path=output_audio_path, language=language)
    return output_audio_path


def synthesize_spanish_audio(
    text: str,
    speaker_wav_path: str | Path,
    output_audio_path: str | Path,
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    language: str = "es",
    gpu: bool = True,
) -> Path:
    """Compatibility wrapper for old call sites; now policy-driven via synthesize_speech."""

    return synthesize_speech(
        text=text,
        speaker_wav_path=speaker_wav_path,
        output_audio_path=output_audio_path,
        model_name=model_name,
        language=language,
        gpu=gpu,
        backend_policy="strict_clone",
    )


def synthesize_aligned_audio_from_segments(
    segments: list[dict],
    speaker_wav_path: str | Path,
    output_audio_path: str | Path,
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    language: str = "es",
    gpu: bool = True,
    backend_policy: str = "strict_clone",
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
