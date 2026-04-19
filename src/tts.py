from __future__ import annotations

from pathlib import Path
import tempfile

import librosa
import numpy as np
import soundfile as sf


def synthesize_spanish_audio(
    text: str,
    speaker_wav_path: str | Path,
    output_audio_path: str | Path,
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    language: str = "es",
    gpu: bool = True,
) -> Path:
    """Generate Spanish speech with XTTS v2 voice cloning.

    If Coqui XTTS is unavailable (common on newer Python/Windows builds),
    falls back to gTTS without voice cloning.
    """

    text = text.strip()
    if not text:
        raise ValueError("Cannot synthesize empty text.")

    speaker_wav_path = Path(speaker_wav_path)
    output_audio_path = Path(output_audio_path)
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    if not speaker_wav_path.exists():
        raise FileNotFoundError(f"Speaker reference audio not found: {speaker_wav_path}")

    try:
        from TTS.api import TTS

        tts = TTS(model_name=model_name, progress_bar=True, gpu=gpu)
        tts.tts_to_file(
            text=text,
            speaker_wav=str(speaker_wav_path),
            language=language,
            file_path=str(output_audio_path),
        )
    except ImportError:
        # Fallback path for Python versions where Coqui XTTS wheels are unavailable.
        try:
            from gtts import gTTS
        except ImportError as exc:
            raise RuntimeError(
                "No TTS backend available. Install either 'TTS' (XTTS voice cloning) "
                "or 'gTTS' (fallback without speaker cloning)."
            ) from exc

        temp_mp3_path = output_audio_path.with_suffix(".tmp.mp3")
        gTTS(text=text, lang=language).save(str(temp_mp3_path))

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
            if temp_mp3_path.exists():
                temp_mp3_path.unlink()

    return output_audio_path


def synthesize_aligned_audio_from_segments(
    segments: list[dict],
    speaker_wav_path: str | Path,
    output_audio_path: str | Path,
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    language: str = "es",
    gpu: bool = True,
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
            synthesize_spanish_audio(
                text=segment["text"],
                speaker_wav_path=speaker_wav_path,
                output_audio_path=temp_segment_path,
                model_name=model_name,
                language=language,
                gpu=gpu,
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
