from __future__ import annotations

from pathlib import Path

import whisper


def transcribe_english_audio(
    audio_path: str | Path,
    model_size: str = "small",
    device: str = "cuda",
    language: str | None = "en",
) -> tuple[str, list[dict], str]:
    """Transcribe speech with Whisper and return detected language."""

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = whisper.load_model(model_size, device=device)
    transcribe_kwargs: dict[str, object] = {"fp16": device == "cuda"}
    if language:
        transcribe_kwargs["language"] = language

    result = model.transcribe(str(audio_path), **transcribe_kwargs)
    detected_language = str(result.get("language", language or "")).strip().lower()
    return result["text"].strip(), list(result.get("segments", [])), detected_language
