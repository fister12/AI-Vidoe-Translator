from __future__ import annotations


def translate_text(text: str, source_language: str = "en", target_language: str = "es") -> str:
    """Translate text between languages."""

    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Cannot translate empty text.")

    try:
        from googletrans import Translator as GoogleTranslator

        translator = GoogleTranslator()
        translated = translator.translate(cleaned_text, src=source_language, dest=target_language)
        return translated.text.strip()
    except Exception:
        pass

    try:
        from deep_translator import GoogleTranslator as DeepGoogleTranslator
    except ImportError as exc:
        raise RuntimeError(
            "No supported translation backend is installed. Install dependencies from requirements.txt "
            "or add deep-translator for Python 3.13."
        ) from exc

    translated_text = DeepGoogleTranslator(source=source_language, target=target_language).translate(cleaned_text)
    if not translated_text:
        raise RuntimeError("Translation backend returned an empty result.")
    return translated_text.strip()


def translate_segments(
    segments: list[dict],
    source_language: str = "en",
    target_language: str = "es",
) -> list[dict]:
    """Translate Whisper segments while preserving timing fields."""

    translated_segments: list[dict] = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        translated_text = translate_text(
            text,
            source_language=source_language,
            target_language=target_language,
        )

        translated_segments.append(
            {
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", 0.0)),
                "text": translated_text,
            }
        )

    return translated_segments


def translate_text_to_spanish(text: str) -> str:
    """Backwards-compatible wrapper for language to language translation."""

    return translate_text(text, source_language="en", target_language="es")
