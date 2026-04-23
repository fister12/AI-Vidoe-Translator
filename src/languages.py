from __future__ import annotations

from typing import Final


SUPPORTED_TARGET_LANGUAGES: Final[dict[str, str]] = {
    "ar": "Arabic",
    "cs": "Czech",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "hu": "Hungarian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "tr": "Turkish",
    "zh-cn": "Chinese (Simplified)",
}


LANGUAGE_ALIASES: Final[dict[str, str]] = {
    "arabic": "ar",
    "czech": "cs",
    "de": "de",
    "deutsch": "de",
    "english": "en",
    "en-us": "en",
    "en-gb": "en",
    "es": "es",
    "espanol": "es",
    "spanish": "es",
    "fr": "fr",
    "french": "fr",
    "hindi": "hi",
    "hungarian": "hu",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "dutch": "nl",
    "polish": "pl",
    "portuguese": "pt",
    "russian": "ru",
    "turkish": "tr",
    "chinese": "zh-cn",
    "chinese-simplified": "zh-cn",
    "simplified-chinese": "zh-cn",
    "zh": "zh-cn",
    "zh_cn": "zh-cn",
}


def normalize_target_language(language_value: str) -> str:
    """Normalize a user-provided target language to canonical code."""

    normalized = str(language_value or "").strip().lower().replace("_", "-")
    if not normalized:
        raise ValueError("Target language cannot be empty.")

    if normalized == "auto":
        raise ValueError("Target language cannot be 'auto'.")

    resolved = LANGUAGE_ALIASES.get(normalized, normalized)
    if resolved not in SUPPORTED_TARGET_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_TARGET_LANGUAGES))
        raise ValueError(
            f"Unsupported target language '{language_value}'. "
            f"Use one of: {supported} (or a common name like 'hindi')."
        )
    return resolved


def format_supported_target_languages() -> str:
    """Build a printable list of supported target languages."""

    lines = ["Supported target languages (code -> name):"]
    for code, label in sorted(SUPPORTED_TARGET_LANGUAGES.items()):
        lines.append(f"  {code:5} -> {label}")
    lines.append("You can pass either code or common name, e.g. --output_language hi or --output_language hindi")
    return "\n".join(lines)
