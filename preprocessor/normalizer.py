from __future__ import annotations
import re2  # google-re2
from urllib.parse import unquote

# Порядок важен: от специфичных к общим
_URL_PATTERNS: list[tuple[str, str]] = [
    # UUID: 8-4-4-4-12 hex
    (r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", "{ID}"),
    # Дата: YYYY-MM-DD
    (r"\d{4}-\d{2}-\d{2}", "{DATE}"),
    # Хэш: 24+ hex-символа подряд (без дефисов — не UUID)
    (r"[0-9a-fA-F]{24,}", "{HASH}"),
    # Целое число
    (r"\d+", "{INT}"),
]

# Компилируем паттерны один раз при импорте
_COMPILED_URL = [(re2.compile(p), r) for p, r in _URL_PATTERNS]

def normalize_url(path: str) -> str:
    """Заменяет переменные части URL на плейсхолдеры."""
    path = unquote(path)  # %34%32 → 42, %2e%2e → ..
    for pattern, replacement in _COMPILED_URL:
        path = pattern.sub(replacement, path)
    return path


