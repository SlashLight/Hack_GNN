from __future__ import annotations

import zlib

from preprocessor.models import ProcessedEvent


class UnifiedVocabulary:
    """Unified vocabulary для param names и response keys.

    Two-pass: сначала build_from_events(), затем get_index().
    Максимум 128 entries, alphabetical ordering для determinism.
    Unknown names используют crc32 fallback в диапазоне [0, MAX_SIZE).
    """

    MAX_SIZE = 128

    def __init__(self) -> None:
        self._name_to_idx: dict[str, int] = {}

    def build_from_events(self, endpoint_events: dict[str, list[ProcessedEvent]]) -> None:
        """Первый проход: собрать все уникальные имена."""
        all_names: set[str] = set()
        for events in endpoint_events.values():
            for e in events:
                for p in e.parameters:
                    all_names.add(p.name)
                for k in e.response_keys:
                    all_names.add(k)

        # Alphabetical, max 128
        sorted_names = sorted(all_names)[: self.MAX_SIZE]
        self._name_to_idx = {name: idx for idx, name in enumerate(sorted_names)}

    def get_index(self, name: str) -> int:
        """Получить индекс. Known names — deterministic. Unknown — hash fallback in [0, MAX_SIZE)."""
        if name in self._name_to_idx:
            return self._name_to_idx[name]
        # Overflow fallback: crc32 into [0, MAX_SIZE) — same bin space as vocabulary
        return zlib.crc32(name.encode()) % self.MAX_SIZE

    @property
    def size(self) -> int:
        return self.MAX_SIZE
