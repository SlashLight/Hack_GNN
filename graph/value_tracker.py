from __future__ import annotations
from collections import OrderedDict
from preprocessor.models import Parameter


class ValueTracker:
    """Отслеживание data dependency: значения из JSON-ответов → параметры последующих запросов.

    Принцип работы:
    - record_response() сохраняет leaf-значения из JSON-ответа эндпоинта.
    - find_dependencies() проверяет каждый параметр запроса против буфера:
      для path-параметров (location="path") использует param.original_value (реальное
      значение из URL, например "42"); для остальных — param.name. Возвращает
      (source, dst, matched_value, param_name) для каждого совпадения.
    """

    MAX_BUFFER: int = 10_000
    MIN_LEN: int = 2
    _IGNORE: frozenset = frozenset({"true", "false", "null", "0", "1", "", "none", "undefined"})

    def __init__(self) -> None:
        self._value_to_endpoint: OrderedDict[str, str] = OrderedDict()  # value → endpoint_id

    def record_response(self, endpoint_id: str, response_values: dict[str, str]) -> None:
        """Запоминает leaf-значения из JSON-ответа."""
        for value in response_values.values():
            v = str(value).strip()
            if len(v) < self.MIN_LEN or v.lower() in self._IGNORE:
                continue
            self._value_to_endpoint[v] = endpoint_id
            # FIFO при переполнении
            while len(self._value_to_endpoint) > self.MAX_BUFFER:
                self._value_to_endpoint.popitem(last=False)

    def find_dependencies(
        self,
        endpoint_id: str,
        parameters: list[Parameter],
    ) -> list[tuple[str, str, str, str]]:
        """Возвращает [(source_endpoint, current_endpoint, matched_value, param_name)].

        Для path-параметров (location="path") сверяет param.original_value с буфером.
        Для остальных — param.name. param_name всегда равен param.name (используется
        для построения dep_params multi-hot в edge features).

        Дедупликация: один и тот же (source, candidate) не эмитируется дважды.
        Один source может появляться несколько раз, если разные параметры матчатся к нему.
        """
        deps: list[tuple[str, str, str, str]] = []
        seen: set[tuple[str, str]] = set()  # (source, candidate) — не допускать дубли одного значения
        for param in parameters:
            candidate = param.original_value if param.location == "path" else param.name
            if candidate in self._value_to_endpoint:
                source = self._value_to_endpoint[candidate]
                if source != endpoint_id and (source, candidate) not in seen:
                    deps.append((source, endpoint_id, candidate, param.name))
                    seen.add((source, candidate))
        return deps
