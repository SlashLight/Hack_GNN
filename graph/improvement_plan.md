# План улучшения препроцессинга для GNN-пентеста

## Текущее состояние

```
models.py       → 2 dataclass'а (Parameter, ProcessedEvent)
cleaner.py      → очистка HTML/JSON/static тел ответов
normalizer.py   → URL нормализация + маскирование токенов
extractor.py    → извлечение параметров из query/body/form
embedder.py     → заглушка (md5 → random vector)
builder.py      → 4 типа узлов, 4 типа рёбер → HeteroData
inspector.py    → Pyvis визуализация
```

**Разрыв с планом:** 4/6 типов узлов (частично), 4/9 рёбер, нет feature engineering.

### Заметка об исследовании библиотек (2026-03-19)

Исследованы библиотеки для PII/secret detection и URL normalization:

| Библиотека | Вердикт | Причина |
|---|---|---|
| **Microsoft Presidio** | Отложить | Ценен как фреймворк при 15+ паттернах. В regex-only режиме (без spaCy) <5ms. Рассмотреть когда `_MASK_PATTERNS` перерастёт 15 записей |
| **detect-secrets** (Yelp) | Запасной вариант | ~20 детекторов секретов (AWS, Stripe, GitHub tokens), 120KB, без ML. Но `analyze_string` — нестабильный внутренний API |
| **scrubadub** | Нет | Неактивен с 2023 |
| **piiranha** | Нет | Transformer-based, 30-150ms на CPU — не для real-time proxy |
| **commonregex-improved** | Нет | Заброшен, медленнее google-re2 |
| **trufflehog / gitleaks** | Нет | Go-бинарники, IPC overhead неприемлем для inline proxy |
| **url-normalize / urlnorm** | Нет | Нормализуют RFC 3986 (case, encoding), не path-параметры |

**Вывод:** для текущего масштаба (<15 паттернов) расширяем `_MASK_PATTERNS` вручную на google-re2. Миграция на Presidio `PatternRecognizer` — тривиальна, когда понадобится (те же regex, другая обёртка).

---

## Фаза 0: Исправления базового препроцессинга (normalizer.py, extractor.py)

### Задача 0.1: URL-decode перед нормализацией

**Проблема:** `normalize_url` получает raw path без декодирования percent-encoding. Это значит:
- `/api/users/%34%32` и `/api/users/42` создают **два разных узла** endpoint в графе
- Path traversal `%2e%2e%2f` не распознаётся regex `\d+`
- Все дальнейшие фазы (temporal edges, similarity, features) работают на «грязных» данных

**Решение:** одна строка в начале `normalize_url`:

```python
from urllib.parse import unquote

def normalize_url(path: str) -> str:
    path = unquote(path)  # %34%32 → 42, %2e%2e → ..
    for pattern, repl in _COMPILED_URL:
        path = pattern.sub(repl, path)
    return path
```

**Критерии приёмки:**
- [ ] `normalize_url("/api/users/%34%32")` == `normalize_url("/api/users/42")`
- [ ] `normalize_url("/files/%2e%2e%2fetc")` нормализует `..` корректно
- [ ] Существующие тесты не ломаются
- [ ] Добавлен тест на percent-encoded пути

**Зачем GNN:** Без этого дедупликация endpoint'ов неполная — граф содержит дубликаты узлов, что искажает degree, message passing и все метрики.

---

## Фаза 1: Обогащение моделей данных (models.py)

### Задача 1.1: Расширить ProcessedEvent

**Промпт для Sonnet:**

> Расширь `ProcessedEvent` в `models.py`. Добавь поля, которые нужны для полноценного гетерогенного графа. Вот текущий код: [вставить models.py]. Вот что нужно добавить:

```python
@dataclass
class Parameter:
    name: str
    value_type: str           # "int" | "uuid" | "string" | "jwt" | "email" | "hash"
    location: str             # "query" | "path" | "body" | "header" | "cookie"
    is_reflected: bool = False
    value_entropy: float = 0.0    # энтропия Шеннона значения
    is_sensitive: bool = False    # имя содержит password/token/secret/key

@dataclass
class ResponseSignature:
    status_code: int
    content_type: str
    body_hash: str                # sha256 первых 1KB
    schema_keys: list[str]        # top-level JSON ключи (если JSON)
    body_length: int
    has_error_pattern: bool       # содержит "error", "exception", "stack trace"

@dataclass
class ProcessedEvent:
    endpoint_id: str              # "POST /api/users/{INT}/upload"
    method: str                   # "GET" | "POST" | "PUT" | "DELETE" | "PATCH"
    path_depth: int               # количество сегментов в пути
    has_dynamic_segment: bool     # содержит {INT}, {ID}, {HASH}
    auth_type: str | None         # "Bearer" | "Cookie" | "Basic" | "ApiKey" | None
    parameters: list[Parameter] = field(default_factory=list)
    response: ResponseSignature | None = None
    trm_payload: str = ""
    timestamp: float = 0.0       # unix timestamp запроса
    # Поля для рёбер
    referer_endpoint: str | None = None    # нормализованный Referer → для ребра "calls"
    redirect_target: str | None = None     # Location header → для ребра "calls"
    user_role: str | None = None           # извлечённая роль из JWT/response
```

**Критерии приёмки:**
- [ ] Все поля с type hints и дефолтами
- [ ] Backward-compatible (старый код не ломается)
- [ ] Docstrings на каждый класс

---

### Задача 1.2: Добавить вычисление энтропии и чувствительности

**Промпт для Sonnet:**

> Добавь в `extractor.py` две функции: `compute_entropy(value: str) -> float` (энтропия Шеннона) и `is_sensitive_name(name: str) -> bool` (проверка по списку паттернов). Обнови `extract_parameters` чтобы заполнять новые поля `value_entropy`, `is_sensitive`, `location`.

```python
import math
from collections import Counter

_SENSITIVE_PATTERNS = [
    "password", "passwd", "pwd", "secret", "token", "api_key",
    "apikey", "auth", "session", "csrf", "ssn", "credit_card",
    "private", "key", "access_token", "refresh_token",
]

def compute_entropy(value: str) -> float:
    """Энтропия Шеннона строки. Высокая → случайные данные (токены, хэши)."""
    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )

def is_sensitive_name(name: str) -> bool:
    lower = name.lower()
    return any(p in lower for p in _SENSITIVE_PATTERNS)
```

**Зачем GNN:** `value_entropy` отличает настоящие токены от обычных строк. `is_sensitive` помечает параметры, утечка которых — уязвимость.

---

## Фаза 2: Расширение extractor.py

### Задача 2.1: Извлечение response signature

**Промпт для Sonnet:**

> Добавь в `extractor.py` функцию `extract_response_signature`. Она принимает status_code (int), headers (dict), cleaned_body (str) и возвращает `ResponseSignature`. Реализуй:
> - `schema_keys`: если content-type JSON, парсим и берём top-level ключи
> - `has_error_pattern`: regex по "error", "exception", "traceback", "stack"
> - `body_hash`: sha256 первых 1024 символов

```python
import hashlib
import re2

_ERROR_RE = re2.compile(
    r"(error|exception|traceback|stack\s*trace|internal\s*server|fatal|panic)",
    flags=re2.IGNORECASE,
)

def extract_response_signature(
    status_code: int,
    headers: dict[str, str],
    cleaned_body: str,
) -> ResponseSignature:
    ct = headers.get("content-type", "").lower().split(";")[0].strip()
    body_hash = hashlib.sha256(cleaned_body[:1024].encode()).hexdigest()[:16]
    
    schema_keys = []
    if "json" in ct:
        try:
            data = orjson.loads(cleaned_body.encode())
            if isinstance(data, dict):
                schema_keys = sorted(data.keys())[:20]
        except orjson.JSONDecodeError:
            pass
    
    return ResponseSignature(
        status_code=status_code,
        content_type=ct,
        body_hash=body_hash,
        schema_keys=schema_keys,
        body_length=len(cleaned_body),
        has_error_pattern=bool(_ERROR_RE.search(cleaned_body[:500])),
    )
```

---

### Задача 2.2: Улучшение reflection check

**Проблема:** Текущий reflection check в `extractor.py:61-63`:

```python
reflected = len(value) >= 4 and value in response_body
```

Это простой `substring in string`, который пропускает:
- **HTML-encoded reflection:** `<script>alert(1)</script>` в запросе → `&lt;script&gt;alert(1)&lt;/script&gt;` в ответе — не ловится, хотя это XSS reflection
- **Case-insensitive reflection:** `admin` → `Admin` — не ловится
- **URL-encoded reflection:** `../etc/passwd` → `%2e%2e/etc/passwd` — не ловится

**Решение:** проверять несколько вариантов значения:

```python
import html
from urllib.parse import unquote

def _is_reflected(value: str, response_body: str) -> bool:
    """Проверяет reflection с учётом encoding-вариантов."""
    if len(value) < _MIN_REFLECTION_LEN:
        return False

    # 1. Exact match (текущее поведение)
    if value in response_body:
        return True

    # 2. Case-insensitive
    value_lower = value.lower()
    body_lower = response_body.lower()
    if value_lower in body_lower:
        return True

    # 3. HTML entity decoded body
    body_unescaped = html.unescape(response_body)
    if value_lower in body_unescaped.lower():
        return True

    # 4. URL-decoded body
    body_url_decoded = unquote(response_body)
    if value_lower in body_url_decoded.lower():
        return True

    return False
```

**Критерии приёмки:**
- [ ] `_is_reflected("admin", "Hello Admin!")` → `True`
- [ ] `_is_reflected("<script>", "... &lt;script&gt; ...")` → `True`
- [ ] `_is_reflected("../etc", "... %2e%2e/etc ...")` → `True`
- [ ] Значения < 4 символов по-прежнему не считаются reflected
- [ ] Обратная совместимость: все текущие тесты проходят

**Зачем GNN:** Ребро `CONSUMES` (parameter → response_data) — единственный сигнал XSS-уязвимости в графе. Больше корректных CONSUMES рёбер = лучше XSS-детекция. Без этого улучшения GNN не видит reflected параметры с encoding-трансформациями, что делает XSS-детекцию неполной.

---

### Задача 2.3: Извлечение роли из JWT

**Промпт для Sonnet:**

> Добавь функцию `extract_jwt_claims(auth_header: str) -> dict | None`. Декодируй payload JWT (без верификации подписи — мы в контексте пентеста), извлеки `role`, `is_admin`, `groups`, `scope`. Не используй PyJWT, декодируй base64 вручную.

```python
import base64

def extract_jwt_claims(auth_header: str) -> dict | None:
    """Декодирует JWT payload без верификации подписи."""
    if not auth_header.lower().startswith("bearer "):
        return None
    token = auth_header.split(" ", 1)[1]
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        # Добавляем padding
        payload_b64 = parts[1] + "=" * (4 - len(parts[1]) % 4)
        payload = base64.urlsafe_b64decode(payload_b64)
        return orjson.loads(payload)
    except Exception:
        return None

def extract_user_role(claims: dict | None) -> str | None:
    """Извлекает роль пользователя из JWT claims."""
    if not claims:
        return None
    for key in ("role", "roles", "groups", "scope", "permissions", "is_admin"):
        if key in claims:
            val = claims[key]
            if isinstance(val, list):
                return ",".join(str(v) for v in val)
            return str(val)
    return None
```

**Зачем GNN:** Роль — это отдельный тип узла (UserRole). Ребро `accesses_as` связывает endpoint с ролью, под которой к нему обращались. GNN через link prediction найдёт эндпоинты, к которым конкретная роль не обращалась, но ДОЛЖНА иметь доступ (IDOR).

---

## Фаза 3: Новые типы рёбер (builder.py)

### Задача 3.1: Temporal edges (precedes)

**Промпт для Sonnet:**

> В `GraphBuilder` добавь отслеживание порядка запросов. После `add_event` записывай предыдущий endpoint. Если текущий endpoint отличается от предыдущего, добавляй ребро `precedes`. Также добавь edge feature `time_delta`.

```python
class GraphBuilder:
    def __init__(self) -> None:
        # ... существующие поля ...
        self._prev_endpoint: str | None = None
        self._prev_timestamp: float = 0.0
        # edge features хранятся параллельно с edge_index
        self._edge_features: dict[tuple, list[np.ndarray]] = defaultdict(list)

    def add_event(self, event: ProcessedEvent) -> None:
        # ... существующая логика ...
        
        # Temporal edge: precedes
        if self._prev_endpoint and self._prev_endpoint != event.endpoint_id:
            prev_idx = self._node_index["endpoint"][self._prev_endpoint]
            time_delta = event.timestamp - self._prev_timestamp
            self._add_edge("endpoint", "PRECEDES", "endpoint", prev_idx, ep_idx)
            # Edge feature: log(time_delta + 1)
        
        self._prev_endpoint = event.endpoint_id
        self._prev_timestamp = event.timestamp
```

**Зачем GNN:** `precedes` кодирует поведенческие паттерны. "login → admin_panel" встречается редко у обычных пользователей → подозрительный паттерн.

---

### Задача 3.2: Call edges (redirect chains, Referer)

**Промпт для Sonnet:**

> Добавь в `normalizer.py` нормализацию Referer header. В `add_event` создавай ребро `CALLS` если у ProcessedEvent заполнен `referer_endpoint` или `redirect_target`. Это кодирует навигационную структуру приложения.

```python
def add_event(self, event: ProcessedEvent) -> None:
    # ... после создания ep_idx ...
    
    # Call edges: Referer → current endpoint
    if event.referer_endpoint:
        ref_idx = self._get_or_create(
            "endpoint", event.referer_endpoint, event.referer_endpoint
        )
        self._add_edge("endpoint", "CALLS", "endpoint", ref_idx, ep_idx)
    
    # Redirect edge: current → redirect target
    if event.redirect_target:
        redir_idx = self._get_or_create(
            "endpoint", event.redirect_target, event.redirect_target
        )
        self._add_edge("endpoint", "REDIRECTS", "endpoint", ep_idx, redir_idx)
```

**Зачем GNN:** Граф навигации показывает какие эндпоинты связаны логически. Если admin_panel CALLS delete_user, но нет PROTECTED_BY на delete_user → missing authorization.

---

### Задача 3.3: Similar_to edges

**Промпт для Sonnet:**

> Реализуй пост-процессинг метод `add_similarity_edges()` в `GraphBuilder`. После добавления всех events, вычисли cosine similarity между endpoint embeddings. Если similarity > 0.85, добавь ребро `SIMILAR_TO`.

```python
from numpy.linalg import norm

def add_similarity_edges(self, threshold: float = 0.85) -> None:
    """Добавляет рёбра SIMILAR_TO между похожими endpoint'ами."""
    if "endpoint" not in self._node_features:
        return
    features = np.stack(self._node_features["endpoint"])
    norms = norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid div by zero
    normalized = features / norms
    sim_matrix = normalized @ normalized.T
    
    n = len(features)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] > threshold:
                self._add_edge("endpoint", "SIMILAR_TO", "endpoint", i, j)
                self._add_edge("endpoint", "SIMILAR_TO", "endpoint", j, i)
```

**Зачем GNN:** Похожие эндпоинты часто имеют одинаковые уязвимости. Если `/api/v1/users/{id}` уязвим к IDOR, то `/api/v1/orders/{id}` с высокой вероятностью тоже.

---

### Задача 3.4: UserRole и accesses_as

**Промпт для Sonnet:**

> Добавь в `add_event` создание узла типа `user_role` и ребра `ACCESSES_AS`. Роль извлекается из JWT claims (поле `event.user_role`). Ребро связывает endpoint → user_role.

```python
def add_event(self, event: ProcessedEvent) -> None:
    # ... после создания ep_idx ...
    
    # UserRole node + ACCESSES_AS edge
    if event.user_role:
        role_idx = self._get_or_create(
            "user_role", event.user_role, f"role:{event.user_role}"
        )
        self._add_edge("endpoint", "ACCESSES_AS", "user_role", ep_idx, role_idx)
```

---

## Фаза 4: Замена embedder.py

### Задача 4.1: SentenceTransformer embeddings

**Промпт для Sonnet:**

> Замени заглушку `embed()` в `embedder.py` на реальные эмбеддинги через SentenceTransformer. Используй модель `all-MiniLM-L6-v2` (быстрая, 384-мерные вектора). Добавь кэширование через LRU cache.

```python
from __future__ import annotations
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

@lru_cache(maxsize=4096)
def embed(text: str, dim: int = 384) -> np.ndarray:
    """Семантический эмбеддинг через SentenceTransformer."""
    model = _get_model()
    vec = model.encode(text, show_progress_bar=False)
    if vec.shape[0] != dim:
        # Если нужна другая размерность — проекция
        vec = vec[:dim] if vec.shape[0] > dim else np.pad(vec, (0, dim - vec.shape[0]))
    return vec.astype(np.float32)
```

**Зачем GNN:** Семантические эмбеддинги дают модели понимание смысла. "GET /api/users/{id}/profile" и "GET /api/accounts/{id}/info" будут близки в embedding space → `similar_to` edge будет точнее.

---

## Фаза 5: Feature engineering для узлов

### Задача 5.1: Структурные фичи endpoint

**Промпт для Sonnet:**

> Создай файл `features.py`. Реализуй функцию `endpoint_features(events: list[ProcessedEvent]) -> dict[str, np.ndarray]`. Для каждого endpoint_id вычисли вектор фичей из множества обращений к нему:

```python
def endpoint_features(events: list[ProcessedEvent]) -> dict[str, np.ndarray]:
    """Агрегированные фичи endpoint'а из всех обращений."""
    grouped: dict[str, list[ProcessedEvent]] = defaultdict(list)
    for e in events:
        grouped[e.endpoint_id].append(e)
    
    result = {}
    for eid, group in grouped.items():
        method_onehot = _method_onehot(group[0].method)        # 5 dim
        path_depth = group[0].path_depth                        # 1 dim
        has_dynamic = float(group[0].has_dynamic_segment)       # 1 dim
        n_params = np.mean([len(e.parameters) for e in group])  # 1 dim
        
        # Status code distribution
        codes = [e.response.status_code for e in group if e.response]
        status_dist = _status_distribution(codes)               # 5 dim (2xx,3xx,4xx,5xx,other)
        
        # Auth ratio
        auth_count = sum(1 for e in group if e.auth_type)
        auth_ratio = auth_count / len(group)                    # 1 dim
        
        # Чувствительных параметров ratio
        all_params = [p for e in group for p in e.parameters]
        sensitive_ratio = (
            sum(1 for p in all_params if p.is_sensitive) / max(len(all_params), 1)
        )                                                       # 1 dim
        
        # Reflected params ratio
        reflected_ratio = (
            sum(1 for p in all_params if p.is_reflected) / max(len(all_params), 1)
        )                                                       # 1 dim
        
        # Error ratio
        error_count = sum(
            1 for e in group if e.response and e.response.has_error_pattern
        )
        error_ratio = error_count / len(group)                  # 1 dim
        
        vec = np.concatenate([
            method_onehot,                      # 5
            [path_depth, has_dynamic, n_params], # 3
            status_dist,                         # 5
            [auth_ratio, sensitive_ratio],        # 2
            [reflected_ratio, error_ratio],       # 2
        ])
        result[eid] = vec.astype(np.float32)    # 17 dim total
    return result
```

**Зачем GNN:** Эти 17 фичей + 384 от embedding = 401-мерный вектор узла. GNN использует их в message passing для классификации чувствительности endpoint'а.

---

### Задача 5.2: Обновить builder для гибридных фичей

**Промпт для Sonnet:**

> Обнови `GraphBuilder._get_or_create` чтобы принимать опциональный `structural_features: np.ndarray`. Финальный вектор узла = concat(embedding, structural_features). Обнови `add_event` и `to_hetero_data` соответственно.

---

## Фаза 6: DataType как отдельный тип узла

### Задача 6.1: Кластеризация типов данных

**Промпт для Sonnet:**

> Добавь новый тип узла `data_type` в `builder.py`. Для каждого уникального `value_type` параметра создавай узел data_type. Ребро `HAS_TYPE` связывает parameter → data_type. Расширенные типы: "int", "uuid", "jwt", "email", "hash", "objectid", "boolean", "url", "date", "string".

> Обнови `extractor.py` → `infer_type()` чтобы определять все эти типы:

```python
_PATTERNS = [
    ("jwt",      re2.compile(r"^eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+$")),
    ("email",    re2.compile(r"^[^@]+@[^@]+\.[a-zA-Z]{2,}$")),
    ("uuid",     re2.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-...")),
    ("objectid", re2.compile(r"^[0-9a-f]{24}$")),
    ("url",      re2.compile(r"^https?://")),
    ("date",     re2.compile(r"^\d{4}-\d{2}-\d{2}")),
    ("boolean",  re2.compile(r"^(true|false|0|1)$", re2.IGNORECASE)),
    ("int",      re2.compile(r"^\d+$")),
    ("hash",     re2.compile(r"^[0-9a-f]{32,}$")),
]
```

**Зачем GNN:** DataType узлы создают «мосты» между параметрами разных endpoint'ов. Если два endpoint'а принимают UUID — GNN через message passing поймёт, что они работают с одним и тем же типом объекта → потенциальный IDOR.

---

## Фаза 7: Orchestrator (pipeline.py)

### Задача 7.1: Единый пайплайн

**Промпт для Sonnet:**

> Создай `pipeline.py` — оркестратор, который принимает сырые HTTP events (из Burp/mitmproxy) и прогоняет через весь пайплайн: normalizer → cleaner → extractor → features → builder → HeteroData.

```python
@dataclass
class RawHTTPEvent:
    method: str
    url: str
    headers: dict[str, str]
    request_body: bytes | None
    status_code: int
    response_headers: dict[str, str]
    response_body: bytes
    timestamp: float

def process_traffic(events: list[RawHTTPEvent]) -> HeteroData:
    builder = GraphBuilder()
    processed = []
    
    for raw in sorted(events, key=lambda e: e.timestamp):
        # Step 1: Normalize URL
        path = normalize_url(urlparse(raw.url).path)
        endpoint_id = f"{raw.method} {path}"
        
        # Step 2: Clean response
        ct = raw.response_headers.get("content-type", "")
        cleaned = clean_body(raw.response_body, ct)
        cleaned = mask_tokens(cleaned)
        
        # Step 3: Extract everything
        query = urlparse(raw.url).query
        params = extract_parameters(query, raw.request_body, None, cleaned)
        auth = extract_auth(raw.headers)
        jwt_claims = extract_jwt_claims(raw.headers.get("authorization", ""))
        role = extract_user_role(jwt_claims)
        response_sig = extract_response_signature(
            raw.status_code, raw.response_headers, cleaned
        )
        referer = raw.headers.get("referer", "")
        referer_ep = normalize_referer(referer) if referer else None
        redirect_target = extract_redirect(raw.status_code, raw.response_headers)
        
        # Step 4: Build ProcessedEvent
        event = ProcessedEvent(
            endpoint_id=endpoint_id,
            method=raw.method,
            path_depth=path.count("/"),
            has_dynamic_segment=any(p in path for p in ["{INT}", "{ID}", "{HASH}"]),
            auth_type=auth,
            parameters=params,
            response=response_sig,
            trm_payload=f"{endpoint_id} {cleaned[:200]}",
            timestamp=raw.timestamp,
            referer_endpoint=referer_ep,
            redirect_target=redirect_target,
            user_role=role,
        )
        processed.append(event)
        builder.add_event(event)
    
    # Post-processing edges
    builder.add_similarity_edges(threshold=0.85)
    
    return builder.to_hetero_data()
```

---

## Порядок выполнения задач

| # | Задача | Зависимости | Сложность | Влияние на GNN |
|---|--------|------------|-----------|----------------|
| **0.1** | **URL-decode в normalizer** | **—** | **Минимальная (1 строка)** | **Критическое (дедупликация)** |
| 1.1 | Расширить models | — | Низкая | Критическое (всё зависит от моделей) |
| 1.2 | Entropy + sensitivity | 1.1 | Низкая | Среднее |
| 2.1 | Response signature | 1.1 | Средняя | Высокое |
| **2.2** | **Улучшение reflection check** | **—** | **Низкая** | **Критическое (CONSUMES рёбра)** |
| 2.3 | JWT claims | 1.1 | Низкая | Высокое |
| 4.1 | ~~SentenceTransformer~~ **[ОТЛОЖЕНО]** | — | Низкая | Критическое |
| 3.1 | Temporal edges | 1.1 | Средняя | Высокое |
| 3.2 | Call edges | 1.1 | Средняя | Высокое |
| 3.3 | ~~Similar_to edges~~ **[ОТЛОЖЕНО]** | 4.1 | Средняя | Среднее |
| 3.4 | UserRole + accesses_as | 2.3 | Низкая | Высокое |
| 6.1 | DataType node | 1.1 | Средняя | Среднее |
| 5.1 | Endpoint features | 2.1 | Средняя | Высокое |
| 5.2 | Гибридные фичи в builder | 4.1, 5.1 | Средняя | Критическое |
| 7.1 | Pipeline orchestrator | Все выше | Высокая | Критическое |

**Рекомендуемый порядок:** **0.1** → 1.1 → 1.2 → 2.1 → **2.2** → 2.3 → 4.1 → 3.1 → 3.2 → 3.4 → 6.1 → 5.1 → 3.3 → 5.2 → 7.1

---

## Итоговый граф после всех фаз

```
Типы узлов: 6
  endpoint        (+ 17 structural features + 384 embedding = 401 dim)
  parameter       (+ entropy, location, sensitivity + 384 embedding)
  auth_context    (+ token_entropy, protects_n + 384 embedding)
  response_data   (+ schema_keys, error_pattern + 384 embedding)
  user_role       (384 embedding)
  data_type       (384 embedding)

Типы рёбер: 9
  ACCEPTS         endpoint → parameter
  PROTECTED_BY    endpoint → auth_context
  RETURNS         endpoint → response_data
  CONSUMES        parameter → response_data (reflected)
  PRECEDES        endpoint → endpoint (temporal)
  CALLS           endpoint → endpoint (referer/redirect)
  SIMILAR_TO      endpoint ↔ endpoint (cosine > 0.85)
  ACCESSES_AS     endpoint → user_role
  HAS_TYPE        parameter → data_type
```

## Отложенные задачи

### 4.1 SentenceTransformer
Отложено: embedder.py всё ещё использует md5-stub. Реализовать после стабилизации структуры графа.

### 3.3 SIMILAR_TO edges
Отложено: зависит от 4.1. Метод `add_similarity_edges()` не реализован в builder.py.
