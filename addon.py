from __future__ import annotations
import os
import time as _time
from urllib.parse import urlencode

import mitmproxy.http
from mitmproxy import ctx

from preprocessor.pipeline import RawHTTPEvent, _make_processed_event
from graph.builder import GraphBuilder
from graph.visualizer import draw_graph

# Фильтрация по хосту: задаётся через переменную окружения или напрямую здесь.
# Пример запуска: GNN_TARGET_HOST=api.example.com mitmproxy -s addon.py
# Если пустая строка — захватывается весь трафик.
_TARGET_HOST: str = os.environ.get("GNN_TARGET_HOST", "")

_builder = GraphBuilder()
_request_count = 0
_STATS_EVERY = 10


class GraphAddon:
    def response(self, flow: mitmproxy.http.HTTPFlow) -> None:
        global _request_count

        # Фильтруем по хосту, если задан.
        # _TARGET_HOST может быть "host" или "host:port".
        if _TARGET_HOST:
            req_host = f"{flow.request.pretty_host}:{flow.request.port}"
            if _TARGET_HOST not in req_host and _TARGET_HOST not in flow.request.pretty_host:
                return

        _request_count += 1

        req = flow.request
        resp = flow.response

        # Формы: перекодируем текстовые поля в url-encoded bytes
        # чтобы pipeline мог распарсить через parse_qs.
        # multipart: берём только текстовые поля (пропускаем файлы).
        # urlencoded: берём всё как есть.
        req_ct = req.headers.get("content-type", "")
        request_body = req.content
        # Переопределяем Content-Type в заголовках при необходимости
        req_headers = dict(req.headers)
        if "multipart/form-data" in req_ct:
            # Конвертируем multipart в url-encoded, чтобы pipeline получил значения.
            items = [
                (k, v.decode("utf-8", errors="replace")) if v.isascii()
                else (k, "<binary>")
                for k, v in req.multipart_form.items(multi=True)
            ]
            if items:
                request_body = urlencode(items).encode()
                req_headers["content-type"] = "application/x-www-form-urlencoded"
            else:
                request_body = b""
        elif "form" in req_ct:
            items = list(req.urlencoded_form.items(multi=True))
            if items:
                request_body = urlencode(items).encode()

        raw = RawHTTPEvent(
            method=req.method,
            url=req.url,
            headers=req_headers,
            request_body=request_body,
            status_code=resp.status_code,
            response_headers=dict(resp.headers),
            response_body=resp.content or b"",
            timestamp=_time.time(),
        )

        try:
            event = _make_processed_event(raw)
        except Exception as exc:
            ctx.log.warn(f"[graph] Skipping {req.url}: {exc}")
            return

        _builder.add_event(event)

        ctx.log.info(
            f"[graph] {event.endpoint_id} | params={len(event.parameters)}"
            f" | session={event.session_id}"
        )

        if _request_count % _STATS_EVERY == 0:
            G = _builder.to_networkx()
            draw_graph(G)
            ctx.log.info(
                f"[graph] Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}"
            )


addons = [GraphAddon()]
