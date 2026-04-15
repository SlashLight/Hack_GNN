import pytest
from smart_radar_graph.extractors.openapi import extract_from_spec
from smart_radar_graph.schema import RawEndpoint

MINIMAL_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Test", "version": "1.0"},
    "paths": {
        "/api/users": {
            "get": {
                "parameters": [
                    {"name": "page", "in": "query", "schema": {"type": "integer"}},
                    {"name": "X-Custom", "in": "header", "schema": {"type": "string"}},
                ],
                "responses": {"200": {"content": {"application/json": {"schema": {
                    "type": "array",
                    "items": {"type": "object", "properties": {
                        "id": {"type": "integer"}, "name": {"type": "string"}
                    }}
                }}}}}
            },
            "post": {
                "requestBody": {"content": {"application/json": {"schema": {
                    "type": "object", "properties": {
                        "name": {"type": "string"},
                        "address": {"type": "object", "properties": {
                            "city": {"type": "string"}
                        }}
                    }
                }}}},
                "responses": {"201": {"content": {"application/json": {"schema": {
                    "type": "object", "properties": {"id": {"type": "integer"}}
                }}}}}
            }
        },
        "/api/users/{id}": {
            "get": {
                "parameters": [{"name": "id", "in": "path", "schema": {"type": "integer"}, "required": True}],
                "responses": {"200": {"content": {"application/json": {"schema": {
                    "type": "object", "properties": {"id": {"type": "integer"}, "name": {"type": "string"}}
                }}}}}
            }
        },
        "/api/health": {
            "get": {
                "security": [],
                "responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}}
            }
        }
    },
    "components": {"securitySchemes": {"bearerAuth": {"type": "http", "scheme": "bearer"}}},
    "security": [{"bearerAuth": []}],
}


class TestExtractFromSpec:
    def test_endpoint_count(self):
        assert len(extract_from_spec(MINIMAL_SPEC)) == 4

    def test_paths(self):
        paths = {(e.path_template, e.method) for e in extract_from_spec(MINIMAL_SPEC)}
        assert ("/api/users", "GET") in paths
        assert ("/api/users", "POST") in paths
        assert ("/api/users/{id}", "GET") in paths
        assert ("/api/health", "GET") in paths

    def test_query_params(self):
        ep = next(e for e in extract_from_spec(MINIMAL_SPEC) if e.path_template == "/api/users" and e.method == "GET")
        assert len(ep.query_params) == 1
        assert ep.query_params[0].name == "page"

    def test_header_params(self):
        ep = next(e for e in extract_from_spec(MINIMAL_SPEC) if e.path_template == "/api/users" and e.method == "GET")
        assert len(ep.header_params) == 1
        assert ep.header_params[0].name == "X-Custom"

    def test_path_params_no_braces(self):
        ep = next(e for e in extract_from_spec(MINIMAL_SPEC) if e.path_template == "/api/users/{id}")
        assert ep.path_params == ["id"]  # NOT ["{id}"]

    def test_request_body_fields(self):
        ep = next(e for e in extract_from_spec(MINIMAL_SPEC) if e.method == "POST")
        assert ep.request_content_type == "application/json"
        names = {f.name for f in ep.request_body_fields}
        assert "name" in names and "city" in names
        city = next(f for f in ep.request_body_fields if f.name == "city")
        assert city.depth == 1

    def test_response_is_array(self):
        ep = next(e for e in extract_from_spec(MINIMAL_SPEC) if e.path_template == "/api/users" and e.method == "GET")
        assert ep.response_is_array is True

    def test_auth_bearer(self):
        ep = next(e for e in extract_from_spec(MINIMAL_SPEC) if e.path_template == "/api/users" and e.method == "GET")
        assert ep.auth_type == "bearer-oauth"
        assert ep.auth_required is True

    def test_auth_override_none(self):
        ep = next(e for e in extract_from_spec(MINIMAL_SPEC) if e.path_template == "/api/health")
        assert ep.auth_type == "none"
        assert ep.auth_required is False

    def test_response_body_fields(self):
        ep = next(e for e in extract_from_spec(MINIMAL_SPEC) if e.path_template == "/api/users/{id}")
        names = {f.name for f in ep.response_body_fields}
        assert "id" in names and "name" in names
