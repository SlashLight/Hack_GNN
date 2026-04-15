import pytest
from smart_radar_graph.schema import ParamInfo, FieldInfo, RawEndpoint


class TestParamInfo:
    def test_creation(self):
        p = ParamInfo(name="user_id", type="integer")
        assert p.name == "user_id"
        assert p.type == "integer"

    def test_default_type(self):
        p = ParamInfo(name="q")
        assert p.type == "string"


class TestFieldInfo:
    def test_creation(self):
        f = FieldInfo(name="email", type="string", depth=0)
        assert f.depth == 0

    def test_nested_field(self):
        f = FieldInfo(name="city", type="string", depth=2)
        assert f.depth == 2


class TestRawEndpoint:
    def test_minimal_endpoint(self):
        ep = RawEndpoint(path_template="/api/health", method="GET")
        assert ep.path_params == []
        assert ep.query_params == []
        assert ep.header_params == []
        assert ep.request_body_fields == []
        assert ep.response_body_fields == []
        assert ep.response_is_array is False
        assert ep.auth_type == "none"
        assert ep.auth_required is False
        assert ep.raw_examples == []
        assert ep.request_content_type is None
        assert ep.response_content_type is None

    def test_full_endpoint(self):
        ep = RawEndpoint(
            path_template="/api/v1/users/{id}/posts",
            method="GET",
            path_params=["id"],
            query_params=[ParamInfo(name="page", type="integer")],
            header_params=[ParamInfo(name="X-Custom", type="string")],
            request_content_type="application/json",
            request_body_fields=[FieldInfo(name="title", type="string", depth=0)],
            response_content_type="application/json",
            response_body_fields=[FieldInfo(name="data", type="array", depth=0)],
            response_is_array=True,
            auth_type="bearer-oauth",
            auth_required=True,
            raw_examples=[{"req": {}, "resp": {}}],
        )
        assert ep.path_params == ["id"]
        assert len(ep.query_params) == 1
        assert ep.auth_type == "bearer-oauth"
