import pytest
from preprocessor.models import ProcessedEvent, Parameter
from smart_radar_graph.extractors.traffic import extract_from_traffic


def _event(eid="GET /api/users", method="GET", **kw) -> ProcessedEvent:
    return ProcessedEvent(endpoint_id=eid, method=method, **kw)

def _param(name="user_id", location="query", value_type="int", **kw) -> Parameter:
    return Parameter(name=name, location=location, value_type=value_type, **kw)


class TestExtractFromTraffic:
    def test_basic(self):
        eps = extract_from_traffic({"GET /api/users": [_event()]})
        assert len(eps) == 1
        assert eps[0].path_template == "/api/users"
        assert eps[0].method == "GET"

    def test_query_params_union(self):
        events = {"GET /api/users": [
            _event(parameters=[_param("page"), _param("limit")]),
            _event(parameters=[_param("page"), _param("sort")]),
        ]}
        eps = extract_from_traffic(events)
        names = {p.name for p in eps[0].query_params}
        assert names == {"page", "limit", "sort"}

    def test_auth_type_mapping(self):
        events = {"GET /api/users": [
            _event(auth_type="Bearer"), _event(auth_type="Bearer"), _event(auth_type=None),
        ]}
        eps = extract_from_traffic(events)
        assert eps[0].auth_type == "bearer-oauth"
        # 2/3 = 66.7% < 80% threshold
        assert eps[0].auth_required is False

    def test_auth_required_at_threshold(self):
        events = {"GET /api/users": [
            _event(auth_type="Bearer"), _event(auth_type="Bearer"),
            _event(auth_type="Bearer"), _event(auth_type="Bearer"),
            _event(auth_type=None),
        ]}
        eps = extract_from_traffic(events)
        assert eps[0].auth_required is True  # 4/5 = 80%

    def test_path_params_stripped(self):
        events = {"GET /api/users/{INT}/posts": [_event(eid="GET /api/users/{INT}/posts")]}
        eps = extract_from_traffic(events)
        assert eps[0].path_params == ["INT"]  # no braces

    def test_response_keys_to_fields(self):
        events = {"GET /api/users": [_event(response_keys=["id", "name", "email"])]}
        eps = extract_from_traffic(events)
        names = {f.name for f in eps[0].response_body_fields}
        assert names == {"id", "name", "email"}

    def test_response_content_type_from_event(self):
        """response_content_type should come from ProcessedEvent.content_type, NOT request CT."""
        events = {"GET /api/users": [_event(content_type="application/json")]}
        eps = extract_from_traffic(events)
        assert eps[0].response_content_type == "application/json"

    def test_raw_examples_max_5(self):
        events = {"GET /api/users": [_event(response_values={"id": str(i)}) for i in range(10)]}
        eps = extract_from_traffic(events)
        assert len(eps[0].raw_examples) <= 5

    def test_multiple_endpoints(self):
        events = {
            "GET /api/users": [_event()],
            "POST /api/users": [_event(eid="POST /api/users", method="POST")],
        }
        assert len(extract_from_traffic(events)) == 2
