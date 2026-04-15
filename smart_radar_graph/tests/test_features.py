import numpy as np
import pytest
from smart_radar_graph.schema import RawEndpoint, ParamInfo, FieldInfo
from smart_radar_graph.features import extract_static_features, STATIC_DIM


def _ep(path="/api/users", method="GET", **kwargs) -> RawEndpoint:
    return RawEndpoint(path_template=path, method=method, **kwargs)


class TestFeatureDimension:
    def test_static_dim_is_32(self):
        assert STATIC_DIM == 32

    def test_output_shape(self):
        features = extract_static_features([_ep(), _ep(path="/api/posts", method="POST")])
        assert features.shape == (2, 32)
        assert features.dtype == np.float32

    def test_no_nan_no_inf(self):
        features = extract_static_features([_ep(), _ep(path="/admin/users/{id}", method="DELETE")])
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))


class TestHttpMethod:
    def test_get(self):
        f = extract_static_features([_ep(method="GET")])
        assert list(f[0, 0:5]) == [1, 0, 0, 0, 0]

    def test_post(self):
        f = extract_static_features([_ep(method="POST")])
        assert list(f[0, 0:5]) == [0, 1, 0, 0, 0]

    def test_put_and_patch_same_bin(self):
        f_put = extract_static_features([_ep(method="PUT")])
        f_patch = extract_static_features([_ep(method="PATCH")])
        assert list(f_put[0, 0:5]) == [0, 0, 1, 0, 0]
        assert list(f_patch[0, 0:5]) == [0, 0, 1, 0, 0]

    def test_delete(self):
        f = extract_static_features([_ep(method="DELETE")])
        assert list(f[0, 0:5]) == [0, 0, 0, 1, 0]

    def test_options_goes_to_other(self):
        f = extract_static_features([_ep(method="OPTIONS")])
        assert list(f[0, 0:5]) == [0, 0, 0, 0, 1]


class TestPathFeatures:
    def test_path_depth(self):
        f = extract_static_features([_ep(path="/api/v1/users")])
        assert f[0, 5] == 3.0

    def test_path_depth_root(self):
        f = extract_static_features([_ep(path="/")])
        assert f[0, 5] == 0.0

    def test_num_path_params(self):
        f = extract_static_features([_ep(path="/users/{id}/posts/{post_id}")])
        assert f[0, 6] == 2.0


class TestResourceCategory:
    def test_admin_path(self):
        f = extract_static_features([_ep(path="/admin/settings")])
        assert f[0, 7] == 1.0 and f[0, 8] == 0.0 and f[0, 9] == 0.0

    def test_auth_path(self):
        f = extract_static_features([_ep(path="/api/auth/login")])
        assert f[0, 8] == 1.0

    def test_user_data_path(self):
        f = extract_static_features([_ep(path="/api/users/profile")])
        assert f[0, 9] == 1.0

    def test_multi_hot(self):
        f = extract_static_features([_ep(path="/admin/users")])
        assert f[0, 7] == 1.0 and f[0, 9] == 1.0

    def test_parametric_segments_ignored(self):
        f = extract_static_features([_ep(path="/api/{admin}/data")])
        assert f[0, 7] == 0.0


class TestInputFeatures:
    def test_num_query_params(self):
        f = extract_static_features([_ep(query_params=[ParamInfo("page"), ParamInfo("limit")])])
        assert f[0, 10] == 2.0

    def test_has_request_body_with_ct(self):
        f = extract_static_features([_ep(request_content_type="application/json",
            request_body_fields=[FieldInfo("name", "string", 0)])])
        assert f[0, 12] == 1.0

    def test_has_request_body_fields_only(self):
        """Body fields without explicit content type → still True."""
        f = extract_static_features([_ep(request_body_fields=[FieldInfo("name", "string", 0)])])
        assert f[0, 12] == 1.0

    def test_no_request_body(self):
        f = extract_static_features([_ep()])
        assert f[0, 12] == 0.0

    def test_request_ct_json(self):
        f = extract_static_features([_ep(request_content_type="application/json")])
        assert list(f[0, 13:17]) == [1, 0, 0, 0]

    def test_input_complexity(self):
        f = extract_static_features([_ep(path="/users/{id}", query_params=[ParamInfo("page")],
            request_body_fields=[FieldInfo("address", "object", 0), FieldInfo("city", "string", 1)])])
        # path_params=1, query=1, header=0, body_top_level=1 → total=3, nesting=1 → 3*1=3
        assert f[0, 17] == 3.0

    def test_has_url_param(self):
        f = extract_static_features([_ep(query_params=[ParamInfo("redirect_url")])])
        assert f[0, 18] == 1.0

    def test_no_url_param(self):
        f = extract_static_features([_ep(query_params=[ParamInfo("page")])])
        assert f[0, 18] == 0.0


class TestAuthFeatures:
    def test_auth_type_none(self):
        assert list(extract_static_features([_ep(auth_type="none")])[0, 19:23]) == [1, 0, 0, 0]

    def test_auth_type_bearer(self):
        assert list(extract_static_features([_ep(auth_type="bearer-oauth")])[0, 19:23]) == [0, 0, 1, 0]

    def test_auth_required(self):
        assert extract_static_features([_ep(auth_required=True)])[0, 23] == 1.0

    def test_privilege_admin(self):
        assert extract_static_features([_ep(path="/admin/settings")])[0, 24] == 2.0

    def test_privilege_public(self):
        assert extract_static_features([_ep(path="/api/health", auth_required=False)])[0, 24] == 0.0

    def test_privilege_user(self):
        assert extract_static_features([_ep(path="/api/orders", auth_required=True)])[0, 24] == 1.0


class TestResponseFeatures:
    def test_response_ct_json(self):
        assert list(extract_static_features([_ep(response_content_type="application/json")])[0, 25:28]) == [1, 0, 0]

    def test_response_is_collection(self):
        assert extract_static_features([_ep(response_is_array=True)])[0, 28] == 1.0

    def test_has_pagination(self):
        f = extract_static_features([_ep(query_params=[ParamInfo("page"), ParamInfo("per_page")])])
        assert f[0, 29] == 1.0

    def test_has_sensitive_password(self):
        f = extract_static_features([_ep(request_body_fields=[FieldInfo("password", "string", 0)])])
        assert f[0, 30] == 1.0

    def test_sensitive_no_false_positive(self):
        """'tokenize' should NOT trigger sensitive indicator."""
        f = extract_static_features([_ep(query_params=[ParamInfo("tokenize")])])
        assert f[0, 30] == 0.0


class TestResourceHierarchyLevel:
    def test_api_prefix_only(self):
        assert extract_static_features([_ep(path="/api")])[0, 31] == 0.0

    def test_single_resource(self):
        assert extract_static_features([_ep(path="/api/v1/users")])[0, 31] == 1.0

    def test_resource_with_param(self):
        assert extract_static_features([_ep(path="/users/{id}")])[0, 31] == 2.0

    def test_sub_resource(self):
        assert extract_static_features([_ep(path="/users/{id}/posts")])[0, 31] == 3.0

    def test_deep_sub_resource_capped(self):
        assert extract_static_features([_ep(path="/users/{id}/posts/{pid}/comments")])[0, 31] == 3.0
