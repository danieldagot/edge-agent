from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

from edge_agent.template import render_template


class TestRenderTemplate:
    def test_current_date_replaced(self):
        result = render_template("Today is {{currentDate}}.")
        expected_date = datetime.date.today().isoformat()
        assert result == f"Today is {expected_date}."

    def test_url_variable_fetched(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"fetched content"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch(
            "edge_agent.template.urllib.request.urlopen",
            return_value=mock_resp,
        ) as mock_urlopen:
            result = render_template("Data: {{url:https://example.com/data}}")

        mock_urlopen.assert_called_once_with(
            "https://example.com/data", timeout=10,
        )
        assert result == "Data: fetched content"

    def test_custom_variable_replaced(self):
        result = render_template(
            "Hello, {{name}}!",
            variables={"name": "Alice"},
        )
        assert result == "Hello, Alice!"

    def test_unknown_key_left_as_is(self):
        result = render_template("Value: {{unknown}}")
        assert result == "Value: {{unknown}}"

    def test_multiple_variables_in_one_template(self):
        template = (
            "Date: {{currentDate}}, User: {{userName}}, "
            "Unknown: {{missing}}"
        )
        result = render_template(template, variables={"userName": "Bob"})

        expected_date = datetime.date.today().isoformat()
        assert f"Date: {expected_date}" in result
        assert "User: Bob" in result
        assert "Unknown: {{missing}}" in result

    def test_no_placeholders_returns_unchanged(self):
        plain = "No placeholders here."
        assert render_template(plain) == plain

    def test_whitespace_around_key_is_stripped(self):
        result = render_template(
            "Hi {{ name }}!",
            variables={"name": "Eve"},
        )
        assert result == "Hi Eve!"

    def test_empty_variables_dict(self):
        result = render_template("{{key}}", variables={})
        assert result == "{{key}}"

    def test_none_variables_same_as_empty(self):
        result = render_template("{{key}}", variables=None)
        assert result == "{{key}}"
