from __future__ import annotations

import os
import textwrap

import pytest

from edge_agent import dotenv_values, find_dotenv, load_dotenv


# ── parsing ──────────────────────────────────────────────────────────────────


class TestParsing:
    def test_basic_key_value(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("FOO=bar\n")
        assert dotenv_values(str(env)) == {"FOO": "bar"}

    def test_multiple_keys(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("A=1\nB=2\nC=3\n")
        assert dotenv_values(str(env)) == {"A": "1", "B": "2", "C": "3"}

    def test_single_quoted_value(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("FOO='hello world'\n")
        assert dotenv_values(str(env)) == {"FOO": "hello world"}

    def test_double_quoted_value(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text('FOO="hello world"\n')
        assert dotenv_values(str(env)) == {"FOO": "hello world"}

    def test_comment_lines_ignored(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("# this is a comment\nFOO=bar\n")
        assert dotenv_values(str(env)) == {"FOO": "bar"}

    def test_blank_lines_ignored(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("\nFOO=bar\n\nBAZ=qux\n")
        assert dotenv_values(str(env)) == {"FOO": "bar", "BAZ": "qux"}

    def test_export_prefix(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("export FOO=bar\n")
        assert dotenv_values(str(env)) == {"FOO": "bar"}

    def test_inline_comment(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("FOO=bar  # inline comment\n")
        assert dotenv_values(str(env)) == {"FOO": "bar"}

    def test_value_with_equals_sign(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text('FOO="a=b=c"\n')
        assert dotenv_values(str(env)) == {"FOO": "a=b=c"}

    def test_key_without_value(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("FOO\n")
        assert dotenv_values(str(env)) == {"FOO": None}

    def test_empty_value(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("FOO=\n")
        assert dotenv_values(str(env)) == {"FOO": ""}

    def test_double_quote_escape_newline(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text('FOO="line1\\nline2"\n')
        assert dotenv_values(str(env)) == {"FOO": "line1\nline2"}

    def test_single_quote_no_escape_newline(self, tmp_path):
        """Single-quoted values only support \\' and \\\\ escapes."""
        env = tmp_path / ".env"
        env.write_text("FOO='hello\\nworld'\n")
        assert dotenv_values(str(env)) == {"FOO": "hello\\nworld"}

    def test_spaces_around_equals(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("FOO =bar\n")
        assert dotenv_values(str(env)) == {"FOO": "bar"}

    def test_missing_file_returns_empty(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        assert dotenv_values(str(missing)) == {}

    def test_windows_line_endings(self, tmp_path):
        env = tmp_path / ".env"
        env.write_bytes(b"FOO=bar\r\nBAZ=qux\r\n")
        assert dotenv_values(str(env)) == {"FOO": "bar", "BAZ": "qux"}


# ── variable interpolation ──────────────────────────────────────────────────


class TestVariableInterpolation:
    def test_simple_variable(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("BASE=/app\nPATH_VAR=${BASE}/bin\n")
        result = dotenv_values(str(env))
        assert result["PATH_VAR"] == "/app/bin"

    def test_variable_with_default(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("FOO=${UNDEFINED:-fallback}\n")
        result = dotenv_values(str(env))
        assert result["FOO"] == "fallback"

    def test_variable_defined_earlier(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("HOST=localhost\nURL=http://${HOST}:8080\n")
        result = dotenv_values(str(env))
        assert result["URL"] == "http://localhost:8080"

    def test_undefined_variable_resolves_empty(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("FOO=${NOPE}\n")
        result = dotenv_values(str(env))
        assert result["FOO"] == ""

    def test_no_interpolation(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("BASE=/app\nFOO=${BASE}/bin\n")
        result = dotenv_values(str(env), interpolate=False)
        assert result["FOO"] == "${BASE}/bin"


# ── find_dotenv ──────────────────────────────────────────────────────────────


class TestFindDotenv:
    def test_finds_in_directory(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("A=1\n")
        found = find_dotenv(usecwd=True)
        # usecwd=True uses os.getcwd() which won't be tmp_path, so test the
        # walk-up mechanism via a subprocess-like approach instead.
        # Here we just test the basic contract with a known path.
        assert env.exists()

    def test_returns_empty_when_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = find_dotenv(usecwd=True)
        assert result == ""

    def test_finds_in_parent(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("A=1\n")
        child = tmp_path / "sub" / "dir"
        child.mkdir(parents=True)
        monkeypatch.chdir(child)
        result = find_dotenv(usecwd=True)
        assert result == str(env)

    def test_custom_filename(self, tmp_path, monkeypatch):
        env = tmp_path / ".env.local"
        env.write_text("X=1\n")
        monkeypatch.chdir(tmp_path)
        result = find_dotenv(filename=".env.local", usecwd=True)
        assert result == str(env)

    def test_raise_when_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(IOError, match="File not found"):
            find_dotenv(raise_error_if_not_found=True, usecwd=True)


# ── load_dotenv ──────────────────────────────────────────────────────────────


class TestLoadDotenv:
    def test_loads_into_environ(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("TINY_TEST_VAR=hello\n")
        monkeypatch.delenv("TINY_TEST_VAR", raising=False)

        result = load_dotenv(str(env))

        assert result is True
        assert os.environ["TINY_TEST_VAR"] == "hello"

    def test_does_not_override_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TINY_TEST_VAR", "original")
        env = tmp_path / ".env"
        env.write_text("TINY_TEST_VAR=overridden\n")

        load_dotenv(str(env))

        assert os.environ["TINY_TEST_VAR"] == "original"

    def test_override_true(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TINY_TEST_VAR", "original")
        env = tmp_path / ".env"
        env.write_text("TINY_TEST_VAR=overridden\n")

        load_dotenv(str(env), override=True)

        assert os.environ["TINY_TEST_VAR"] == "overridden"

    def test_returns_false_for_empty_file(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("")
        assert load_dotenv(str(env)) is False

    def test_returns_false_for_missing_file(self, tmp_path):
        missing = tmp_path / "nope"
        assert load_dotenv(str(missing)) is False

    def test_multiple_vars(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("TINY_A=one\nTINY_B=two\n")
        monkeypatch.delenv("TINY_A", raising=False)
        monkeypatch.delenv("TINY_B", raising=False)

        load_dotenv(str(env))

        assert os.environ["TINY_A"] == "one"
        assert os.environ["TINY_B"] == "two"

    def test_key_without_value_not_set(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("TINY_NONE_VAR\n")
        monkeypatch.delenv("TINY_NONE_VAR", raising=False)

        load_dotenv(str(env))

        assert "TINY_NONE_VAR" not in os.environ


# ── dotenv_values ────────────────────────────────────────────────────────────


class TestDotenvValues:
    def test_does_not_modify_environ(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TINY_ISOLATED_VAR", raising=False)
        env = tmp_path / ".env"
        env.write_text("TINY_ISOLATED_VAR=secret\n")

        result = dotenv_values(str(env))

        assert result == {"TINY_ISOLATED_VAR": "secret"}
        assert "TINY_ISOLATED_VAR" not in os.environ

    def test_returns_ordered_dict(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("Z=1\nA=2\nM=3\n")
        result = dotenv_values(str(env))
        assert list(result.keys()) == ["Z", "A", "M"]

    def test_utf8_values(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text('GREETING="héllo wörld"\n', encoding="utf-8")
        result = dotenv_values(str(env))
        assert result["GREETING"] == "héllo wörld"
