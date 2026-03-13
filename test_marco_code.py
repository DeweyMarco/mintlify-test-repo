"""Tests for marco-code CLI tool."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from marco_code import (
    read,
    write,
    bash,
    call_model,
    main,
    _build_system_prompt,
    _get_client,
    TOOLS,
    MAX_ITERATIONS,
)


# =============================================================================
# Tool unit tests
# =============================================================================


class TestReadTool:
    def test_reads_file_contents(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        assert read(str(f)) == "hello world"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read("/nonexistent/path/file.txt")


class TestWriteTool:
    def test_writes_file(self, tmp_path):
        path = str(tmp_path / "output.txt")
        result = write(path, "content here")
        assert "Successfully wrote" in result
        assert open(path).read() == "content here"

    def test_creates_directories(self, tmp_path):
        path = str(tmp_path / "sub" / "dir" / "file.txt")
        write(path, "nested")
        assert open(path).read() == "nested"


class TestBashTool:
    def test_captures_stdout(self):
        result = bash("echo hello")
        assert "hello" in result

    def test_captures_stderr(self):
        result = bash("echo err >&2")
        assert "err" in result

    def test_nonzero_exit_does_not_raise(self):
        result = bash("exit 1")
        assert isinstance(result, str)


# =============================================================================
# System prompt
# =============================================================================


class TestSystemPrompt:
    def test_includes_cwd(self):
        prompt = _build_system_prompt()
        assert os.getcwd() in prompt

    def test_includes_platform(self):
        prompt = _build_system_prompt()
        assert "Platform:" in prompt


# =============================================================================
# Client singleton
# =============================================================================


class TestGetClient:
    def test_raises_without_api_key(self):
        import marco_code

        marco_code._client = None
        with patch.object(marco_code, "API_KEY", None):
            with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
                _get_client()
        marco_code._client = None


# =============================================================================
# Agent loop integration test (mocked API)
# =============================================================================


def _make_choice(content=None, tool_calls=None, finish_reason="stop"):
    """Helper to build a mock API response choice."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason
    return choice


def _make_tool_call(name, arguments, call_id="call_1"):
    """Helper to build a mock tool call."""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


class TestAgentLoop:
    @patch("marco_code._get_client")
    def test_simple_text_response(self, mock_get_client, capsys):
        """LLM responds with text immediately (no tool calls)."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        response = MagicMock()
        response.choices = [_make_choice(content="Hello!")]
        mock_client.chat.completions.create.return_value = response

        with patch("sys.argv", ["marco-code", "-p", "say hello"]):
            main()

        captured = capsys.readouterr()
        assert "Hello!" in captured.out

    @patch("marco_code._get_client")
    def test_tool_call_then_stop(self, mock_get_client, capsys, tmp_path):
        """LLM calls read tool, then responds with text."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create a file for the read tool to find
        test_file = tmp_path / "data.txt"
        test_file.write_text("file contents")

        # First response: tool call
        tool_call = _make_tool_call("read", {"file_path": str(test_file)})
        resp1 = MagicMock()
        resp1.choices = [_make_choice(tool_calls=[tool_call], finish_reason="tool_calls")]

        # Second response: text
        resp2 = MagicMock()
        resp2.choices = [_make_choice(content="I read the file.")]

        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        with patch("sys.argv", ["marco-code", "-p", "read data.txt"]):
            main()

        captured = capsys.readouterr()
        assert "I read the file." in captured.out

    @patch("marco_code._get_client")
    def test_unknown_tool_returns_error(self, mock_get_client, capsys):
        """LLM calls a non-existent tool — error returned as tool result, not crash."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        tool_call = _make_tool_call("nonexistent", {"x": 1})
        resp1 = MagicMock()
        resp1.choices = [_make_choice(tool_calls=[tool_call], finish_reason="tool_calls")]

        resp2 = MagicMock()
        resp2.choices = [_make_choice(content="Sorry, that tool failed.")]

        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        with patch("sys.argv", ["marco-code", "-p", "do something"]):
            main()

        captured = capsys.readouterr()
        assert "Sorry" in captured.out

    @patch("marco_code._get_client")
    def test_malformed_json_returns_error(self, mock_get_client, capsys):
        """LLM sends malformed JSON arguments — error returned, not crash."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        tool_call = MagicMock()
        tool_call.id = "call_bad"
        tool_call.function.name = "read"
        tool_call.function.arguments = "not valid json"

        resp1 = MagicMock()
        resp1.choices = [_make_choice(tool_calls=[tool_call], finish_reason="tool_calls")]

        resp2 = MagicMock()
        resp2.choices = [_make_choice(content="I'll try differently.")]

        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        with patch("sys.argv", ["marco-code", "-p", "test"]):
            main()

        captured = capsys.readouterr()
        assert "differently" in captured.out

    @patch("marco_code._get_client")
    def test_max_iterations_cap(self, mock_get_client, capsys):
        """Agent loop stops after MAX_ITERATIONS even if LLM keeps calling tools."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        tool_call = _make_tool_call("bash", {"command": "echo loop"})
        looping_resp = MagicMock()
        looping_resp.choices = [_make_choice(
            content="still going",
            tool_calls=[tool_call],
            finish_reason="tool_calls",
        )]

        mock_client.chat.completions.create.return_value = looping_resp

        with patch("sys.argv", ["marco-code", "-p", "loop forever"]):
            main()

        captured = capsys.readouterr()
        assert "maximum iterations" in captured.err
