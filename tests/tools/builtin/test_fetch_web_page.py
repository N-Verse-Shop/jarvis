"""Tests for fetch web page tool."""

import pytest
from unittest.mock import Mock, patch
import requests

from src.jarvis.tools.builtin.fetch_web_page import FetchWebPageTool
from src.jarvis.tools.base import ToolContext
from src.jarvis.tools.types import ToolExecutionResult


def _make_response_mock(**attrs) -> Mock:
    """Build a Mock that doubles as both the requests response and a context
    manager (the production code uses ``with requests.get(...) as resp`` so
    the connection is released deterministically).

    Round 17: production code now uses ``stream=True`` + ``iter_content``
    to enforce a byte cap on the response body. Wire up sensible
    defaults so existing tests that supply ``content=`` still work:
    - ``iter_content`` yields the entire ``content`` in one chunk.
    - ``encoding`` / ``apparent_encoding`` default to utf-8 so the
      decode path succeeds.
    - ``headers`` defaults to an empty dict so ``Content-Length`` lookup
      is a clean None.
    """
    resp = Mock(**attrs)
    resp.__enter__ = Mock(return_value=resp)
    resp.__exit__ = Mock(return_value=False)
    if not isinstance(getattr(resp, "headers", None), dict):
        resp.headers = attrs.get("headers", {}) or {}
    body = attrs.get("content", b"")
    if not isinstance(body, (bytes, bytearray)):
        body = b""
    resp.iter_content = Mock(return_value=iter([body])) if body else Mock(return_value=iter([]))
    resp.encoding = attrs.get("encoding", "utf-8")
    resp.apparent_encoding = attrs.get("apparent_encoding", "utf-8")
    return resp


class TestFetchWebPageTool:
    """Test fetch web page tool functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = FetchWebPageTool()
        self.context = Mock(spec=ToolContext)
        self.context.user_print = Mock()

    def test_tool_properties(self):
        """Test tool metadata properties."""
        assert self.tool.name == "fetchWebPage"
        assert "fetch" in self.tool.description.lower()
        assert self.tool.inputSchema["type"] == "object"
        assert "url" in self.tool.inputSchema["required"]

    def test_run_no_args(self):
        """Test fetch web page with no arguments."""
        result = self.tool.run(None, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "url" in result.error_message.lower()

    def test_run_empty_url(self):
        """Test fetch web page with empty URL."""
        args = {"url": ""}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "url" in result.error_message.lower()

    @patch('requests.get')
    def test_run_success(self, mock_get):
        """Test successful web page fetch."""
        mock_response = _make_response_mock(
            status_code=200,
            text='<html><head><title>Test</title></head><body><p>Content</p></body></html>',
            content=b'<html><head><title>Test</title></head><body><p>Content</p></body></html>',
            headers={'content-type': 'text/html'},
            raise_for_status=Mock(),
        )
        mock_get.return_value = mock_response

        args = {"url": "https://example.com"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert "example.com" in result.reply_text
        self.context.user_print.assert_called()

    @patch('requests.get')
    def test_run_success_without_beautifulsoup(self, mock_get):
        """Test successful web page fetch without BeautifulSoup."""
        mock_response = _make_response_mock(
            status_code=200,
            text='<html><body>Raw content</body></html>',
            content=b'<html><body>Raw content</body></html>',
            headers={'content-type': 'text/html'},
            raise_for_status=Mock(),
        )
        mock_get.return_value = mock_response

        with patch('builtins.__import__', side_effect=ImportError):
            args = {"url": "https://example.com"}
            result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert "Raw Content" in result.reply_text

    @patch('requests.get')
    def test_run_http_error(self, mock_get):
        """Test fetch web page with HTTP error."""
        mock_response = _make_response_mock(status_code=404)
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        args = {"url": "https://example.com/notfound"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "Failed to fetch page" in result.error_message

    @patch('requests.get')
    def test_run_request_error(self, mock_get):
        """Test fetch web page with network error."""
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        args = {"url": "https://example.com"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "Failed to fetch page" in result.error_message

    def test_run_invalid_url(self):
        """Test fetch web page with invalid URL."""
        args = {"url": "not-a-url"}
        result = self.tool.run(args, self.context)
        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        # Round 15: types.py __post_init__ migrates failure messages
        # from reply_text into error_message. The SSRF guard rejects
        # "not-a-url" before any HTTP attempt with a "blocked" message
        # — verify any of the three semantically equivalent phrasings.
        lowered = (result.error_message or "").lower()
        assert any(kw in lowered for kw in ("failed", "error", "blocked", "invalid"))

    @patch('requests.get')
    def test_run_with_links_extraction(self, mock_get):
        """Test fetch web page including link extraction when include_links=True."""
        html = (
            '<html><head><title>Links Page</title></head>'
            '<body><p>Intro</p>'
            '<a href="/relative">Relative Link</a>'
            '<a href="https://absolute.test/page">Absolute Link</a>'
            '<a href="mailto:test@example.com">Mail</a>'
            '</body></html>'
        )
        mock_response = _make_response_mock(
            status_code=200,
            text=html,
            content=html.encode(),
            raise_for_status=Mock(),
        )
        mock_get.return_value = mock_response

        args = {"url": "https://example.com", "include_links": True}
        result = self.tool.run(args, self.context)
        assert result.success is True
        assert isinstance(result, ToolExecutionResult)
        assert "Links found on page" in result.reply_text
        # relative link should be resolved to absolute
        assert "https://example.com/relative" in result.reply_text
        assert "absolute.test" in result.reply_text
