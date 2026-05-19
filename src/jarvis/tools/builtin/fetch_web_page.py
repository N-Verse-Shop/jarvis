"""Fetch web page tool implementation for extracting content from URLs."""

import requests
from typing import Dict, Any, Optional
from ...debug import debug_log
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult
from .web_search import _is_public_url  # SSRF guard — see round 8 audit


# Audit round 17 fix: hard cap on bytes pulled from any single page.
# Without this, ``r.content`` reads the entire response body into RAM
# before any length check runs. A hostile or accidentally-misconfigured
# server can stream 10 GB before we'd notice — the 50 000-char output
# truncation happens AFTER full decode. 2 MiB is well above what any
# legitimate text page produces and leaves enough headroom for inline
# CSS / scripts to be stripped without truncating the actual content.
_MAX_FETCH_BYTES = 2 * 1024 * 1024  # 2 MiB


class FetchWebPageTool(Tool):
    """Tool for fetching and extracting content from web pages."""

    @property
    def name(self) -> str:
        return "fetchWebPage"

    @property
    def description(self) -> str:
        return "Fetch and extract text content from a web page URL."

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch content from"},
                "include_links": {"type": "boolean", "description": "Whether to include links found on the page"}
            },
            "required": ["url"]
        }

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        """Fetch and extract content from a web page.

        Audit round 11 fix M3: every failure path now passes ``error_message=``
        rather than ``reply_text=``. The engine renders tool-error messages
        via ``result.error_message or "(no result)"`` (see engine.py); when
        we put the reason in ``reply_text`` instead, the agent saw
        ``"(no result)"`` and either gave up or hallucinated content.
        Now the actual failure ("Blocked: private URL", "Too many
        redirects") reaches the LLM so it can decide to retry differently
        or report honestly to the user.
        """
        context.user_print("🌐 Fetching page content…")
        try:
            if not (args and isinstance(args, dict)):
                return ToolExecutionResult(success=False, reply_text=None, error_message="fetchWebPage requires a JSON object with 'url'.")
            url = str(args.get("url", "")).strip()
            include_links = bool(args.get("include_links", False))
            if not url:
                return ToolExecutionResult(success=False, reply_text=None, error_message="fetchWebPage requires a valid 'url'.")
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            # Audit round 8 SECURITY fix S3: SSRF guard. Without this,
            # an LLM (or compromised search result) could request
            # `http://169.254.169.254/latest/meta-data/iam/security-credentials/`
            # (AWS metadata), `http://127.0.0.1:11434/api/show` (local
            # Ollama), or `http://[::1]/admin` etc. and exfiltrate
            # credentials / probe internal services. Reuses the
            # `_is_public_url` validator from `web_search.py` which
            # already covers IPv4/IPv6 private, loopback, link-local,
            # CGNAT (Tailscale 100.64/10), .local mDNS, etc.
            if not _is_public_url(url):
                debug_log(f"fetchWebPage: blocked non-public URL: {url}", "web")
                return ToolExecutionResult(
                    success=False,
                    reply_text=None,
                    error_message=(
                        "Blocked: that URL points at a private/internal "
                        "address. Only public HTTPS URLs are allowed."
                    ),
                )
            debug_log(f"fetchWebPage: fetching {url}", "web")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            # ``with`` releases the connection back to the pool deterministically
            # even if BeautifulSoup or the link extraction raises midway.
            # SSRF defence in depth: walk redirects manually, re-validating each hop.
            current_url = url
            response = None
            for _hop in range(5):  # max 5 redirects
                # Audit round 17 fix: ``stream=True`` keeps the body off
                # the wire until we iterate it; the chunked read below
                # bails out the moment ``_MAX_FETCH_BYTES`` is exceeded,
                # so a hostile gigabyte payload never reaches RAM.
                with requests.get(current_url, headers=headers, timeout=15,
                                  allow_redirects=False, stream=True) as r:
                    if r.status_code in (301, 302, 303, 307, 308):
                        nxt = r.headers.get("Location", "")
                        if not nxt:
                            break
                        if nxt.startswith("/"):
                            from urllib.parse import urljoin
                            nxt = urljoin(current_url, nxt)
                        if not _is_public_url(nxt):
                            debug_log(f"fetchWebPage: blocked redirect to non-public URL: {nxt}", "web")
                            return ToolExecutionResult(
                                success=False,
                                reply_text=None,
                                error_message=(
                                    "Blocked: server tried to redirect to a "
                                    "private/internal address."
                                ),
                            )
                        current_url = nxt
                        continue
                    r.raise_for_status()
                    # Pre-check Content-Length when the server provides
                    # one — short-circuits the chunked read for the
                    # obvious-bad case without setting up any buffers.
                    declared = r.headers.get("Content-Length")
                    try:
                        if declared and int(declared) > _MAX_FETCH_BYTES:
                            return ToolExecutionResult(
                                success=False,
                                reply_text=None,
                                error_message=(
                                    f"Refused: server reports "
                                    f"{int(declared):,} bytes "
                                    f"(cap is {_MAX_FETCH_BYTES:,})."
                                ),
                            )
                    except ValueError:
                        pass
                    chunks: list[bytes] = []
                    fetched = 0
                    truncated = False
                    for chunk in r.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        chunks.append(chunk)
                        fetched += len(chunk)
                        if fetched >= _MAX_FETCH_BYTES:
                            truncated = True
                            debug_log(
                                f"fetchWebPage: truncated at "
                                f"{_MAX_FETCH_BYTES} bytes (server "
                                f"sent more)",
                                "web",
                            )
                            break
                    response_content = b"".join(chunks)
                    # Decode using the server's declared charset where
                    # possible; ``requests`` exposes the negotiated
                    # encoding via ``r.encoding`` even on streamed
                    # responses once headers are parsed.
                    enc = r.encoding or r.apparent_encoding or "utf-8"
                    try:
                        response_text = response_content.decode(enc, errors="replace")
                    except (LookupError, TypeError):
                        response_text = response_content.decode("utf-8", errors="replace")
                    response = r
                    if truncated:
                        # Surface the truncation to the model so it can
                        # decide whether to re-fetch with a more
                        # specific URL.
                        response_text = (
                            "[fetchWebPage truncated upstream body to "
                            f"{_MAX_FETCH_BYTES} bytes — partial content "
                            "below]\n" + response_text
                        )
                    break
            else:
                return ToolExecutionResult(success=False, reply_text=None, error_message="Too many redirects.")
            if response is None:
                return ToolExecutionResult(success=False, reply_text=None, error_message="No response from server.")
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response_content, 'html.parser')
                for script in soup(["script", "style", "meta", "link", "noscript"]):
                    script.decompose()
                title = ""
                title_tag = soup.find('title')
                if title_tag:
                    title = title_tag.get_text().strip()
                text_content = soup.get_text()
                lines = []
                for line in text_content.split('\n'):
                    cleaned_line = line.strip()
                    if cleaned_line and len(cleaned_line) > 3:
                        lines.append(cleaned_line)
                seen_lines = set()
                unique_lines = []
                for line in lines:
                    if line not in seen_lines:
                        unique_lines.append(line)
                        seen_lines.add(line)
                content = '\n'.join(unique_lines[:500])
                links_section = ""
                if include_links:
                    links = []
                    for link in soup.find_all('a', href=True):
                        href = link.get('href', '').strip()
                        link_text = link.get_text().strip()
                        if href and link_text and len(link_text) > 3:
                            if href.startswith('/'):
                                from urllib.parse import urljoin
                                href = urljoin(url, href)
                            elif not href.startswith(('http://', 'https://', 'mailto:', 'tel:')):
                                continue
                            links.append(f"• {link_text}: {href}")
                    if links:
                        links_section = f"\n\n**Links found on page:**\n" + '\n'.join(links[:20])
                reply_parts = []
                if title:
                    reply_parts.append(f"**Title:** {title}")
                reply_parts.append(f"**URL:** {url}")
                reply_parts.append(f"**Content:**\n{content}")
                if links_section:
                    reply_parts.append(links_section)
                reply_text = '\n\n'.join(reply_parts)
                max_chars = 50_000
                if len(reply_text) > max_chars:
                    reply_text = f"[Truncated to {max_chars} chars]\n\n" + reply_text[:max_chars]
                debug_log(f"fetchWebPage: extracted {len(content)} chars of content", "web")
                context.user_print("✅ Page content fetched.")
                return ToolExecutionResult(success=True, reply_text=reply_text)
            except ImportError:
                text = response_text[:10000]
                reply_text = f"**URL:** {url}\n**Raw Content:**\n{text}"
                debug_log("fetchWebPage: BeautifulSoup not available, returning raw text", "web")
                context.user_print("✅ Page content fetched (raw).")
                return ToolExecutionResult(success=True, reply_text=reply_text)
        except requests.exceptions.RequestException as e:
            debug_log(f"fetchWebPage: request failed: {e}", "web")
            context.user_print("⚠️ Failed to fetch page.")
            return ToolExecutionResult(success=False, reply_text=None, error_message=f"Failed to fetch page: {e}")
        except Exception as e:  # pragma: no cover (safety net)
            debug_log(f"fetchWebPage: error: {e}", "web")
            context.user_print("⚠️ Error fetching page.")
            return ToolExecutionResult(success=False, reply_text=None, error_message=f"Error fetching page: {e}")
