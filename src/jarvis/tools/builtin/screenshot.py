"""Screenshot tool implementation for OCR capture."""

from typing import Dict, Any, Optional
import os
import tempfile
import subprocess
import shutil
from ...debug import debug_log
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult

class ScreenshotTool(Tool):
    """Tool for capturing screenshots and performing OCR."""

    @property
    def name(self) -> str:
        return "screenshot"

    @property
    def description(self) -> str:
        return "Capture a selected screen region and OCR the text. Use only if the OCR will materially help."

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": []
        }

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        """Execute the screenshot tool."""
        context.user_print("📸 Capturing a screenshot for OCR…")
        debug_log("screenshot: capturing OCR...", "screenshot")
        # Inline OCR capture logic (previously in separate helper)
        ocr_text: str = ""
        sc = shutil.which("screencapture")
        if sc:
            # Audit round 21 fix (F20): use a user-private cache
            # directory instead of /tmp. The PNG of the user's
            # selected screen region (which may contain passwords,
            # private chats, secrets visible on-screen) is no longer
            # world-readable. We also chmod 0o600 the PNG immediately
            # after screencapture writes it.
            base_cache = os.path.expanduser("~/.cache/jarvis/ocr")
            try:
                os.makedirs(base_cache, exist_ok=True)
                try:
                    os.chmod(base_cache, 0o700)
                except OSError:
                    pass
                tmpdir = tempfile.mkdtemp(prefix="jarvis_ocr_", dir=base_cache)
            except OSError:
                # Sandboxed test fallback — still better than failing.
                tmpdir = tempfile.mkdtemp(prefix="jarvis_ocr_")
            png_path = os.path.join(tmpdir, "shot.png")
            try:
                cmd = [sc, "-i", png_path]
                # Audit round 10 fix: `screencapture -i` waits for the
                # user to either select a region or hit Escape. If the
                # user does NEITHER, the process waits forever and
                # blocks the whole tool-call turn. Cap at 60s — plenty
                # of time for any deliberate selection, immediate
                # release if the user walked away.
                try:
                    ret = subprocess.run(cmd, timeout=60)
                except subprocess.TimeoutExpired:
                    debug_log("screenshot: screencapture timed out (60s)", "screenshot")
                    ret = None  # type: ignore
                except Exception:
                    ret = None  # type: ignore
                if ret and getattr(ret, "returncode", 1) == 0 and os.path.exists(png_path):
                    # Audit round 21 (F20): close the read-window
                    # for any other process running as the same or
                    # different uid by tightening the PNG mode.
                    try:
                        os.chmod(png_path, 0o600)
                    except OSError:
                        pass
                    tess = shutil.which("tesseract")
                    if tess:
                        try:
                            import pytesseract  # type: ignore
                            from PIL import Image  # type: ignore
                            with Image.open(png_path) as im:
                                text = pytesseract.image_to_string(im)
                                if text and text.strip():
                                    ocr_text = text.strip()
                        except Exception:
                            pass
            finally:
                try:
                    if os.path.exists(png_path):
                        os.remove(png_path)
                    os.rmdir(tmpdir)
                except Exception:
                    pass
        debug_log(f"screenshot: ocr_chars={len(ocr_text)}", "screenshot")
        # Audit round 13 fix: previously returned ``success=True,
        # reply_text=""`` when the user hit Escape, screencapture
        # timed out, or OCR produced no text. That violated the
        # ToolExecutionResult contract (success=True requires
        # non-empty reply_text). Engine then hit the
        # `not result.user_text` branch and reported "(no result)" to
        # the agent — a confusing failure mode for a user-initiated
        # cancel. Now: distinguish empty OCR from a real read.
        if not ocr_text:
            context.user_print("⚠️ Screenshot cancelled or OCR produced no text.")
            return ToolExecutionResult(
                success=False,
                reply_text=None,
                error_message="Screenshot was cancelled or returned no readable text.",
            )
        context.user_print("✅ Screenshot processed.")
        # Return raw OCR text as tool result (no LLM processing here)
        return ToolExecutionResult(success=True, reply_text=ocr_text)
