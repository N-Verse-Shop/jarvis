"""Cross-platform launcher for Claude Code preview_start.

Detects the OS and delegates to the appropriate platform-specific script
(bat on Windows, sh on macOS/Linux). Can be invoked with any Python 3.x.

Usage:
    python scripts/launch.py <script_name> [args...]

Examples:
    python scripts/launch.py run_desktop_app
    python scripts/launch.py run_desktop_app --voice-debug
    python scripts/launch.py run_evals
"""

import os
import platform
import re
import subprocess
import sys


# Audit round 19 SECURITY fix: ``script_name`` came straight from
# ``sys.argv[1]`` and was concatenated into a shell-resolved path. On
# Windows ``subprocess.run(..., shell=True)`` then evaluated the
# resulting string through ``cmd.exe`` — a value like
# ``run_desktop_app & calc.exe`` (or, worse, a piped payload from an
# automation script) would execute arbitrary commands. The fix:
#   1. Restrict ``script_name`` to a strict ``[A-Za-z0-9_-]+`` allowlist
#      that cannot contain shell metacharacters, separators, or
#      directory traversal.
#   2. Pass the resolved script path as the FIRST argv element rather
#      than running through the shell. On Windows ``.bat`` files are
#      invocable via ``cmd /c`` without ``shell=True``.
_SCRIPT_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/launch.py <script_name> [args...]")
        sys.exit(1)

    script_name = sys.argv[1]
    if not _SCRIPT_NAME_RE.match(script_name):
        print(
            f"ERROR: script name {script_name!r} contains disallowed "
            "characters; only [A-Za-z0-9_-] are permitted.",
            file=sys.stderr,
        )
        sys.exit(2)
    extra_args = sys.argv[2:]

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scripts_dir = os.path.join(project_root, "scripts")

    if platform.system() == "Windows":
        script_path = os.path.join(scripts_dir, f"{script_name}.bat")
        if not os.path.isfile(script_path):
            print(f"ERROR: {script_path} not found")
            sys.exit(1)
        # Invoke the .bat via cmd /c WITHOUT shell=True so the
        # individual argv elements are passed as a parsed argument
        # array rather than re-joined into a shell string. The
        # allowlist above is still the primary defence — this is
        # defence-in-depth.
        result = subprocess.run(
            ["cmd", "/c", script_path] + extra_args,
            cwd=project_root,
        )
    else:
        script_path = os.path.join(scripts_dir, f"{script_name}.sh")
        if not os.path.isfile(script_path):
            print(f"ERROR: {script_path} not found")
            sys.exit(1)
        result = subprocess.run(
            ["bash", script_path] + extra_args,
            cwd=project_root,
        )

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
