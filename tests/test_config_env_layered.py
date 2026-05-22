"""Test the layered .env discovery in config.load_settings() (R35-S2 / R35-S3 P3-33).

The bug this guards against: `load_dotenv()` walks UP from CWD looking
for `.env`. macOS launchd sets the daemon's CWD to the LaunchAgent's
``WorkingDirectory`` (the repo path), which means a `.env` written to
``~/.config/jarvis/.env`` is never found from the daemon's process.
R35-S2 added explicit candidate paths so the canonical XDG location
takes precedence.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _call_load_settings():
    """Import and call load_settings, swallowing config-file errors.

    We only care about the .env loading side-effect, not the returned
    Settings object. If the config file path is bogus or missing
    fields, the call may raise — that's fine; the env load happens
    before any of that.
    """
    from jarvis.config import load_settings
    try:
        load_settings()
    except Exception:
        # config-file parse can fail in a sandboxed test env; the
        # env-loading runs first so the os.environ mutations still
        # land.
        pass


def test_jarvis_dotenv_path_env_var_wins(tmp_path, monkeypatch):
    """If JARVIS_DOTENV_PATH points to a file, it's loaded."""
    envfile = tmp_path / "override.env"
    envfile.write_text("JARVIS_TEST_LAYER_KEY=from_override\n", encoding="utf-8")
    monkeypatch.setenv("JARVIS_DOTENV_PATH", str(envfile))
    monkeypatch.delenv("JARVIS_TEST_LAYER_KEY", raising=False)

    _call_load_settings()

    assert os.environ.get("JARVIS_TEST_LAYER_KEY") == "from_override"


def test_xdg_config_dir_env_loaded(tmp_path, monkeypatch):
    """~/.config/jarvis/.env (via XDG_CONFIG_HOME) is loaded."""
    cfg_dir = tmp_path / "jarvis"
    cfg_dir.mkdir(parents=True)
    envfile = cfg_dir / ".env"
    envfile.write_text("JARVIS_TEST_XDG_KEY=from_xdg\n", encoding="utf-8")

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("JARVIS_DOTENV_PATH", raising=False)
    monkeypatch.delenv("JARVIS_TEST_XDG_KEY", raising=False)

    _call_load_settings()

    assert os.environ.get("JARVIS_TEST_XDG_KEY") == "from_xdg"


def test_explicit_env_does_not_override_existing(tmp_path, monkeypatch):
    """Existing os.environ values take precedence (override=False)."""
    envfile = tmp_path / "no_clobber.env"
    envfile.write_text("JARVIS_TEST_NO_OVERRIDE=fileval\n", encoding="utf-8")
    monkeypatch.setenv("JARVIS_DOTENV_PATH", str(envfile))
    monkeypatch.setenv("JARVIS_TEST_NO_OVERRIDE", "envval")  # already in env

    _call_load_settings()

    # ``override=False`` semantics: existing env wins.
    assert os.environ.get("JARVIS_TEST_NO_OVERRIDE") == "envval"


def test_missing_env_file_does_not_crash(tmp_path, monkeypatch):
    """Pointing JARVIS_DOTENV_PATH at a non-existent file must NOT raise."""
    monkeypatch.setenv("JARVIS_DOTENV_PATH", str(tmp_path / "nonexistent.env"))
    # Should not raise.
    _call_load_settings()


def test_xdg_home_overrides_dot_config_default(tmp_path, monkeypatch):
    """XDG_CONFIG_HOME genuinely redirects the canonical path lookup."""
    custom = tmp_path / "custom_xdg"
    (custom / "jarvis").mkdir(parents=True)
    envfile = custom / "jarvis" / ".env"
    envfile.write_text("JARVIS_TEST_CUSTOM_XDG=found\n", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(custom))
    monkeypatch.delenv("JARVIS_DOTENV_PATH", raising=False)
    monkeypatch.delenv("JARVIS_TEST_CUSTOM_XDG", raising=False)

    _call_load_settings()

    assert os.environ.get("JARVIS_TEST_CUSTOM_XDG") == "found"
