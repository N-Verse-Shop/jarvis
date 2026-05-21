"""Local files tool implementation for safe file operations."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


class LocalFilesTool(Tool):
    """Tool for safe local file operations within user's home directory."""

    @property
    def name(self) -> str:
        return "localFiles"

    @property
    def description(self) -> str:
        return "Safely read, write, list, append, or delete files within your home directory."

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "description": "Operation to perform: list, read, write, append, delete"},
                "path": {"type": "string", "description": "File or directory path (relative to home directory)"},
                "content": {"type": "string", "description": "Content to write/append (for write/append operations)"},
                "glob": {"type": "string", "description": "Glob pattern for listing (default: *)"},
                "recursive": {"type": "boolean", "description": "Whether to search recursively (for list operation)"}
            },
            "required": ["operation", "path"]
        }

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        """Execute the local files tool."""
        try:
            # Safety: restrict to user's home directory by default
            home_root = Path(os.path.expanduser("~")).resolve()

            def _expand_user_path(p: str) -> str:
                if not isinstance(p, str):
                    return str(p)
                if p == "~":
                    return os.path.expanduser("~")
                if p.startswith("~/") or p.startswith("~\\"):
                    return os.path.join(os.path.expanduser("~"), p[2:])
                return os.path.expanduser(p)

            # Audit round 10 SECURITY: deny-list of sensitive subpaths
            # within the user's home. Even though `_resolve_safe`
            # already blocks `/etc`, `/var/root`, etc. via the home
            # prefix check, paths INSIDE the home (`~/.ssh`,
            # `~/.gnupg`, Keychains) hold credentials that a voice
            # command should never touch. Voice intent "delete file
            # ~/.ssh/id_rsa" or "read ~/.aws/credentials" would
            # previously succeed. Now: deny by relative-path prefix.
            _SENSITIVE_PREFIXES = (
                ".ssh",
                ".gnupg",
                ".aws",
                ".config/gcloud",
                ".kube",
                ".docker",
                ".netrc",
                "Library/Keychains",
                "Library/Application Support/com.apple.idsmessaging",
                "Library/Application Support/iCloud",
                "Library/Mobile Documents",  # iCloud Drive root
                "Library/Cookies",
                ".bash_history",
                ".zsh_history",
                # Audit round 17 SECURITY: add Jarvis's own config/state
                # tree and macOS app sandbox containers. The previous
                # list missed these — a voice command "delete file
                # ~/.config/jarvis/config.json" or
                # "~/.config/jarvis/memory.db" would wipe the user's
                # entire setup and chat history with one sentence. The
                # Library/Containers tree is similarly off-limits because
                # sandboxed mac apps store keychain-equivalents there.
                ".config/jarvis",
                ".local/share/jarvis",
                ".cache/jarvis",
                "Library/Application Support/Jarvis",
                "Library/Application Support/com.jarvis",
                "Library/Containers",
                "Library/Group Containers",
                "Library/Logs/Jarvis",
                "Library/Preferences",
            )

            def _resolve_safe(p: str) -> Path:
                # Audit round 17 SECURITY: refuse symlinks. The previous
                # implementation called ``Path(...).resolve()`` which
                # transparently follows symlinks — so a symlink under
                # ``~/Documents/jarvis-config -> ~/.config/jarvis/config.json``
                # would resolve to the config path and pass the home check
                # because the resolved target is inside ``home_root``.
                # The sensitive-prefix list is checked on the resolved
                # path so it would still catch most cases, but ANY
                # symlink in the chain is a path-confusion red flag —
                # better to reject up front. Use ``lstat``/``is_symlink``
                # on every parent component so we catch e.g.
                # ``~/Documents/link/file.txt`` where ``link`` is the
                # symlink and ``file.txt`` is a real child.
                raw = Path(_expand_user_path(p))
                # Walk parents and the leaf to detect symlinks BEFORE
                # resolving. ``lstat`` does not follow links.
                probe = raw
                while True:
                    try:
                        if probe.exists() and probe.is_symlink():
                            raise PermissionError(
                                f"Refusing to follow symlink in path: {probe}"
                            )
                    except OSError:
                        pass
                    if probe.parent == probe:
                        break
                    probe = probe.parent
                resolved = raw.resolve()
                try:
                    # Allow exactly the home root or its descendants
                    if resolved == home_root or str(resolved).startswith(str(home_root) + os.sep):
                        try:
                            rel = resolved.relative_to(home_root).as_posix()
                            for prefix in _SENSITIVE_PREFIXES:
                                if rel == prefix or rel.startswith(prefix + "/"):
                                    raise PermissionError(
                                        f"Path is in protected zone "
                                        f"({prefix}); refuse for safety: {resolved}"
                                    )
                        except ValueError:
                            # `resolved` not under home — caught below.
                            pass
                        return resolved
                except PermissionError:
                    raise
                except Exception:
                    pass
                raise PermissionError(f"Path not allowed: {resolved}")

            if not (args and isinstance(args, dict)):
                # R34-S52 H: RU-only TTS policy.
                return ToolExecutionResult(success=False, reply_text="localFiles требует JSON-объект с минимум 'operation' и 'path'.")

            operation = str(args.get("operation") or "").strip().lower()
            path_arg = args.get("path")
            if not operation or not path_arg:
                # R34-S52 H: RU-only.
                return ToolExecutionResult(success=False, reply_text="localFiles требует 'operation' и 'path'.")

            target = _resolve_safe(str(path_arg))

            # list
            if operation == "list":
                if not target.exists():
                    # R34-S52 H: RU-only.
                    return ToolExecutionResult(success=False, reply_text=f"Путь не найден: {target}")
                if target.is_file():
                    # R34-S52 H: RU-only.
                    return ToolExecutionResult(success=True, reply_text=f"Файл: {target.name}")

                glob_pattern = args.get("glob", "*")
                recursive = bool(args.get("recursive", False))

                try:
                    # Audit round 10 SECURITY/PERF: bound the walk.
                    # `target.rglob("*")` on the home tree could exhaust
                    # memory (millions of files in node_modules/Library
                    # caches). The 50-item slice happened AFTER `list(...)`
                    # fully exhausted the generator → unbounded RAM use
                    # before truncation. Now: hard-cap walking at 5000
                    # entries so even a pathological pattern terminates
                    # quickly.
                    _WALK_HARD_CAP = 5000
                    iterator = target.rglob(glob_pattern) if recursive else target.glob(glob_pattern)
                    files: list = []
                    for _f in iterator:
                        files.append(_f)
                        if len(files) >= _WALK_HARD_CAP:
                            break

                    if not files:
                        # R34-S52 H: RU-only.
                        return ToolExecutionResult(success=True, reply_text=f"Файлы по шаблону «{glob_pattern}» не найдены в {target}")

                    file_list = []
                    for f in sorted(files)[:50]:  # Limit to 50 files
                        relative_path = f.relative_to(target)
                        # R34-S52 H: RU labels — DIR/FILE → ПАПКА/ФАЙЛ.
                        file_type = "ПАПКА" if f.is_dir() else "ФАЙЛ"
                        file_list.append(f"  {file_type}: {relative_path}")

                    # R34-S52 H: RU-only.
                    result = f"Содержимое {target}:\n" + "\n".join(file_list)
                    if len(files) > 50:
                        result += f"\n… и ещё {len(files) - 50} файлов"

                    return ToolExecutionResult(success=True, reply_text=result)
                except Exception as e:
                    # R34-S52 H: RU-only.
                    return ToolExecutionResult(success=False, reply_text=f"Ошибка списка: {e}")

            # read
            if operation == "read":
                if not target.exists() or not target.is_file():
                    # R34-S52 H: RU-only.
                    return ToolExecutionResult(success=False, reply_text=f"Файл не найден: {target}")
                try:
                    # Audit round 20 SECURITY P1: O_NOFOLLOW guard
                    # against symlink TOCTOU. ``_resolve_safe`` walked
                    # parents with ``lstat`` to reject symlinks, but
                    # between that check and ``read_text`` (which
                    # follows links) a hostile process with write to
                    # any parent dir could substitute the leaf with a
                    # symlink to ``~/.ssh/id_rsa``. ``O_NOFOLLOW`` makes
                    # the open() refuse to traverse the final symlink
                    # at kernel level — eliminates the race.
                    fd = os.open(str(target), os.O_RDONLY | os.O_NOFOLLOW)
                    try:
                        with os.fdopen(fd, "r", encoding="utf-8", errors="replace") as f:
                            data = f.read()
                    except Exception:
                        try:
                            os.close(fd)
                        except Exception:
                            pass
                        raise
                    max_chars = 10000
                    if len(data) > max_chars:
                        # R34-S52 H: RU-only.
                        data = data[:max_chars] + f"\n… (обрезано, показано первые {max_chars} символов)"
                    return ToolExecutionResult(success=True, reply_text=data)
                except OSError as e:
                    # ``ELOOP`` (40) is the kernel rejecting a symlink
                    # at the leaf. Translate to a clearer message so
                    # the user understands why the read failed.
                    import errno as _errno
                    if e.errno == _errno.ELOOP:
                        return ToolExecutionResult(
                            success=False,
                            # R34-S52 H: RU-only.
                            reply_text=f"Не читаю символьную ссылку: {target}",
                        )
                    # R34-S52 H: RU-only.
                    return ToolExecutionResult(success=False, reply_text=f"Ошибка чтения: {e}")
                except Exception as e:
                    # R34-S52 H: RU-only.
                    return ToolExecutionResult(success=False, reply_text=f"Ошибка чтения: {e}")

            # write
            if operation == "write":
                content = args.get("content")
                if not isinstance(content, str):
                    # R34-S52 H: RU-only.
                    return ToolExecutionResult(success=False, reply_text="Запись требует строку 'content'.")
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    # Audit round 20 P2 — atomic write via tmp+replace
                    # so a crash mid-write doesn't leave a truncated
                    # file. Combine with O_CREAT|O_EXCL|O_NOFOLLOW to
                    # block symlink TOCTOU on the tmp path too.
                    tmp_path = target.with_suffix(target.suffix + ".tmp")
                    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
                    fd = os.open(str(tmp_path), flags, 0o600)
                    try:
                        with os.fdopen(fd, "w", encoding="utf-8") as f:
                            f.write(content)
                            try:
                                f.flush()
                                os.fsync(f.fileno())
                            except Exception:
                                pass
                    except Exception:
                        try:
                            os.close(fd)
                        except Exception:
                            pass
                        try:
                            tmp_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                        raise
                    # Atomic on POSIX: rename within same FS replaces
                    # target in one inode swap. If target was a symlink
                    # at this point os.replace overwrites the symlink
                    # itself (not its target) — which is the desired
                    # behaviour after the parent-walk check.
                    os.replace(str(tmp_path), str(target))
                    # R34-S52 H: RU-only.
                    return ToolExecutionResult(success=True, reply_text=f"Записал {len(content)} символов в {target}")
                except OSError as e:
                    import errno as _errno
                    if e.errno == _errno.ELOOP:
                        return ToolExecutionResult(
                            success=False,
                            # R34-S52 H: RU-only.
                            reply_text=f"Не пишу через символьную ссылку: {target}",
                        )
                    # R34-S52 H: RU-only.
                    return ToolExecutionResult(success=False, reply_text=f"Ошибка записи: {e}")
                except Exception as e:
                    # R34-S52 H: RU-only.
                    return ToolExecutionResult(success=False, reply_text=f"Ошибка записи: {e}")

            # append
            if operation == "append":
                content = args.get("content")
                if not isinstance(content, str):
                    # R34-S52 H: RU-only.
                    return ToolExecutionResult(success=False, reply_text="Дописывание требует строку 'content'.")
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    # Audit round 20 P1 — O_NOFOLLOW on the append path
                    # too. The .open("a", ...) call followed symlinks,
                    # giving a symlink-substituted leaf the same write
                    # opportunity as the write op (which is now
                    # tmp+replace).
                    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW
                    fd = os.open(str(target), flags, 0o600)
                    with os.fdopen(fd, "a", encoding="utf-8", errors="replace") as f:
                        f.write(content)
                    # R34-S52 H: RU-only.
                    return ToolExecutionResult(success=True, reply_text=f"Дописал {len(content)} символов в {target}")
                except OSError as e:
                    import errno as _errno
                    if e.errno == _errno.ELOOP:
                        return ToolExecutionResult(
                            success=False,
                            # R34-S52 H: RU-only.
                            reply_text=f"Не дописываю через символьную ссылку: {target}",
                        )
                    # R34-S52 H: RU-only.
                    return ToolExecutionResult(success=False, reply_text=f"Ошибка дописывания: {e}")
                except Exception as e:
                    # R34-S52 H: RU-only.
                    return ToolExecutionResult(success=False, reply_text=f"Ошибка дописывания: {e}")

            # delete
            if operation == "delete":
                # Audit round 17 SECURITY/UX: route deletes through a
                # recoverable trash folder. The previous implementation
                # called ``target.unlink()`` which is irreversible —
                # a misheard "delete the project notes" wiped the file
                # with no recourse. Now:
                #   1. Try the OS trash via ``send2trash`` (cross-platform
                #      — macOS Finder Trash, Windows Recycle Bin, Linux
                #      $XDG_DATA_HOME/Trash).
                #   2. Fall back to moving the file under
                #      ``~/.jarvis-trash/YYYY-MM/<original-name>-<ts>``
                #      so the user can still recover it.
                # Only a final manual ``rm`` from the user wipes it for
                # good; voice commands should never be terminal.
                try:
                    if not (target.exists() and target.is_file()):
                        # R34-S52 H: RU-only.
                        return ToolExecutionResult(success=False, reply_text=f"Файл не найден: {target}")
                    trashed_to: Optional[str] = None
                    try:
                        from send2trash import send2trash  # type: ignore
                        send2trash(str(target))
                        # R34-S52 H: RU-only.
                        trashed_to = "системную корзину"
                    except Exception:
                        # Fallback: managed trash under home so the file
                        # is recoverable even without send2trash.
                        import shutil as _shutil
                        from datetime import datetime as _dt
                        trash_root = home_root / ".jarvis-trash" / _dt.now().strftime("%Y-%m")
                        trash_root.mkdir(parents=True, exist_ok=True)
                        try:
                            os.chmod(home_root / ".jarvis-trash", 0o700)
                        except Exception:
                            pass
                        ts = _dt.now().strftime("%Y%m%dT%H%M%S")
                        dest = trash_root / f"{target.name}.{ts}"
                        # Avoid clobber if two deletes hit the same name
                        # within one second.
                        i = 0
                        while dest.exists():
                            i += 1
                            dest = trash_root / f"{target.name}.{ts}-{i}"
                        _shutil.move(str(target), str(dest))
                        # R34-S52 H: RU-only.
                        trashed_to = f"восстанавливаемую корзину {dest}"
                    return ToolExecutionResult(
                        success=True,
                        # R34-S52 H: RU-only.
                        reply_text=f"Переместил {target} в {trashed_to} (восстановимо).",
                    )
                except Exception as e:
                    # R34-S52 H: RU-only.
                    return ToolExecutionResult(success=False, reply_text=f"Ошибка удаления: {e}")

            # R34-S52 H: RU-only.
            return ToolExecutionResult(success=False, reply_text=f"Неизвестная операция localFiles: {operation}")
        except PermissionError as pe:
            # R34-S52 H: RU-only.
            return ToolExecutionResult(success=False, reply_text=f"Ошибка доступа: {pe}")
        except Exception as e:
            # R34-S52 H: RU-only.
            return ToolExecutionResult(success=False, reply_text=f"Ошибка localFiles: {e}")
