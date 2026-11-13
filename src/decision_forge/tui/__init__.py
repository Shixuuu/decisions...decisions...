"""Cross-platform UTF-8 safe Rich console and prompt helpers for Decision Forge TUI."""

from __future__ import annotations

import io
import os
import sys
from dataclasses import dataclass
from typing import Any

from rich import box
from rich.console import Console


def is_utf8_stream(stream: Any = None) -> bool:
    """Return True if the target output stream uses UTF-8 encoding."""
    target = sys.stdout if stream is None else stream
    enc = getattr(target, "encoding", None) or ""
    return enc.lower().replace("-", "") == "utf8"


def get_icons(utf8: bool | None = None) -> dict[str, str]:
    """Return terminal-safe Unicode or ASCII status icons."""
    use_unicode = is_utf8_stream() if utf8 is None else utf8
    if use_unicode:
        return {
            "ok": "✓",
            "warn": "⚠",
            "err": "✗",
            "arrow": "→",
            "bullet": "•",
        }
    return {
        "ok": "[OK]",
        "warn": "[!]",
        "err": "[X]",
        "arrow": "->",
        "bullet": "*",
    }


def get_box_style(stream: Any = None) -> box.Box:
    """Choose rounded Rich boxes on UTF-8 terminals and pure ASCII boxes elsewhere."""
    return box.ROUNDED if is_utf8_stream(stream) else box.ASCII


def make_console(stream: Any = None, force_terminal: bool | None = None) -> Console:
    """Construct a Rich Console that never crashes on legacy Windows cp1252 terminals."""
    target = sys.stdout if stream is None else stream
    if (
        stream is None
        and sys.platform == "win32"
        and isinstance(target, io.TextIOWrapper)
        and getattr(target, "isatty", lambda: False)()
        and not is_utf8_stream(target)
    ):
        try:
            target.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    utf8 = is_utf8_stream(target)
    no_color = bool(os.environ.get("NO_COLOR"))
    kwargs: dict[str, Any] = {
        "file": stream,
        "safe_box": not utf8,
        "no_color": no_color,
    }
    if force_terminal is not None:
        kwargs["force_terminal"] = force_terminal
    if sys.platform == "win32" and utf8:
        kwargs["legacy_windows"] = False
    return Console(**kwargs)


@dataclass
class PromptDriver:
    """Cross-platform interactive prompt driver using prompt_toolkit on TTY and input() otherwise."""

    console: Console
    force_input_fallback: bool = False

    def is_interactive_tty(self) -> bool:
        if self.force_input_fallback:
            return False
        try:
            return bool(sys.stdin.isatty() and sys.stdout.isatty())
        except Exception:
            return False

    def _raw_prompt(
        self,
        label: str,
        default: str = "",
        completer_words: list[str] | None = None,
    ) -> str:
        suffix = f" [{default}]" if default != "" else ""
        prompt_text = f"{label}{suffix}: "
        if self.is_interactive_tty():
            try:
                from prompt_toolkit import prompt as pt_prompt
                from prompt_toolkit.completion import WordCompleter

                completer = (
                    WordCompleter(completer_words, ignore_case=True)
                    if completer_words
                    else None
                )
                ans = pt_prompt(prompt_text, completer=completer)
                cleaned = ans.strip() if ans is not None else ""
                return cleaned if cleaned != "" else default
            except EOFError:
                return default
            except KeyboardInterrupt:
                raise
            except Exception:
                pass

        try:
            raw = input(prompt_text)
        except EOFError:
            return default
        cleaned = raw.strip()
        return cleaned if cleaned != "" else default

    def ask_text(
        self,
        label: str,
        default: str = "",
        completer_words: list[str] | None = None,
        allow_empty: bool = True,
    ) -> str:
        for _ in range(5):
            val = self._raw_prompt(label, default=default, completer_words=completer_words)
            if val != "" or allow_empty:
                return val
            self.console.print("[yellow]Please enter a non-empty value.[/yellow]")
        raise ValueError(f"Missing required input for: {label}")

    def ask_confirm(self, label: str, default: bool = False) -> bool:
        hint = "Y/n" if default else "y/N"
        default_str = "y" if default else "n"
        for _ in range(5):
            raw = self._raw_prompt(f"{label} ({hint})", default="").lower()
            if raw == "":
                raw = default_str
            if raw in ("y", "yes", "true", "1"):
                return True
            if raw in ("n", "no", "false", "0"):
                return False
            self.console.print("[yellow]Please answer 'y' or 'n'.[/yellow]")
        return default

    def ask_choice(
        self,
        label: str,
        options: list[tuple[str, str]],
        default: str | None = None,
    ) -> str:
        if not options:
            raise ValueError("ask_choice requires at least one option")
        keys = [k for k, _ in options]
        default_key = default if (default in keys) else keys[0]
        for idx, (key, desc) in enumerate(options, 1):
            marker = " [default]" if key == default_key else ""
            self.console.print(f"  [cyan][{idx}][/cyan] [bold]{key}[/bold] - {desc}{marker}")

        completers = list(keys) + [str(i) for i in range(1, len(options) + 1)]
        for _ in range(5):
            raw = self._raw_prompt(label, default=default_key, completer_words=completers)
            if raw == "":
                return default_key
            if raw.isdigit():
                idx = int(raw)
                if 1 <= idx <= len(options):
                    return options[idx - 1][0]
            lowered = raw.lower()
            for key, _ in options:
                if key.lower() == lowered:
                    return key
            self.console.print(
                f"[yellow]Invalid choice '{raw}'. Pick 1-{len(options)} or one of: {', '.join(keys)}[/yellow]"
            )
        raise ValueError(f"Invalid selection after retries for: {label}")

    def ask_int(
        self,
        label: str,
        default: int,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int:
        for _ in range(5):
            raw = self._raw_prompt(label, default=str(default))
            try:
                val = int(raw)
                if minimum is not None and val < minimum:
                    self.console.print(f"[yellow]Value must be >= {minimum}.[/yellow]")
                    continue
                if maximum is not None and val > maximum:
                    self.console.print(f"[yellow]Value must be <= {maximum}.[/yellow]")
                    continue
                return val
            except ValueError:
                self.console.print(f"[yellow]Please enter an integer (got '{raw}').[/yellow]")
        return default

    def ask_float(
        self,
        label: str,
        default: float,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        for _ in range(5):
            raw = self._raw_prompt(label, default=str(default))
            try:
                val = float(raw)
                if minimum is not None and val < minimum:
                    self.console.print(f"[yellow]Value must be >= {minimum}.[/yellow]")
                    continue
                if maximum is not None and val > maximum:
                    self.console.print(f"[yellow]Value must be <= {maximum}.[/yellow]")
                    continue
                return val
            except ValueError:
                self.console.print(f"[yellow]Please enter a number (got '{raw}').[/yellow]")
        return default


def run_wizard(*args: Any, **kwargs: Any) -> Any:
    """Lazy wrapper delegating to `decision_forge.tui.wizard.run_wizard`."""
    from decision_forge.tui.wizard import run_wizard as _run_wizard

    return _run_wizard(*args, **kwargs)


__all__ = [
    "PromptDriver",
    "get_box_style",
    "get_icons",
    "is_utf8_stream",
    "make_console",
    "run_wizard",
]
