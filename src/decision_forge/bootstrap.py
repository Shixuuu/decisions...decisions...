"""OS environment health checks, Rich diagnostics, and optimal backend bootstrapper."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape as rich_escape
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

from decision_forge.hardware import (
    _linux_distro_family,
    _run_cmd,
    available_execution_targets,
    detect_hardware,
    resolve_kernel_stack,
    smoke_test_device,
)
from decision_forge.types import ExecutionTarget, HardwareProfile

_CORE_PYTHON_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "datasets",
    "rich",
    "pydantic",
)


def _is_module_available(module_name: str) -> bool:
    if module_name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except ValueError:
        return module_name in sys.modules


def _detect_cuda_uv_tag() -> str:
    """Choose the best uv PyTorch CUDA wheel tag based on installed NVIDIA driver."""
    out = _run_cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if out:
        first = out.splitlines()[0].strip()
        try:
            major = int(first.split(".", 1)[0])
            if major >= 560:
                return "cu126"
            if major >= 550:
                return "cu124"
            if major >= 530:
                return "cu121"
        except ValueError:
            pass
    return "cu126"


def _resolve_uv_torch_backend(backend: str) -> str | None:
    if backend == "xpu":
        return "xpu"
    if backend == "cuda":
        return _detect_cuda_uv_tag()
    if backend == "rocm":
        return "rocm6.3"
    if backend == "cpu":
        return "cpu"
    if backend == "mps":
        # macOS PyPI wheels include Metal/MPS directly; avoid overriding with cpu-only index.
        return None
    return "cpu"


def _format_cmd(cmd: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    return " ".join(shlex.quote(part) for part in cmd)


def _build_install_plan(
    profile: HardwareProfile,
    target_backend: str,
    missing_os_packages: list[str],
    yes: bool = False,
) -> dict[str, list[list[str]]]:
    os_commands: list[list[str]] = []
    python_commands: list[list[str]] = []

    if profile.os_name == "Linux" and missing_os_packages:
        distro = _linux_distro_family()
        need_sudo = hasattr(os, "geteuid") and os.geteuid() != 0 and bool(shutil.which("sudo"))
        sudo_prefix = ["sudo"] if need_sudo else []

        if target_backend == "xpu":
            if distro == "arch":
                cmd = [*sudo_prefix, "pacman", "-S", "--needed"]
                if yes:
                    cmd.append("--noconfirm")
                cmd.extend(["intel-compute-runtime", "level-zero-loader", "level-zero-headers"])
                os_commands.append(cmd)
            elif distro == "debian":
                os_commands.append([*sudo_prefix, "apt-get", "update"])
                os_commands.append(
                    [*sudo_prefix, "apt-get", "install", "-y", "intel-opencl-icd", "libze1", "libze-dev"]
                )
            elif distro == "fedora":
                os_commands.append(
                    [
                        *sudo_prefix,
                        "dnf",
                        "install",
                        "-y",
                        "intel-compute-runtime",
                        "level-zero",
                        "level-zero-devel",
                    ]
                )
            elif distro == "suse":
                os_commands.append(
                    [*sudo_prefix, "zypper", "install", "-y", "intel-opencl", "level-zero"]
                )
        elif target_backend == "rocm" and "rocm-dkms" in missing_os_packages:
            if distro == "arch":
                cmd = [*sudo_prefix, "pacman", "-S", "--needed"]
                if yes:
                    cmd.append("--noconfirm")
                cmd.extend(["rocm-hip-sdk", "rocm-smi-lib"])
                os_commands.append(cmd)
            elif distro == "fedora":
                os_commands.append([*sudo_prefix, "dnf", "install", "-y", "rocm-hip", "rocm-smi"])

    uv_bin = shutil.which("uv")
    uv_tag = _resolve_uv_torch_backend(target_backend)
    in_repo = Path("pyproject.toml").is_file()

    if uv_bin:
        base_cmd = [uv_bin, "pip", "install", "--python", sys.executable, "--reinstall-package", "torch"]
        if uv_tag:
            base_cmd.extend(["--torch-backend", uv_tag])
        if in_repo:
            extras = ".[cuda]" if (target_backend == "cuda" and profile.os_name != "Darwin") else "."
            base_cmd.extend(["-e", extras])
        else:
            base_cmd.append("torch")
        python_commands.append(base_cmd)

        if profile.os_name == "Linux" and target_backend == "xpu":
            python_commands.append(
                [
                    uv_bin,
                    "pip",
                    "install",
                    "--python",
                    sys.executable,
                    "--torch-backend",
                    "xpu",
                    "pytorch-triton-xpu",
                ]
            )
    else:
        pip_cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "torch"]
        if uv_tag and uv_tag != "cpu":
            pip_cmd.extend(["--index-url", f"https://download.pytorch.org/whl/{uv_tag}"])
        python_commands.append(pip_cmd)

    return {"os_commands": os_commands, "python_commands": python_commands}


def check_environment_health(
    profile: HardwareProfile | None = None,
    target_backend: str | None = None,
) -> dict[str, Any]:
    """Audit active PyTorch, OS compute runtimes, and Python dependencies against hardware."""
    prof = profile if profile is not None else detect_hardware()
    desired_backend = (
        prof.recommended_backend
        if target_backend in (None, "auto")
        else str(target_backend)
    )

    matching_devs = [d for d in prof.devices if d.backend == desired_backend]
    missing_os_packages: list[str] = []
    missing_python_packages: list[str] = []
    notes: list[str] = []

    for dev in matching_devs:
        for item in dev.missing_runtime:
            if item.startswith("torch"):
                if item not in missing_python_packages:
                    missing_python_packages.append(item)
            elif item not in missing_os_packages:
                missing_os_packages.append(item)

    if prof.active_torch_backend == "none" and "torch" not in missing_python_packages:
        missing_python_packages.append("torch")
        notes.append("PyTorch is not installed in the active Python environment.")

    for pkg in _CORE_PYTHON_PACKAGES:
        if pkg != "torch" and not _is_module_available(pkg):
            missing_python_packages.append(pkg)

    backend_matched = prof.active_torch_backend == desired_backend
    if not backend_matched and prof.active_torch_backend == "cpu" and desired_backend != "cpu":
        notes.append(
            f"CPU-only PyTorch ({prof.active_torch_version}) is active while {desired_backend} hardware is present."
        )
        torch_tag = f"torch-{desired_backend}"
        if torch_tag not in missing_python_packages:
            missing_python_packages.append(torch_tag)

    if prof.os_name == "Linux" and desired_backend == "xpu" and not _is_module_available("triton"):
        notes.append("Optional pytorch-triton-xpu compiler package is not installed.")

    plan = _build_install_plan(prof, desired_backend, missing_os_packages, yes=False)
    install_commands = [
        _format_cmd(c) for c in [*plan["os_commands"], *plan["python_commands"]]
    ]

    if prof.os_name == "Windows" and desired_backend == "xpu" and "ze_loader.dll" in missing_os_packages:
        notes.append(
            "Install latest Intel Arc/Core Ultra Windows Graphics Driver: "
            "https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html"
        )

    healthy = bool(backend_matched and not missing_os_packages and not missing_python_packages)

    return {
        "os_name": prof.os_name,
        "os_distro": prof.os_distro,
        "arch": prof.arch,
        "cpu_model": prof.cpu_model,
        "physical_cores": prof.physical_cores,
        "logical_cores": prof.logical_cores,
        "cpu_flags": list(prof.cpu_flags),
        "ram_total_gb": prof.ram_total_gb,
        "ram_available_gb": prof.ram_available_gb,
        "has_discrete_gpu": prof.has_discrete_gpu,
        "has_igpu": prof.has_igpu,
        "recommended_backend": prof.recommended_backend,
        "target_backend": desired_backend,
        "active_torch_version": prof.active_torch_version,
        "active_torch_backend": prof.active_torch_backend,
        "backend_matched": backend_matched,
        "missing_os_packages": missing_os_packages,
        "missing_python_packages": missing_python_packages,
        "uv_available": bool(shutil.which("uv")),
        "healthy": healthy,
        "install_commands": install_commands,
        "execution_targets": available_execution_targets(prof),
        "devices": [asdict(d) for d in prof.devices],
        "notes": notes,
    }


def _render_diagnostics(console: Console, profile: HardwareProfile, health: dict[str, Any]) -> None:
    summary = (
        f"[bold]OS:[/bold] {rich_escape(profile.os_distro)} ({rich_escape(profile.os_name)} / {rich_escape(profile.arch)})\n"
        f"[bold]CPU:[/bold] {rich_escape(profile.cpu_model)} ({profile.physical_cores}c / {profile.logical_cores}t) "
        f"flags={rich_escape(','.join(profile.cpu_flags) or 'none')}\n"
        f"[bold]RAM:[/bold] {profile.ram_available_gb:.1f} GB avail / {profile.ram_total_gb:.1f} GB total\n"
        f"[bold]PyTorch:[/bold] {rich_escape(profile.active_torch_version)} (active={rich_escape(profile.active_torch_backend)}, "
        f"recommended={rich_escape(profile.recommended_backend)})"
    )
    console.print(Panel(summary, title="Decision Forge Hardware Audit", border_style="cyan"))

    table = Table(title="Detected Compute Devices", show_lines=False)
    table.add_column("Vendor", style="bold")
    table.add_column("Device")
    table.add_column("Kind")
    table.add_column("Backend")
    table.add_column("Memory")
    table.add_column("Driver")
    table.add_column("Torch")
    table.add_column("Missing Runtime")

    for dev in profile.devices:
        mem_str = f"{dev.memory_gb:.1f} GB" + (" (unified)" if dev.unified_memory else "")
        torch_str = "[green]yes[/green]" if dev.available_in_torch else "[yellow]no[/yellow]"
        missing_str = rich_escape(", ".join(dev.missing_runtime)) if dev.missing_runtime else "-"
        table.add_row(
            rich_escape(dev.vendor),
            rich_escape(dev.name),
            rich_escape(dev.kind),
            rich_escape(dev.backend),
            mem_str,
            rich_escape(dev.driver or "-"),
            torch_str,
            missing_str,
        )
    console.print(table)


def _target_for_backend(profile: HardwareProfile, backend: str) -> ExecutionTarget:
    if backend == "cpu":
        return "cpu"
    for dev in profile.devices:
        if dev.backend == backend:
            if dev.kind == "discrete_gpu":
                return "gpu"
            if dev.kind == "igpu":
                return "igpu"
    return "cpu"


def ensure_optimal_backend(
    profile: HardwareProfile,
    auto_install: bool = False,
    yes: bool = False,
    target_backend: str | None = None,
    console: Console | None = None,
) -> bool:
    """Print Rich hardware diagnostics and optionally install missing OS/PyTorch runtimes."""
    out = console if console is not None else Console()
    desired_backend = (
        profile.recommended_backend
        if target_backend in (None, "auto")
        else str(target_backend)
    )
    health = check_environment_health(profile, target_backend=desired_backend)
    _render_diagnostics(out, profile, health)

    if health["healthy"]:
        out.print(
            f"[bold green]OK:[/bold green] Active PyTorch backend '{profile.active_torch_backend}' "
            f"matches optimal target '{desired_backend}'."
        )
        return True

    notes_lines = list(health["notes"])
    for dev in profile.devices:
        if dev.backend == desired_backend and dev.install_hint:
            notes_lines.append(f"Hint ({dev.name}): {dev.install_hint}")
    if health["install_commands"]:
        notes_lines.append("Commands:")
        notes_lines.extend(f"  {cmd}" for cmd in health["install_commands"])

    out.print(
        Panel(
            Text("\n".join(notes_lines) or "Environment needs runtime setup."),
            title=f"Action Required for '{desired_backend}' Backend",
            border_style="yellow",
        )
    )

    if not auto_install:
        return False

    if not yes:
        if not sys.stdin.isatty():
            out.print("[red]Non-interactive session: re-run with --yes to apply installation.[/red]")
            return False
        if not Confirm.ask(f"Install missing runtime & PyTorch for '{desired_backend}' now?", default=True):
            return False

    plan = _build_install_plan(
        profile,
        desired_backend,
        list(health["missing_os_packages"]),
        yes=True,
    )
    for cmd in [*plan["os_commands"], *plan["python_commands"]]:
        out.print(f"[cyan]$ {rich_escape(_format_cmd(cmd))}[/cyan]")
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            out.print(
                f"[red]Command exited with status {proc.returncode}: {rich_escape(_format_cmd(cmd))}[/red]"
            )
            return False

    refreshed = detect_hardware()
    refreshed_health = check_environment_health(refreshed, target_backend=desired_backend)
    exec_target = _target_for_backend(refreshed, desired_backend)
    kernel = resolve_kernel_stack(exec_target, refreshed)
    smoke_ok, smoke_msg = smoke_test_device(kernel.device_str, kernel.dtype)
    if smoke_ok:
        out.print(f"[bold green]Smoke test passed:[/bold green] {rich_escape(smoke_msg)}")
    else:
        out.print(f"[bold red]Smoke test failed:[/bold red] {rich_escape(smoke_msg)}")
    return bool(refreshed_health["healthy"] and smoke_ok)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `decide-setup`."""
    parser = argparse.ArgumentParser(
        prog="decide-setup",
        description="Inspect hardware and bootstrap optimal PyTorch + compute runtimes for Decision Forge.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check hardware and active PyTorch backend health (default action).",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install missing OS compute runtimes and optimal PyTorch wheel.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "cuda", "rocm", "xpu", "mps", "cpu"),
        default="auto",
        help="Override target backend instead of auto-detection.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Assume yes for non-interactive package installation.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON diagnostics to stdout.",
    )
    args = parser.parse_args(argv)

    profile = detect_hardware()
    target_backend = None if args.backend == "auto" else args.backend

    if args.json and not args.install:
        health = check_environment_health(profile, target_backend=target_backend)
        print(json.dumps(health, indent=2))
        return 0 if health["healthy"] else 1

    if args.install:
        console = Console(quiet=args.json)
        ok = ensure_optimal_backend(
            profile,
            auto_install=True,
            yes=args.yes,
            target_backend=args.backend,
            console=console,
        )
        if args.json:
            updated = detect_hardware()
            health = check_environment_health(updated, target_backend=target_backend)
            health["install_succeeded"] = ok
            print(json.dumps(health, indent=2))
        return 0 if ok else 1

    ok = ensure_optimal_backend(
        profile,
        auto_install=False,
        yes=args.yes,
        target_backend=args.backend,
    )
    return 0 if ok else 1


__all__ = [
    "check_environment_health",
    "ensure_optimal_backend",
    "main",
]
