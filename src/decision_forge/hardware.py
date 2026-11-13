"""Cross-platform hardware detection, kernel stack resolution, and runtime smoke checks."""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from decision_forge.types import (
    BackendKind,
    DeviceKind,
    ExecutionTarget,
    HardwareDevice,
    HardwareProfile,
    KernelStackConfig,
)

try:
    import psutil
except ImportError:  # pragma: no cover - optional fallback when psutil not installed
    psutil = None  # type: ignore[assignment]

# Overridable Linux probe paths (kept as Path objects so tests can redirect via monkeypatch)
_DRM_PATH = Path("/sys/class/drm")
_PCI_PATH = Path("/sys/bus/pci/devices")
_OS_RELEASE_PATH = Path("/etc/os-release")
_CPUINFO_PATH = Path("/proc/cpuinfo")
_MEMINFO_PATH = Path("/proc/meminfo")
_KFD_PATH = Path("/dev/kfd")

_INTERESTING_CPU_FLAGS = ("avx512_bf16", "amx", "avx512f", "avx2", "neon")

_INTEL_GPU_DEVICE_TABLE: dict[str, tuple[str, DeviceKind]] = {
    # Lunar Lake / Core Ultra 200V integrated Arc 130V/140V
    "0x64a0": ("Intel Arc Graphics 130V/140V", "igpu"),
    "0x64b0": ("Intel Arc Graphics 130V/140V", "igpu"),
    # Meteor Lake / Arrow Lake integrated Arc Graphics
    "0x7d55": ("Intel Arc Graphics (Meteor Lake)", "igpu"),
    "0x7dd5": ("Intel Arc Graphics (Meteor Lake)", "igpu"),
    "0x7d45": ("Intel Graphics (Meteor Lake)", "igpu"),
    "0x7d51": ("Intel Arc Graphics (Arrow Lake)", "igpu"),
    "0x7d67": ("Intel Graphics (Arrow Lake)", "igpu"),
    # Alder Lake / Raptor Lake Iris Xe & UHD iGPU
    "0x46a6": ("Intel Iris Xe Graphics", "igpu"),
    "0x46a8": ("Intel Iris Xe Graphics", "igpu"),
    "0xa7a0": ("Intel Iris Xe Graphics", "igpu"),
    "0xa780": ("Intel UHD Graphics 770", "igpu"),
    # Arc Alchemist discrete GPUs (DG2)
    "0x56a0": ("Intel Arc A770 Graphics", "discrete_gpu"),
    "0x56a1": ("Intel Arc A750 Graphics", "discrete_gpu"),
    "0x56a2": ("Intel Arc A580 Graphics", "discrete_gpu"),
    "0x56a5": ("Intel Arc A380 Graphics", "discrete_gpu"),
    "0x56a6": ("Intel Arc A310 Graphics", "discrete_gpu"),
    "0x5690": ("Intel Arc A770M Graphics", "discrete_gpu"),
    "0x5691": ("Intel Arc A730M Graphics", "discrete_gpu"),
    "0x5692": ("Intel Arc A550M Graphics", "discrete_gpu"),
    "0x5693": ("Intel Arc A370M Graphics", "discrete_gpu"),
    "0x5694": ("Intel Arc A350M Graphics", "discrete_gpu"),
    "0x56b0": ("Intel Arc Pro A30M Graphics", "discrete_gpu"),
    "0x56b1": ("Intel Arc Pro A40/A50 Graphics", "discrete_gpu"),
    "0x56be": ("Intel Arc Pro A60 Graphics", "discrete_gpu"),
    # Arc Battlemage discrete GPUs (BMG)
    "0xe20b": ("Intel Arc B580 Graphics", "discrete_gpu"),
    "0xe20c": ("Intel Arc B570 Graphics", "discrete_gpu"),
}

_VIRTUAL_ADAPTER_PATTERNS = (
    "microsoft basic display",
    "microsoft hyper-v",
    "microsoft remote display",
    "vmware svga",
    "virtualbox",
    "parallels display",
    "qxl",
    "virtio gpu",
    "bochs",
    "cirrus",
)


def _run_cmd(args: list[str], timeout: float = 5.0) -> str | None:
    """Run a subprocess safely and return stripped stdout, or None on error."""
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return ""


def _read_linux_os_release() -> dict[str, str]:
    """Parse /etc/os-release key-value pairs cleanly."""
    raw = _safe_read_text(_OS_RELEASE_PATH)
    info: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        info[key.strip()] = val.strip().strip('"').strip("'")
    return info


def _linux_distro_family(os_release: dict[str, str] | None = None) -> str:
    """Map /etc/os-release fields into package-manager distro families."""
    info = os_release if os_release is not None else _read_linux_os_release()
    tokens = f"{info.get('ID', '')} {info.get('ID_LIKE', '')} {info.get('NAME', '')}".lower()
    if any(k in tokens for k in ("cachyos", "arch", "endeavouros", "manjaro", "garuda")):
        return "arch"
    if any(k in tokens for k in ("ubuntu", "debian", "pop", "linuxmint", "kali")):
        return "debian"
    if any(k in tokens for k in ("fedora", "rhel", "centos", "rocky", "almalinux", "nobara")):
        return "fedora"
    if any(k in tokens for k in ("suse", "opensuse", "sles", "tumbleweed", "leap")):
        return "suse"
    return "linux"


def _detect_os_distro(os_name: str, os_release: dict[str, str]) -> str:
    if os_name == "Linux":
        return (
            os_release.get("PRETTY_NAME")
            or os_release.get("NAME")
            or os_release.get("ID")
            or "Linux"
        )
    if os_name == "Darwin":
        mac_ver = platform.mac_ver()[0]
        return f"macOS {mac_ver}".strip() if mac_ver else "macOS"
    if os_name == "Windows":
        rel = platform.release()
        return f"Windows {rel}".strip() if rel else "Windows"
    return platform.platform() or os_name


def _extract_interesting_flags(raw_tokens: set[str]) -> list[str]:
    lowered = {t.lower() for t in raw_tokens if t}
    found: list[str] = []
    if "avx512_bf16" in lowered:
        found.append("avx512_bf16")
    if "amx" in lowered or any(t.startswith("amx_") or t.startswith("amx-") for t in lowered):
        found.append("amx")
    if "avx512f" in lowered:
        found.append("avx512f")
    if "avx2" in lowered:
        found.append("avx2")
    if "neon" in lowered or "asimd" in lowered:
        found.append("neon")
    return [f for f in _INTERESTING_CPU_FLAGS if f in found]


def _probe_cpu_linux(arch: str) -> tuple[str, list[str]]:
    raw = _safe_read_text(_CPUINFO_PATH)
    model_name = ""
    hardware_fallback = ""
    model_fallback = ""
    raw_flags: set[str] = set()
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, val = [p.strip() for p in line.split(":", 1)]
        low_key = key.lower()
        if not model_name and low_key == "model name":
            model_name = val
        elif not hardware_fallback and low_key == "hardware":
            hardware_fallback = val
        elif not model_fallback and low_key == "model" and not val.isdigit():
            model_fallback = val
        elif low_key in ("flags", "features"):
            raw_flags.update(val.split())
    model = model_name or hardware_fallback or model_fallback or platform.processor() or f"Linux {arch} CPU"
    if arch.lower() in ("aarch64", "arm64"):
        raw_flags.add("neon")
    return model, _extract_interesting_flags(raw_flags)


def _probe_cpu_darwin(arch: str) -> tuple[str, list[str]]:
    model = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"]) or ""
    if not model:
        hw_model = _run_cmd(["sysctl", "-n", "hw.model"]) or ""
        model = f"Apple Silicon ({hw_model})" if hw_model else (platform.processor() or f"Apple {arch}")

    raw_flags: set[str] = set()
    if arch.lower() in ("arm64", "aarch64"):
        raw_flags.add("neon")
    else:
        for sysctl_key in ("machdep.cpu.features", "machdep.cpu.leaf7_features"):
            out = _run_cmd(["sysctl", "-n", sysctl_key])
            if out:
                raw_flags.update(t.lower().replace(".", "_") for t in out.split())
    return model, _extract_interesting_flags(raw_flags)


def _probe_cpu_windows(arch: str) -> tuple[str, list[str]]:
    model = ""
    if sys.platform == "win32":
        try:
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as handle:
                val, _ = winreg.QueryValueEx(handle, "ProcessorNameString")
                model = str(val).strip()
        except OSError:
            model = ""

    if not model:
        out = _run_cmd(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
            ]
        )
        if out:
            model = out.splitlines()[0].strip()

    if not model:
        model = platform.processor() or f"Windows {arch} CPU"

    raw_flags: set[str] = set()
    if arch.lower() in ("arm64", "aarch64"):
        raw_flags.add("neon")
    elif sys.platform == "win32" and hasattr(ctypes, "windll"):
        try:
            is_proc_feature = ctypes.windll.kernel32.IsProcessorFeaturePresent  # type: ignore[attr-defined]
            if is_proc_feature(40):  # PF_AVX2_INSTRUCTIONS_AVAILABLE
                raw_flags.add("avx2")
            if is_proc_feature(42):  # PF_AVX512F_INSTRUCTIONS_AVAILABLE
                raw_flags.add("avx512f")
        except (OSError, AttributeError):
            pass
    return model, _extract_interesting_flags(raw_flags)


def _probe_cpu(os_name: str, arch: str) -> tuple[str, int, int, list[str]]:
    if os_name == "Linux":
        model, flags = _probe_cpu_linux(arch)
    elif os_name == "Darwin":
        model, flags = _probe_cpu_darwin(arch)
    elif os_name == "Windows":
        model, flags = _probe_cpu_windows(arch)
    else:
        model = platform.processor() or f"{os_name} {arch} CPU"
        flags = ["neon"] if arch.lower() in ("arm64", "aarch64") else []

    logical = None
    physical = None
    if psutil is not None:
        try:
            logical = psutil.cpu_count(logical=True)
            physical = psutil.cpu_count(logical=False)
        except Exception:
            logical = None
            physical = None

    if not logical:
        if os_name == "Darwin":
            val = _run_cmd(["sysctl", "-n", "hw.logicalcpu"])
            logical = int(val) if val and val.isdigit() else None
        logical = logical or os.cpu_count() or 1

    if not physical:
        if os_name == "Darwin":
            val = _run_cmd(["sysctl", "-n", "hw.physicalcpu"])
            physical = int(val) if val and val.isdigit() else None
        physical = physical or logical

    logical = max(1, int(logical))
    physical = max(1, min(int(physical), logical))
    return model, physical, logical, flags


def _probe_ram(os_name: str) -> tuple[float, float]:
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            gb = 1024**3
            return round(vm.total / gb, 2), round(vm.available / gb, 2)
        except Exception:
            pass

    if os_name == "Linux":
        raw = _safe_read_text(_MEMINFO_PATH)
        kb_vals: dict[str, int] = {}
        for line in raw.splitlines():
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            parts = val.strip().split()
            if parts and parts[0].isdigit():
                kb_vals[key.strip()] = int(parts[0])
        total_kb = kb_vals.get("MemTotal", 0)
        avail_kb = kb_vals.get("MemAvailable", kb_vals.get("MemFree", total_kb))
        if total_kb > 0:
            return round(total_kb / (1024**2), 2), round(avail_kb / (1024**2), 2)

    if os_name == "Darwin":
        mem_bytes = _run_cmd(["sysctl", "-n", "hw.memsize"])
        if mem_bytes and mem_bytes.isdigit():
            total_gb = round(int(mem_bytes) / (1024**3), 2)
            return total_gb, round(total_gb * 0.75, 2)

    if os_name == "Windows" and sys.platform == "win32" and hasattr(ctypes, "windll"):
        try:

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
                gb = 1024**3
                return round(stat.ullTotalPhys / gb, 2), round(stat.ullAvailPhys / gb, 2)
        except Exception:
            pass

    return 0.0, 0.0


def _inspect_torch() -> dict[str, Any]:
    """Probe active PyTorch installation and accelerator availability safely."""
    info: dict[str, Any] = {
        "installed": False,
        "version": "not_installed",
        "backend": "none",
        "cuda_available": False,
        "rocm_available": False,
        "xpu_available": False,
        "mps_available": False,
        "cuda_devices": [],
        "xpu_devices": [],
    }
    try:
        import torch
    except ImportError:
        return info

    version = str(getattr(torch, "__version__", "unknown"))
    info["installed"] = True
    info["version"] = version

    torch_ver_obj = getattr(torch, "version", None)
    hip_ver = getattr(torch_ver_obj, "hip", None) if torch_ver_obj else None
    cuda_ver = getattr(torch_ver_obj, "cuda", None) if torch_ver_obj else None

    cuda_mod = getattr(torch, "cuda", None)
    cuda_avail = False
    if cuda_mod is not None and callable(getattr(cuda_mod, "is_available", None)):
        try:
            cuda_avail = bool(cuda_mod.is_available())
        except Exception:
            cuda_avail = False

    if hip_ver or "+rocm" in version.lower():
        info["rocm_available"] = cuda_avail
    else:
        info["cuda_available"] = cuda_avail

    if cuda_avail and cuda_mod is not None:
        try:
            count = int(cuda_mod.device_count())
        except Exception:
            count = 1
        for idx in range(max(0, count)):
            dev_entry: dict[str, Any] = {"index": idx, "name": "", "memory_gb": 0.0, "major": None}
            try:
                dev_entry["name"] = str(cuda_mod.get_device_name(idx))
            except Exception:
                pass
            try:
                props = cuda_mod.get_device_properties(idx)
                total_mem = getattr(props, "total_memory", 0)
                if total_mem:
                    dev_entry["memory_gb"] = round(float(total_mem) / (1024**3), 2)
                if hasattr(props, "major"):
                    dev_entry["major"] = int(props.major)
            except Exception:
                pass
            info["cuda_devices"].append(dev_entry)

    xpu_mod = getattr(torch, "xpu", None)
    xpu_avail = False
    if xpu_mod is not None and callable(getattr(xpu_mod, "is_available", None)):
        try:
            xpu_avail = bool(xpu_mod.is_available())
        except Exception:
            xpu_avail = False
    info["xpu_available"] = xpu_avail

    if xpu_avail and xpu_mod is not None:
        try:
            count = int(xpu_mod.device_count())
        except Exception:
            count = 1
        for idx in range(max(0, count)):
            dev_entry = {"index": idx, "name": "", "memory_gb": 0.0}
            try:
                dev_entry["name"] = str(xpu_mod.get_device_name(idx))
            except Exception:
                pass
            try:
                props = xpu_mod.get_device_properties(idx)
                total_mem = getattr(props, "total_memory", 0)
                if total_mem:
                    dev_entry["memory_gb"] = round(float(total_mem) / (1024**3), 2)
            except Exception:
                pass
            info["xpu_devices"].append(dev_entry)

    mps_avail = False
    backends = getattr(torch, "backends", None)
    mps_mod = getattr(backends, "mps", None) if backends is not None else None
    if mps_mod is not None and callable(getattr(mps_mod, "is_available", None)):
        try:
            mps_avail = bool(mps_mod.is_available())
        except Exception:
            mps_avail = False
    info["mps_available"] = mps_avail

    low_ver = version.lower()
    if hip_ver or "+rocm" in low_ver:
        info["backend"] = "rocm"
    elif cuda_ver or "+cu" in low_ver or info["cuda_available"]:
        info["backend"] = "cuda"
    elif "+xpu" in low_ver or xpu_avail:
        info["backend"] = "xpu"
    elif mps_avail:
        info["backend"] = "mps"
    else:
        info["backend"] = "cpu"

    return info


def _normalize_pci_slot(slot: str) -> str:
    slot = slot.strip().lower()
    if slot.startswith("0000:"):
        return slot[5:]
    return slot


def _parse_lspci_display_devices() -> list[dict[str, str]]:
    """Return display/VGA/3D PCI controllers reported by `lspci -nn`."""
    out = _run_cmd(["lspci", "-nn"])
    if not out:
        return []
    devices: list[dict[str, str]] = []
    pattern = re.compile(
        r"^(?P<slot>[0-9a-fA-F:.]+)\s+(?P<cls>[^:\[]+?)(?:\s+\[(?P<cls_id>[0-9a-fA-F]{4})\])?:\s+"
        r"(?P<desc>.+?)\s+\[(?P<vendor>[0-9a-fA-F]{4}):(?P<device>[0-9a-fA-F]{4})\](?:\s+\(rev[^)]+\))?$"
    )
    for line in out.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        cls_text = match.group("cls").lower()
        cls_id = (match.group("cls_id") or "").lower()
        is_display = cls_id.startswith("03") or any(
            kw in cls_text for kw in ("vga", "3d", "display")
        )
        if not is_display:
            continue
        devices.append(
            {
                "slot": _normalize_pci_slot(match.group("slot")),
                "vendor_id": f"0x{match.group('vendor').lower()}",
                "device_id": f"0x{match.group('device').lower()}",
                "name": match.group("desc").strip(),
                "driver": "",
            }
        )
    return devices


def _scan_linux_pci_gpus() -> list[dict[str, str]]:
    """Discover display controllers from /sys/class/drm, /sys/bus/pci/devices, and lspci."""
    by_slot: dict[str, dict[str, str]] = {}
    lspci_entries = _parse_lspci_display_devices()
    lspci_by_slot = {entry["slot"]: entry for entry in lspci_entries}

    def _register_sysfs_device(dev_dir: Path, require_display_class: bool) -> None:
        if not dev_dir.exists():
            return
        vendor = _safe_read_text(dev_dir / "vendor").lower()
        if vendor not in ("0x10de", "0x1002", "0x8086"):
            return
        pci_class = _safe_read_text(dev_dir / "class").lower()
        if require_display_class and pci_class and not pci_class.startswith("0x03"):
            return

        uevent = _safe_read_text(dev_dir / "uevent")
        slot = ""
        driver = ""
        for uline in uevent.splitlines():
            if uline.startswith("PCI_SLOT_NAME="):
                slot = _normalize_pci_slot(uline.split("=", 1)[1])
            elif uline.startswith("DRIVER="):
                driver = uline.split("=", 1)[1].strip()

        if not slot:
            try:
                slot = _normalize_pci_slot(dev_dir.resolve().name)
            except OSError:
                slot = _normalize_pci_slot(dev_dir.name)

        if not driver:
            driver_link = dev_dir / "driver"
            try:
                if driver_link.is_symlink() or driver_link.exists():
                    driver = driver_link.resolve().name
            except OSError:
                pass

        device_id = _safe_read_text(dev_dir / "device").lower()
        lspci_match = lspci_by_slot.get(slot, {})
        name = lspci_match.get("name", "")
        entry = by_slot.setdefault(
            slot,
            {
                "slot": slot,
                "vendor_id": vendor,
                "device_id": device_id or lspci_match.get("device_id", ""),
                "name": name,
                "driver": driver,
            },
        )
        if driver and not entry["driver"]:
            entry["driver"] = driver
        if name and not entry["name"]:
            entry["name"] = name
        if device_id and not entry["device_id"]:
            entry["device_id"] = device_id

    if _DRM_PATH.exists():
        try:
            for card in sorted(_DRM_PATH.iterdir()):
                if re.fullmatch(r"card\d+", card.name):
                    _register_sysfs_device(card / "device", require_display_class=False)
        except OSError:
            pass

    if _PCI_PATH.exists():
        try:
            for pci_dev in sorted(_PCI_PATH.iterdir()):
                _register_sysfs_device(pci_dev, require_display_class=True)
        except OSError:
            pass

    for entry in lspci_entries:
        if entry["vendor_id"] not in ("0x10de", "0x1002", "0x8086"):
            continue
        if entry["slot"] not in by_slot:
            by_slot[entry["slot"]] = dict(entry)

    return list(by_slot.values())


def _probe_windows_video_controllers() -> list[dict[str, Any]]:
    """Query Win32_VideoController on Windows hosts."""
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,AdapterRAM,DriverVersion,PNPDeviceID | "
            "ConvertTo-Json -Compress"
        ),
    ]
    out = _run_cmd(cmd, timeout=8.0)
    if not out:
        return []
    try:
        parsed = json.loads(out)
    except json.JSONDecodeError:
        return []
    items = [parsed] if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])
    results: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("Name") or "").strip()
        if not name or any(p in name.lower() for p in _VIRTUAL_ADAPTER_PATTERNS):
            continue
        raw_ram = item.get("AdapterRAM")
        ram_gb = 0.0
        try:
            if raw_ram is not None and int(raw_ram) > 0:
                ram_gb = round(int(raw_ram) / (1024**3), 2)
        except (TypeError, ValueError):
            ram_gb = 0.0
        results.append(
            {
                "name": name,
                "ram_gb": ram_gb,
                "driver": str(item.get("DriverVersion") or "").strip(),
                "pnp_id": str(item.get("PNPDeviceID") or "").upper(),
            }
        )
    return results


def _probe_nvidia_smi() -> list[dict[str, Any]]:
    out = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return []
    gpus: list[dict[str, Any]] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        name = parts[0]
        mem_gb = 0.0
        try:
            mem_mib = float(parts[1])
            mem_gb = round(mem_mib / 1024.0, 2)
        except ValueError:
            pass
        driver = parts[2] if len(parts) >= 3 else ""
        gpus.append({"name": name, "memory_gb": mem_gb, "driver": driver})
    return gpus


def _probe_rocm_smi() -> list[dict[str, Any]]:
    out = _run_cmd(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if out:
        try:
            parsed = json.loads(out)
            if isinstance(parsed, dict):
                gpus: list[dict[str, Any]] = []
                for card_key, card_val in parsed.items():
                    if not isinstance(card_val, dict):
                        continue
                    name = str(
                        card_val.get("Card series")
                        or card_val.get("Card Series")
                        or card_val.get("Market Name")
                        or card_key
                    )
                    total_bytes = card_val.get("VRAM Total Memory (B)") or 0
                    mem_gb = 0.0
                    try:
                        if float(total_bytes) > 0:
                            mem_gb = round(float(total_bytes) / (1024**3), 2)
                    except (TypeError, ValueError):
                        pass
                    gpus.append({"name": name, "memory_gb": mem_gb})
                if gpus:
                    return gpus
        except json.JSONDecodeError:
            pass

    plain = _run_cmd(["rocm-smi"])
    if not plain:
        return []
    gpus = []
    for line in plain.splitlines():
        if re.match(r"^\s*\d+\s+", line):
            gpus.append({"name": "AMD ROCm GPU", "memory_gb": 0.0})
    return gpus


def _check_level_zero_loader(os_name: str) -> bool:
    """Check whether Level Zero loader library (libze_loader.so / ze_loader.dll) is present."""
    if ctypes.util.find_library("ze_loader"):
        return True
    if os_name == "Linux":
        ldconfig_out = _run_cmd(["ldconfig", "-p"])
        if ldconfig_out and "libze_loader.so" in ldconfig_out:
            return True
    elif os_name == "Windows":
        sys_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        if (sys_root / "System32" / "ze_loader.dll").exists():
            return True
    return False


def _classify_intel_gpu(name: str, device_id: str) -> tuple[str, DeviceKind]:
    """Distinguish Intel Arc integrated graphics (e.g. 130V/140V) vs Arc A/B discrete."""
    norm_id = device_id.lower()
    if norm_id and not norm_id.startswith("0x"):
        norm_id = f"0x{norm_id}"
    table_hit = _INTEL_GPU_DEVICE_TABLE.get(norm_id)

    low_name = name.lower()
    is_discrete = bool(
        re.search(r"\barc(?:\(tm\))?\s*(?:pro\s*)?[ab]\d{3}m?\b", low_name)
        or any(tok in low_name for tok in ("dg1", "dg2", "battlemage", "data center gpu", "flex"))
    )
    is_igpu = bool(
        re.search(r"\b1[34]0[vt]\b", low_name)
        or any(
            tok in low_name
            for tok in (
                "core ultra",
                "meteor lake",
                "lunar lake",
                "arrow lake",
                "panther lake",
                "iris",
                "uhd graphics",
                "hd graphics",
            )
        )
        or ("arc" in low_name and "graphics" in low_name and not is_discrete)
    )

    if is_discrete:
        kind: DeviceKind = "discrete_gpu"
    elif is_igpu:
        kind = "igpu"
    elif table_hit is not None:
        kind = table_hit[1]
    else:
        kind = "igpu"

    resolved_name = name.strip() or (table_hit[0] if table_hit else "Intel Arc GPU")
    return resolved_name, kind


def _classify_amd_gpu(name: str, vram_gb: float = 0.0) -> DeviceKind:
    """Distinguish AMD discrete GPUs (RX / Instinct / Pro) from Radeon iGPUs / APUs."""
    low = name.lower()
    if any(
        tok in low
        for tok in (
            "radeon rx",
            "instinct",
            "mi100",
            "mi200",
            "mi210",
            "mi250",
            "mi300",
            "pro w",
            "pro v",
            "navi",
        )
    ) or re.search(r"\brx\s*[56789]\d{3}\b", low):
        return "discrete_gpu"

    if any(
        tok in low
        for tok in (
            "radeon(tm) graphics",
            "radeon graphics",
            "vega 3",
            "vega 6",
            "vega 7",
            "vega 8",
            "vega 11",
            "phoenix",
            "rembrandt",
            "cezanne",
            "lucienne",
            "renoir",
            "picasso",
            "raven",
            "strix",
            "hawk point",
            "raphael",
            "granite ridge",
            "mendocino",
            "van gogh",
            "apu",
            "igpu",
            "integrated",
        )
    ) or re.search(r"\b[678]\d{2}m\b", low):
        return "igpu"

    if 0.0 < vram_gb <= 2.5:
        return "igpu"
    return "discrete_gpu"


def _build_install_hint(
    backend: BackendKind,
    os_name: str,
    distro_family: str,
    missing_runtime: list[str],
) -> str:
    if not missing_runtime:
        return ""
    if backend == "xpu":
        if os_name == "Windows":
            return (
                "Install latest Intel Arc/Core Ultra Graphics Driver: "
                "https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html "
                "then run: uv pip install -e . --torch-backend xpu"
            )
        if distro_family == "arch":
            return (
                "sudo pacman -S --needed intel-compute-runtime level-zero-loader level-zero-headers && "
                "uv pip install -e . --torch-backend xpu && "
                "uv pip install --torch-backend xpu pytorch-triton-xpu"
            )
        if distro_family == "debian":
            return (
                "sudo apt-get update && sudo apt-get install -y intel-opencl-icd libze1 libze-dev && "
                "uv pip install -e . --torch-backend xpu && "
                "uv pip install --torch-backend xpu pytorch-triton-xpu"
            )
        if distro_family == "fedora":
            return (
                "sudo dnf install -y intel-compute-runtime level-zero level-zero-devel && "
                "uv pip install -e . --torch-backend xpu && "
                "uv pip install --torch-backend xpu pytorch-triton-xpu"
            )
        if distro_family == "suse":
            return (
                "sudo zypper install -y intel-opencl level-zero && "
                "uv pip install -e . --torch-backend xpu && "
                "uv pip install --torch-backend xpu pytorch-triton-xpu"
            )
        return (
            "Install intel-compute-runtime and level-zero-loader, then run: "
            "uv pip install -e . --torch-backend xpu"
        )

    if backend == "cuda":
        if os_name == "Windows":
            if "nvidia-driver" in missing_runtime:
                return (
                    "Install NVIDIA GeForce/RTX Driver (https://www.nvidia.com/drivers), "
                    "then run: uv pip install -e . --torch-backend cu126"
                )
            return "uv pip install -e . --torch-backend cu126"
        if "nvidia-driver" in missing_runtime:
            if distro_family == "arch":
                return (
                    "sudo pacman -S --needed nvidia-dkms nvidia-utils && "
                    'uv pip install -e ".[cuda]" --torch-backend cu126'
                )
            if distro_family == "debian":
                return (
                    "sudo apt-get update && sudo apt-get install -y nvidia-driver-550 && "
                    'uv pip install -e ".[cuda]" --torch-backend cu126'
                )
            if distro_family == "fedora":
                return (
                    "sudo dnf install -y akmod-nvidia xorg-x11-drv-nvidia-cuda && "
                    'uv pip install -e ".[cuda]" --torch-backend cu126'
                )
        return 'uv pip install -e ".[cuda]" --torch-backend cu126'

    if backend == "rocm":
        if os_name == "Windows":
            return (
                "PyTorch ROCm wheels primarily target Linux; install AMD Adrenalin/HIP SDK or use "
                "WSL2 Linux ROCm / CPU backend on Windows."
            )
        if distro_family == "arch":
            return (
                "sudo pacman -S --needed rocm-hip-sdk rocm-smi-lib && "
                "uv pip install -e . --torch-backend rocm6.3"
            )
        if distro_family == "debian":
            return (
                "Install AMD ROCm kernel & HIP runtime (amdgpu-install / rocm-libs), then run: "
                "uv pip install -e . --torch-backend rocm6.3"
            )
        if distro_family == "fedora":
            return (
                "sudo dnf install -y rocm-hip rocm-smi && "
                "uv pip install -e . --torch-backend rocm6.3"
            )
        return "uv pip install -e . --torch-backend rocm6.3"

    if backend == "mps":
        return "uv pip install -e . (ensure macOS >= 12.3 and Apple Silicon Python)"

    return "uv pip install -e . --torch-backend cpu"


def _build_nvidia_devices(
    os_name: str,
    distro_family: str,
    pci_devs: list[dict[str, str]],
    win_controllers: list[dict[str, Any]],
    torch_info: dict[str, Any],
) -> list[HardwareDevice]:
    smi_gpus = _probe_nvidia_smi()
    pci_nvidia = [d for d in pci_devs if d.get("vendor_id") == "0x10de"]
    win_nvidia = [
        c for c in win_controllers if "nvidia" in c["name"].lower() or "VEN_10DE" in c["pnp_id"]
    ]
    torch_cuda_devs = torch_info.get("cuda_devices", [])
    cuda_avail = bool(torch_info.get("cuda_available"))

    count = max(len(smi_gpus), len(pci_nvidia), len(win_nvidia), len(torch_cuda_devs))
    if count == 0:
        return []

    devices: list[HardwareDevice] = []
    for idx in range(count):
        smi = smi_gpus[idx] if idx < len(smi_gpus) else {}
        pci = pci_nvidia[idx] if idx < len(pci_nvidia) else {}
        win = win_nvidia[idx] if idx < len(win_nvidia) else {}
        tdev = torch_cuda_devs[idx] if idx < len(torch_cuda_devs) else {}

        name = str(
            tdev.get("name") or smi.get("name") or win.get("name") or pci.get("name") or "NVIDIA GPU"
        )
        mem_gb = float(tdev.get("memory_gb") or smi.get("memory_gb") or win.get("ram_gb") or 0.0)
        driver = str(smi.get("driver") or win.get("driver") or pci.get("driver") or "")

        missing: list[str] = []
        if not smi_gpus and not cuda_avail and not driver:
            missing.append("nvidia-driver")
        if not cuda_avail:
            missing.append("torch-cuda")

        devices.append(
            HardwareDevice(
                vendor="NVIDIA",
                name=name,
                kind="discrete_gpu",
                backend="cuda",
                memory_gb=round(mem_gb, 2),
                unified_memory=False,
                driver=driver,
                available_in_torch=cuda_avail,
                missing_runtime=missing,
                install_hint=_build_install_hint("cuda", os_name, distro_family, missing),
            )
        )
    return devices


def _build_intel_devices(
    os_name: str,
    distro_family: str,
    ram_total_gb: float,
    pci_devs: list[dict[str, str]],
    win_controllers: list[dict[str, Any]],
    torch_info: dict[str, Any],
) -> list[HardwareDevice]:
    pci_intel = [d for d in pci_devs if d.get("vendor_id") == "0x8086"]
    win_intel = [
        c for c in win_controllers if "intel" in c["name"].lower() or "VEN_8086" in c["pnp_id"]
    ]
    torch_xpu_devs = torch_info.get("xpu_devices", [])
    xpu_avail = bool(torch_info.get("xpu_available"))

    count = max(len(pci_intel), len(win_intel), len(torch_xpu_devs))
    if count == 0:
        return []

    has_ze_loader = _check_level_zero_loader(os_name)
    devices: list[HardwareDevice] = []
    for idx in range(count):
        pci = pci_intel[idx] if idx < len(pci_intel) else {}
        win = win_intel[idx] if idx < len(win_intel) else {}
        tdev = torch_xpu_devs[idx] if idx < len(torch_xpu_devs) else {}

        raw_name = str(tdev.get("name") or win.get("name") or pci.get("name") or "")
        dev_id = str(pci.get("device_id") or "")
        if not dev_id and win.get("pnp_id"):
            m = re.search(r"DEV_([0-9A-F]{4})", str(win["pnp_id"]))
            if m:
                dev_id = f"0x{m.group(1).lower()}"
        name, kind = _classify_intel_gpu(raw_name, dev_id)
        unified = kind == "igpu"

        mem_gb = float(tdev.get("memory_gb") or win.get("ram_gb") or 0.0)
        if unified and mem_gb <= 0.0 and ram_total_gb > 0.0:
            mem_gb = round(ram_total_gb * 0.5, 2)

        driver = str(pci.get("driver") or win.get("driver") or "")
        missing: list[str] = []
        if not xpu_avail:
            if os_name == "Linux" and not has_ze_loader:
                missing.extend(["intel-compute-runtime", "level-zero-loader"])
            elif os_name == "Windows" and not has_ze_loader:
                missing.append("ze_loader.dll")
            missing.append("torch-xpu")

        devices.append(
            HardwareDevice(
                vendor="Intel",
                name=name,
                kind=kind,
                backend="xpu",
                memory_gb=round(mem_gb, 2),
                unified_memory=unified,
                driver=driver,
                available_in_torch=xpu_avail,
                missing_runtime=missing,
                install_hint=_build_install_hint("xpu", os_name, distro_family, missing),
            )
        )
    return devices


def _build_amd_devices(
    os_name: str,
    distro_family: str,
    ram_total_gb: float,
    pci_devs: list[dict[str, str]],
    win_controllers: list[dict[str, Any]],
    torch_info: dict[str, Any],
) -> list[HardwareDevice]:
    rocm_gpus = _probe_rocm_smi()
    pci_amd = [d for d in pci_devs if d.get("vendor_id") == "0x1002"]
    win_amd = [
        c
        for c in win_controllers
        if any(tok in c["name"].lower() for tok in ("amd", "radeon", "ati"))
        or "VEN_1002" in c["pnp_id"]
    ]
    rocm_avail = bool(torch_info.get("rocm_available"))
    torch_rocm_devs = torch_info.get("cuda_devices", []) if rocm_avail else []

    count = max(len(pci_amd), len(win_amd), len(rocm_gpus), len(torch_rocm_devs))
    if count == 0:
        return []

    kfd_present = os_name == "Linux" and _KFD_PATH.exists()
    devices: list[HardwareDevice] = []
    for idx in range(count):
        pci = pci_amd[idx] if idx < len(pci_amd) else {}
        win = win_amd[idx] if idx < len(win_amd) else {}
        rsmi = rocm_gpus[idx] if idx < len(rocm_gpus) else {}
        tdev = torch_rocm_devs[idx] if idx < len(torch_rocm_devs) else {}

        name = str(
            tdev.get("name") or win.get("name") or pci.get("name") or rsmi.get("name") or "AMD GPU"
        )
        mem_gb = float(tdev.get("memory_gb") or rsmi.get("memory_gb") or win.get("ram_gb") or 0.0)
        kind = _classify_amd_gpu(name, mem_gb)
        unified = kind == "igpu"
        if unified and mem_gb <= 2.5 and ram_total_gb > 0.0:
            mem_gb = round(max(mem_gb, ram_total_gb * 0.5), 2)

        driver = str(pci.get("driver") or win.get("driver") or ("amdgpu" if kfd_present else ""))
        missing: list[str] = []
        if os_name == "Linux":
            if not kfd_present and not rocm_avail:
                missing.append("rocm-dkms")
            if not rocm_avail:
                missing.append("torch-rocm")
        elif not rocm_avail:
            missing.append("torch-rocm")

        devices.append(
            HardwareDevice(
                vendor="AMD",
                name=name,
                kind=kind,
                backend="rocm",
                memory_gb=round(mem_gb, 2),
                unified_memory=unified,
                driver=driver,
                available_in_torch=rocm_avail,
                missing_runtime=missing,
                install_hint=_build_install_hint("rocm", os_name, distro_family, missing),
            )
        )
    return devices


def _choose_recommended_backend(devices: list[HardwareDevice], os_name: str) -> BackendKind:
    discrete = [d for d in devices if d.kind == "discrete_gpu"]
    for backend in ("cuda", "rocm", "xpu"):
        if any(d.backend == backend for d in discrete):
            return backend

    igpus = [d for d in devices if d.kind == "igpu"]
    if any(d.backend == "mps" for d in igpus):
        return "mps"
    if any(d.backend == "xpu" for d in igpus):
        return "xpu"
    # On Windows, AMD Radeon iGPUs lack official ROCm PyTorch wheels; prefer CPU unless Linux.
    if os_name == "Linux" and any(d.backend == "rocm" for d in igpus):
        return "rocm"
    return "cpu"


def detect_hardware() -> HardwareProfile:
    """Probe CPU, RAM, OS, GPUs/iGPUs, and active PyTorch backend across Linux, Windows, and macOS."""
    os_name = platform.system() or "Unknown"
    arch = platform.machine() or "unknown"
    os_release = _read_linux_os_release() if os_name == "Linux" else {}
    distro_family = _linux_distro_family(os_release) if os_name == "Linux" else os_name.lower()
    os_distro = _detect_os_distro(os_name, os_release)

    cpu_model, physical_cores, logical_cores, cpu_flags = _probe_cpu(os_name, arch)
    ram_total_gb, ram_available_gb = _probe_ram(os_name)
    torch_info = _inspect_torch()

    pci_devs = _scan_linux_pci_gpus() if os_name == "Linux" else []
    win_controllers = _probe_windows_video_controllers() if os_name == "Windows" else []

    devices: list[HardwareDevice] = []
    devices.extend(_build_nvidia_devices(os_name, distro_family, pci_devs, win_controllers, torch_info))
    devices.extend(
        _build_amd_devices(os_name, distro_family, ram_total_gb, pci_devs, win_controllers, torch_info)
    )
    devices.extend(
        _build_intel_devices(os_name, distro_family, ram_total_gb, pci_devs, win_controllers, torch_info)
    )

    if os_name == "Darwin" and arch.lower() in ("arm64", "aarch64"):
        mps_avail = bool(torch_info.get("mps_available"))
        missing = [] if mps_avail else ["torch-mps"]
        devices.append(
            HardwareDevice(
                vendor="Apple",
                name=f"{cpu_model} GPU" if "gpu" not in cpu_model.lower() else cpu_model,
                kind="igpu",
                backend="mps",
                memory_gb=round(ram_total_gb * 0.75, 2) if ram_total_gb > 0 else 0.0,
                unified_memory=True,
                driver="Metal",
                available_in_torch=mps_avail,
                missing_runtime=missing,
                install_hint=_build_install_hint("mps", os_name, distro_family, missing),
            )
        )

    cpu_vendor = (
        "Apple"
        if "apple" in cpu_model.lower()
        else ("AMD" if "amd" in cpu_model.lower() else ("Intel" if "intel" in cpu_model.lower() else "CPU"))
    )
    devices.append(
        HardwareDevice(
            vendor=cpu_vendor,
            name=cpu_model,
            kind="cpu",
            backend="cpu",
            memory_gb=ram_total_gb,
            unified_memory=True,
            driver="",
            available_in_torch=bool(torch_info.get("installed")),
            missing_runtime=[] if torch_info.get("installed") else ["torch-cpu"],
            install_hint=""
            if torch_info.get("installed")
            else _build_install_hint("cpu", os_name, distro_family, ["torch-cpu"]),
        )
    )

    has_discrete = any(d.kind == "discrete_gpu" for d in devices)
    has_igpu = any(d.kind == "igpu" for d in devices)
    recommended = _choose_recommended_backend(devices, os_name)

    return HardwareProfile(
        os_name=os_name,
        os_distro=os_distro,
        arch=arch,
        cpu_model=cpu_model,
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        cpu_flags=cpu_flags,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
        devices=devices,
        has_discrete_gpu=has_discrete,
        has_igpu=has_igpu,
        recommended_backend=recommended,
        active_torch_version=str(torch_info["version"]),
        active_torch_backend=str(torch_info["backend"]),
    )


def available_execution_targets(profile: HardwareProfile) -> list[ExecutionTarget]:
    """Return available execution targets in priority order for this hardware profile."""
    targets: list[ExecutionTarget] = []
    if profile.has_discrete_gpu:
        targets.append("gpu")
    if profile.has_igpu:
        targets.append("igpu")
    targets.append("cpu")
    targets.append("colab")
    return targets


def _parse_torch_major_minor(version_str: str) -> tuple[int, int]:
    match = re.match(r"^\s*(\d+)\.(\d+)", version_str)
    if not match:
        return (0, 0)
    return int(match.group(1)), int(match.group(2))


def _cuda_supports_bf16(device_name: str, torch_info: dict[str, Any] | None = None) -> bool:
    if torch_info:
        for dev in torch_info.get("cuda_devices", []):
            major = dev.get("major")
            if isinstance(major, int):
                return major >= 8
    low = device_name.lower()
    if any(
        tok in low
        for tok in (
            "t4",
            "v100",
            "p100",
            "p40",
            "k80",
            "gtx",
            "rtx 20",
            "titan rtx",
            "quadro rtx",
        )
    ):
        return False
    if any(
        tok in low
        for tok in (
            "a100",
            "a10",
            "a30",
            "a40",
            "a6000",
            "l4",
            "l40",
            "h100",
            "h200",
            "b100",
            "b200",
            "rtx 30",
            "rtx 40",
            "rtx 50",
            "ampere",
            "ada",
            "hopper",
            "blackwell",
        )
    ):
        return True
    return True


def _pick_device_for_target(
    target: ExecutionTarget, profile: HardwareProfile
) -> tuple[BackendKind, str, HardwareDevice | None]:
    if target == "colab":
        return "cuda", "cuda:0", None

    if target == "gpu":
        discrete = [d for d in profile.devices if d.kind == "discrete_gpu"]
        for backend in ("cuda", "rocm", "xpu"):
            for dev in discrete:
                if dev.backend == backend:
                    dev_str = "xpu:0" if backend == "xpu" else "cuda:0"
                    return backend, dev_str, dev

    if target == "igpu":
        igpus = [d for d in profile.devices if d.kind == "igpu"]
        for backend in ("mps", "xpu", "rocm"):
            for dev in igpus:
                if dev.backend == backend:
                    if backend == "mps":
                        return "mps", "mps", dev
                    if backend == "xpu":
                        return "xpu", "xpu:0", dev
                    return "rocm", "cuda:0", dev

    cpu_dev = next((d for d in profile.devices if d.kind == "cpu"), None)
    return "cpu", "cpu", cpu_dev


def resolve_kernel_stack(
    target: ExecutionTarget,
    profile: HardwareProfile,
    param_count_b: float | None = None,
    batch_size: int | None = None,
) -> KernelStackConfig:
    """Resolve optimal dtype, attention implementation, compile policy, and memory knobs."""
    backend, device_str, device = _pick_device_for_target(target, profile)
    notes: list[str] = []

    if target in ("gpu", "igpu") and backend == "cpu":
        notes.append(f"Requested target '{target}' not detected; falling back to CPU.")

    if target == "colab":
        dtype: Any = "float16"
        avail_mem_gb = 15.0
        notes.append("Colab default profile targets T4 (15 GB, fp16); switch to bf16 on L4/A100.")
    elif backend == "cuda":
        torch_info = _inspect_torch() if profile.active_torch_backend == "cuda" else None
        dev_name = device.name if device else ""
        dtype = "bfloat16" if _cuda_supports_bf16(dev_name, torch_info) else "float16"
        avail_mem_gb = device.memory_gb if (device and device.memory_gb > 0) else 16.0
    elif backend in ("xpu", "rocm"):
        dtype = "bfloat16"
        avail_mem_gb = (
            device.memory_gb
            if (device and device.memory_gb > 0)
            else max(4.0, profile.ram_available_gb * 0.5)
        )
    elif backend == "mps":
        dtype = "float16"
        avail_mem_gb = (
            device.memory_gb
            if (device and device.memory_gb > 0)
            else max(4.0, profile.ram_total_gb * 0.75)
        )
    else:
        has_bf16_cpu = "avx512_bf16" in profile.cpu_flags or "amx" in profile.cpu_flags
        dtype = "bfloat16" if has_bf16_cpu else "float32"
        avail_mem_gb = profile.ram_available_gb or profile.ram_total_gb or 8.0

    torch_ver = _parse_torch_major_minor(profile.active_torch_version)
    use_compile = False
    compile_backend: str | None = None
    if target == "gpu" and backend == "cuda" and profile.os_name == "Linux" and torch_ver >= (2, 6):
        use_compile = True
        compile_backend = "inductor"
    else:
        if profile.os_name == "Windows":
            notes.append("torch.compile disabled by default on Windows (requires MSVC/Triton).")
        elif backend == "mps":
            notes.append("torch.compile disabled on Apple MPS for stability.")
        elif target == "igpu":
            notes.append("torch.compile disabled by default on iGPU unless Triton confirmed safe.")

    est_params = param_count_b if param_count_b is not None else 1.5
    est_lora_gb = est_params * (3.5 if dtype == "float32" else 2.5) + 2.0

    use_grad_ckpt = bool(
        target == "igpu" or avail_mem_gb <= 16.0 or est_lora_gb > avail_mem_gb * 0.8
    )
    use_4bit_qlora = bool(
        backend == "cuda"
        and param_count_b is not None
        and (param_count_b * 3.0 + 3.0) > avail_mem_gb
    )

    if avail_mem_gb >= 40.0:
        micro_bs = 8
    elif avail_mem_gb >= 20.0:
        micro_bs = 4
    elif avail_mem_gb >= 10.0:
        micro_bs = 2
    else:
        micro_bs = 1

    if param_count_b is not None and param_count_b <= 0.6 and avail_mem_gb >= 12.0 and target != "cpu":
        micro_bs = max(micro_bs, 4)

    if batch_size is not None and batch_size > 0:
        micro_bs = max(1, min(micro_bs, batch_size))
        grad_accum = max(1, (batch_size + micro_bs - 1) // micro_bs)
    else:
        grad_accum = max(1, 16 // micro_bs)

    num_threads = max(1, profile.physical_cores)

    return KernelStackConfig(
        target=target,
        backend=backend,
        device_str=device_str,
        dtype=dtype,
        attn_implementation="sdpa",
        use_compile=use_compile,
        compile_backend=compile_backend,
        use_gradient_checkpointing=use_grad_ckpt,
        use_4bit_qlora=use_4bit_qlora,
        recommended_micro_batch_size=micro_bs,
        recommended_grad_accum=grad_accum,
        num_threads=num_threads,
        notes=notes,
    )


def _format_fit(
    param_count_b: float,
    mem_gb: float,
    label: str,
    supports_qlora: bool,
    fp32_only: bool = False,
) -> str:
    lora_gb = param_count_b * (4.5 if fp32_only else 2.5) + 1.5
    qlora_gb = param_count_b * 0.85 + 1.5
    if mem_gb <= 0.0:
        return f"unknown ({label} memory unverified)"
    if lora_gb <= mem_gb * 0.75:
        return f"fits ({lora_gb:.1f}GB est. on {mem_gb:.1f}GB {label})"
    if lora_gb <= mem_gb * 0.95:
        return f"tight - use grad checkpointing ({lora_gb:.1f}GB est. on {mem_gb:.1f}GB {label})"
    if supports_qlora and qlora_gb <= mem_gb * 0.85:
        return f"fits with 4-bit QLoRA (~{qlora_gb:.1f}GB est. on {mem_gb:.1f}GB {label})"
    return f"too_large ({lora_gb:.1f}GB est. > {mem_gb:.1f}GB {label})"


def estimate_model_fit(param_count_b: float | None, profile: HardwareProfile) -> dict[str, str]:
    """Estimate memory fit across local gpu, igpu, cpu, and Colab T4/A100 profiles."""
    if param_count_b is None:
        return {
            "gpu": "unknown (parameter count unavailable)"
            if profile.has_discrete_gpu
            else "unavailable (no discrete GPU)",
            "igpu": "unknown (parameter count unavailable)"
            if profile.has_igpu
            else "unavailable (no iGPU)",
            "cpu": "unknown (parameter count unavailable)",
            "colab_t4": "unknown (parameter count unavailable)",
            "colab_a100": "unknown (parameter count unavailable)",
        }

    discrete_devs = [d for d in profile.devices if d.kind == "discrete_gpu"]
    if discrete_devs:
        best_d = max(discrete_devs, key=lambda d: d.memory_gb)
        gpu_fit = _format_fit(
            param_count_b,
            best_d.memory_gb,
            "VRAM",
            supports_qlora=(best_d.backend == "cuda"),
        )
    else:
        gpu_fit = "unavailable (no discrete GPU)"

    igpus = [d for d in profile.devices if d.kind == "igpu"]
    if igpus:
        best_i = max(igpus, key=lambda d: d.memory_gb)
        igpu_fit = _format_fit(
            param_count_b,
            best_i.memory_gb,
            "shared memory" if best_i.unified_memory else "VRAM",
            supports_qlora=False,
        )
    else:
        igpu_fit = "unavailable (no iGPU)"

    cpu_bf16 = "avx512_bf16" in profile.cpu_flags or "amx" in profile.cpu_flags
    cpu_mem = profile.ram_available_gb or profile.ram_total_gb
    cpu_fit = _format_fit(
        param_count_b,
        cpu_mem,
        "RAM",
        supports_qlora=False,
        fp32_only=not cpu_bf16,
    )

    return {
        "gpu": gpu_fit,
        "igpu": igpu_fit,
        "cpu": cpu_fit,
        "colab_t4": _format_fit(param_count_b, 15.0, "T4 VRAM", supports_qlora=True),
        "colab_a100": _format_fit(param_count_b, 40.0, "A100 VRAM", supports_qlora=True),
    }


def apply_runtime_env(kernel: KernelStackConfig) -> None:
    """Apply safe cross-platform threading and backend environment variables."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if kernel.backend == "cpu":
        threads = str(max(1, kernel.num_threads))
        os.environ["OMP_NUM_THREADS"] = threads
        os.environ["MKL_NUM_THREADS"] = threads
        try:
            import torch

            if callable(getattr(torch, "set_num_threads", None)):
                torch.set_num_threads(max(1, kernel.num_threads))
        except ImportError:
            pass
    elif kernel.backend == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    elif kernel.backend == "xpu":
        os.environ.setdefault("SYCL_CACHE_PERSISTENT", "1")


def smoke_test_device(device_str: str, dtype_str: str = "bfloat16") -> tuple[bool, str]:
    """Execute a live torch.mm matmul check on the target device and dtype."""
    try:
        import torch
    except ImportError:
        return False, "PyTorch is not installed."

    dtype_map = {
        "bfloat16": getattr(torch, "bfloat16", torch.float32),
        "float16": getattr(torch, "float16", torch.float32),
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)

    dev_type = device_str.split(":", 1)[0].lower()
    if dev_type == "cuda":
        cuda_mod = getattr(torch, "cuda", None)
        if not (cuda_mod and cuda_mod.is_available()):
            return False, f"CUDA is not available in PyTorch for device '{device_str}'."
    elif dev_type == "xpu":
        xpu_mod = getattr(torch, "xpu", None)
        if not (xpu_mod and xpu_mod.is_available()):
            return False, f"Intel XPU is not available in PyTorch for device '{device_str}'."
    elif dev_type == "mps":
        mps_mod = getattr(getattr(torch, "backends", None), "mps", None)
        if not (mps_mod and mps_mod.is_available()):
            return False, "Apple MPS is not available in PyTorch."

    try:
        t0 = time.perf_counter()
        a = torch.randn((32, 32), device=device_str, dtype=dtype)
        b = torch.randn((32, 32), device=device_str, dtype=dtype)
        c = torch.mm(a, b)
        if dev_type == "cuda" and hasattr(torch.cuda, "synchronize"):
            torch.cuda.synchronize()
        elif dev_type == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "synchronize"):
            torch.xpu.synchronize()
        elif dev_type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        val = float(c[0, 0].item())
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if val != val:  # NaN check without math import
            return False, f"Smoke test produced NaN on {device_str} ({dtype_str})."
        return True, f"OK: {device_str} {dtype_str} matmul ({elapsed_ms:.1f}ms)"
    except Exception as exc:
        return False, f"Smoke test failed on {device_str} ({dtype_str}): {exc}"


__all__ = [
    "apply_runtime_env",
    "available_execution_targets",
    "detect_hardware",
    "estimate_model_fit",
    "resolve_kernel_stack",
    "smoke_test_device",
]
