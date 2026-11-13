#!/usr/bin/env sh
# Cross-platform POSIX installer for Decision Forge (Linux & macOS).

set -eu

BACKEND="auto"
AUTO_YES=0
SKIP_OS_DEPS=0
PYTHON_VERSION="3.12"

usage() {
    cat <<'EOF'
Usage: ./install.sh [OPTIONS]

Options:
  --backend <auto|cuda|rocm|xpu|mps|cpu>  Override detected accelerator backend (default: auto)
  --python <version>                      Python version for uv venv (default: 3.12)
  -y, --yes                               Non-interactive mode; confirm OS runtime installs
  --no-os-deps                            Skip OS compute runtime installation
  -h, --help                              Show this help message
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --backend)
            [ "$#" -ge 2 ] || { echo "Missing value for --backend" >&2; exit 2; }
            BACKEND="$2"
            shift 2
            ;;
        --python)
            [ "$#" -ge 2 ] || { echo "Missing value for --python" >&2; exit 2; }
            PYTHON_VERSION="$2"
            shift 2
            ;;
        -y|--yes)
            AUTO_YES=1
            shift
            ;;
        --no-os-deps)
            SKIP_OS_DEPS=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
DISTRO_FAMILY="unknown"

detect_distro_family() {
    if [ "$OS_NAME" = "Darwin" ]; then
        DISTRO_FAMILY="macos"
        return
    fi
    if [ -r /etc/os-release ]; then
        # shellcheck disable=SC1091
        os_tokens="$(. /etc/os-release 2>/dev/null && printf '%s %s %s' "${ID:-}" "${ID_LIKE:-}" "${NAME:-}" | tr '[:upper:]' '[:lower:]')"
        case "$os_tokens" in
            *cachyos*|*arch*|*endeavouros*|*manjaro*|*garuda*)
                DISTRO_FAMILY="arch"
                ;;
            *ubuntu*|*debian*|*pop*|*linuxmint*|*kali*)
                DISTRO_FAMILY="debian"
                ;;
            *fedora*|*rhel*|*centos*|*rocky*|*almalinux*|*nobara*)
                DISTRO_FAMILY="fedora"
                ;;
            *suse*|*opensuse*|*sles*|*tumbleweed*|*leap*)
                DISTRO_FAMILY="suse"
                ;;
            *)
                DISTRO_FAMILY="linux"
                ;;
        esac
    fi
}

run_privileged() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        echo "Root privileges required to run: $*" >&2
        return 1
    fi
}

confirm_prompt() {
    prompt_msg="$1"
    if [ "$AUTO_YES" -eq 1 ]; then
        return 0
    fi
    if [ ! -t 0 ]; then
        echo "Non-interactive shell without --yes; skipping: $prompt_msg"
        return 1
    fi
    printf "%s [Y/n] " "$prompt_msg"
    read -r reply || return 1
    case "$reply" in
        ""|[Yy]|[Yy][Ee][Ss]) return 0 ;;
        *) return 1 ;;
    esac
}

has_pci_vendor() {
    target_vendor="$1"
    for dev in /sys/bus/pci/devices/*; do
        [ -r "$dev/vendor" ] || continue
        v="$(tr '[:upper:]' '[:lower:]' < "$dev/vendor" 2>/dev/null || true)"
        c="$(tr '[:upper:]' '[:lower:]' < "$dev/class" 2>/dev/null || true)"
        if [ "$v" = "$target_vendor" ]; then
            case "$c" in
                0x03*|0x12*) return 0 ;;
            esac
        fi
    done
    for card in /sys/class/drm/card*; do
        [ -r "$card/device/vendor" ] || continue
        v="$(tr '[:upper:]' '[:lower:]' < "$card/device/vendor" 2>/dev/null || true)"
        if [ "$v" = "$target_vendor" ]; then
            return 0
        fi
    done
    return 1
}

detect_backend() {
    if [ "$BACKEND" != "auto" ]; then
        return
    fi
    if [ "$OS_NAME" = "Darwin" ]; then
        if [ "$ARCH_NAME" = "arm64" ] || [ "$ARCH_NAME" = "aarch64" ]; then
            BACKEND="mps"
        else
            BACKEND="cpu"
        fi
        return
    fi

    if command -v nvidia-smi >/dev/null 2>&1 || has_pci_vendor "0x10de"; then
        BACKEND="cuda"
        return
    fi
    if [ -e /dev/kfd ] || command -v rocm-smi >/dev/null 2>&1 || has_pci_vendor "0x1002"; then
        BACKEND="rocm"
        return
    fi
    if has_pci_vendor "0x8086"; then
        BACKEND="xpu"
        return
    fi
    if command -v lspci >/dev/null 2>&1; then
        lspci_out="$(lspci -nn 2>/dev/null || true)"
        if printf '%s\n' "$lspci_out" | grep -qiE '(vga|3d|display).*10de:'; then
            BACKEND="cuda"
            return
        fi
        if printf '%s\n' "$lspci_out" | grep -qiE '(vga|3d|display).*1002:'; then
            BACKEND="rocm"
            return
        fi
        if printf '%s\n' "$lspci_out" | grep -qiE '(vga|3d|display).*8086:'; then
            BACKEND="xpu"
            return
        fi
    fi
    BACKEND="cpu"
}

has_libze_loader() {
    if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q "libze_loader\.so"; then
        return 0
    fi
    for candidate in /usr/lib/libze_loader.so.1 /usr/lib64/libze_loader.so.1 /usr/lib/x86_64-linux-gnu/libze_loader.so.1; do
        if [ -e "$candidate" ]; then
            return 0
        fi
    done
    return 1
}

install_os_compute_runtime() {
    [ "$SKIP_OS_DEPS" -eq 0 ] || return 0
    [ "$OS_NAME" = "Linux" ] || return 0

    case "$BACKEND" in
        xpu)
            if ! has_libze_loader; then
                echo "Intel XPU hardware detected, but Level Zero / Intel compute runtime is missing."
                case "$DISTRO_FAMILY" in
                    arch)
                        if confirm_prompt "Install intel-compute-runtime, level-zero-loader, level-zero-headers via pacman?"; then
                            if [ "$AUTO_YES" -eq 1 ]; then
                                run_privileged pacman -S --needed --noconfirm intel-compute-runtime level-zero-loader level-zero-headers
                            else
                                run_privileged pacman -S --needed intel-compute-runtime level-zero-loader level-zero-headers
                            fi
                        fi
                        ;;
                    debian)
                        if confirm_prompt "Install intel-opencl-icd, libze1, libze-dev via apt-get?"; then
                            run_privileged apt-get update
                            run_privileged apt-get install -y intel-opencl-icd libze1 libze-dev
                        fi
                        ;;
                    fedora)
                        if confirm_prompt "Install intel-compute-runtime, level-zero, level-zero-devel via dnf?"; then
                            run_privileged dnf install -y intel-compute-runtime level-zero level-zero-devel
                        fi
                        ;;
                    suse)
                        if confirm_prompt "Install intel-opencl, level-zero via zypper?"; then
                            run_privileged zypper install -y intel-opencl level-zero
                        fi
                        ;;
                    *)
                        echo "Please install intel-compute-runtime and level-zero-loader for your Linux distribution."
                        ;;
                esac
            fi
            ;;
        rocm)
            if [ ! -e /dev/kfd ] || ! command -v rocm-smi >/dev/null 2>&1; then
                echo "AMD GPU detected; checking ROCm runtime (/dev/kfd, rocm-smi)."
                case "$DISTRO_FAMILY" in
                    arch)
                        if confirm_prompt "Install rocm-hip-sdk and rocm-smi-lib via pacman?"; then
                            if [ "$AUTO_YES" -eq 1 ]; then
                                run_privileged pacman -S --needed --noconfirm rocm-hip-sdk rocm-smi-lib
                            else
                                run_privileged pacman -S --needed rocm-hip-sdk rocm-smi-lib
                            fi
                        fi
                        ;;
                    fedora)
                        if confirm_prompt "Install rocm-hip and rocm-smi via dnf?"; then
                            run_privileged dnf install -y rocm-hip rocm-smi
                        fi
                        ;;
                    debian)
                        echo "Note: For Ubuntu/Debian ROCm, ensure amdgpu-dkms / rocm-libs are installed."
                        ;;
                    suse)
                        echo "Note: For openSUSE ROCm, install ROCm HIP packages via zypper."
                        ;;
                esac
            fi
            ;;
        cuda)
            if ! command -v nvidia-smi >/dev/null 2>&1; then
                echo "NVIDIA GPU detected, but nvidia-smi is unavailable. Ensure proprietary NVIDIA drivers are installed."
            fi
            ;;
    esac
}

ensure_uv() {
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if command -v uv >/dev/null 2>&1; then
        return 0
    fi
    echo "Installing uv package manager..."
    if [ "$OS_NAME" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
        brew install uv
    elif command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        echo "Neither curl, wget, nor brew is available to install uv." >&2
        exit 1
    fi
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    command -v uv >/dev/null 2>&1 || { echo "Failed to locate uv after installation." >&2; exit 1; }
}

resolve_uv_torch_backend() {
    case "$BACKEND" in
        xpu)
            printf '%s' "xpu"
            ;;
        rocm)
            printf '%s' "rocm6.3"
            ;;
        cuda)
            if command -v nvidia-smi >/dev/null 2>&1; then
                drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 | cut -d. -f1 || true)"
                case "$drv" in
                    ''|*[!0-9]*) printf '%s' "cu126" ;;
                    *)
                        if [ "$drv" -ge 560 ]; then
                            printf '%s' "cu126"
                        elif [ "$drv" -ge 550 ]; then
                            printf '%s' "cu124"
                        else
                            printf '%s' "cu121"
                        fi
                        ;;
                esac
            else
                printf '%s' "cu126"
            fi
            ;;
        mps)
            printf '%s' "auto"
            ;;
        cpu|*)
            printf '%s' "cpu"
            ;;
    esac
}

detect_distro_family
detect_backend

echo "Decision Forge Installer"
echo "  OS / Arch : ${OS_NAME} (${DISTRO_FAMILY}) / ${ARCH_NAME}"
echo "  Backend   : ${BACKEND}"

install_os_compute_runtime
ensure_uv

UV_TORCH_BACKEND="$(resolve_uv_torch_backend)"
export UV_TORCH_BACKEND
echo "  UV_TORCH_BACKEND : ${UV_TORCH_BACKEND}"

uv venv --python "$PYTHON_VERSION" --allow-existing .venv

if [ "$BACKEND" = "cuda" ] && [ "$OS_NAME" = "Linux" ]; then
    uv pip install --python .venv/bin/python --reinstall-package torch -e ".[cuda]"
else
    uv pip install --python .venv/bin/python --reinstall-package torch -e .
fi

if [ "$OS_NAME" = "Linux" ] && [ "$BACKEND" = "xpu" ]; then
    echo "Installing pytorch-triton-xpu for Linux Intel XPU..."
    uv pip install --python .venv/bin/python --torch-backend xpu pytorch-triton-xpu
fi

echo "Running live PyTorch device smoke test..."
.venv/bin/python - <<'PY'
from decision_forge.hardware import (
    available_execution_targets,
    detect_hardware,
    resolve_kernel_stack,
    smoke_test_device,
)

profile = detect_hardware()
targets = available_execution_targets(profile)
preferred_target = targets[0] if targets else "cpu"
kernel = resolve_kernel_stack(preferred_target, profile)
ok, msg = smoke_test_device(kernel.device_str, kernel.dtype)
print(f"Target={preferred_target} Device={kernel.device_str} Dtype={kernel.dtype} -> {msg}")
if not ok and kernel.device_str != "cpu":
    cpu_ok, cpu_msg = smoke_test_device("cpu", "float32")
    print(f"CPU fallback check -> {cpu_msg}")
PY

if command -v cargo >/dev/null 2>&1; then
    echo "Building the terminal studio (tui/)..."
    cargo build --release --manifest-path tui/Cargo.toml
else
    echo "cargo not found, so the terminal studio was skipped. Install Rust from https://rustup.rs and"
    echo "rerun this script to get it; until then 'decide' opens the prompt wizard."
fi

echo "Running decide-setup --check..."
.venv/bin/decide-setup --check
