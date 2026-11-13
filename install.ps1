<#
.SYNOPSIS
    Cross-platform Windows 10/11 installer for Decision Forge.
.DESCRIPTION
    Detects NVIDIA CUDA, Intel Arc XPU, AMD Radeon, or CPU hardware, installs uv
    if missing, creates a Python 3.12 virtual environment, installs Decision Forge
    with the optimal PyTorch backend, and runs a live smoke test.
#>
[CmdletBinding()]
param(
    [ValidateSet("auto", "cuda", "xpu", "cpu")]
    [string]$Backend = "auto",
    [string]$PythonVersion = "3.12",
    [switch]$Yes
)

$ErrorActionPreference = "Stop"

function Ensure-Uv {
    $localBin = Join-Path $env:USERPROFILE ".local\bin"
    $cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
    foreach ($dir in @($localBin, $cargoBin)) {
        if ((Test-Path $dir) -and ($env:Path -notlike "*$dir*")) {
            $env:Path = "$dir;$env:Path"
        }
    }

    if (Get-Command uv -ErrorAction SilentlyContinue) {
        return
    }

    Write-Host "Installing uv package manager..." -ForegroundColor Cyan
    powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"

    foreach ($dir in @($localBin, $cargoBin)) {
        if ((Test-Path $dir) -and ($env:Path -notlike "*$dir*")) {
            $env:Path = "$dir;$env:Path"
        }
    }

    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "Failed to locate 'uv' on PATH after installation."
    }
}

function Get-VideoControllers {
    try {
        $controllers = Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop
        return @($controllers)
    } catch {
        return @()
    }
}

function Resolve-TargetBackend {
    param([string]$Requested, [array]$Controllers)

    if ($Requested -ne "auto") {
        return $Requested
    }

    $hasNvidiaSmi = [bool](Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)
    $hasNvidiaPci = $Controllers | Where-Object {
        ($_.Name -match "NVIDIA") -or ($_.PNPDeviceID -match "VEN_10DE")
    }
    if ($hasNvidiaSmi -or $hasNvidiaPci) {
        return "cuda"
    }

    $hasIntelGpu = $Controllers | Where-Object {
        ($_.Name -match "Intel") -or ($_.PNPDeviceID -match "VEN_8086")
    }
    if ($hasIntelGpu) {
        return "xpu"
    }

    $hasAmdGpu = $Controllers | Where-Object {
        ($_.Name -match "AMD|Radeon") -or ($_.PNPDeviceID -match "VEN_1002")
    }
    if ($hasAmdGpu) {
        Write-Warning "AMD Radeon GPU detected on Windows. PyTorch ROCm wheels target Linux; falling back to CPU backend on Windows."
        return "cpu"
    }

    return "cpu"
}

function Resolve-UvTorchBackend {
    param([string]$Target)

    if ($Target -eq "xpu") {
        return "xpu"
    }
    if ($Target -eq "cuda") {
        $smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
        if ($smi) {
            try {
                $drvLine = (& $smi.Source --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
                if ($drvLine) {
                    $major = [int](($drvLine.Trim() -split "\.")[0])
                    if ($major -ge 560) { return "cu126" }
                    if ($major -ge 550) { return "cu124" }
                    return "cu121"
                }
            } catch {
            }
        }
        return "cu126"
    }
    return "cpu"
}

function Show-IntelWindowsDriverHint {
    param([array]$Controllers)

    $intelControllers = $Controllers | Where-Object {
        ($_.Name -match "Intel") -or ($_.PNPDeviceID -match "VEN_8086")
    }
    $zeDll = Join-Path $env:SystemRoot "System32\ze_loader.dll"
    $hasZe = Test-Path $zeDll

    foreach ($gpu in $intelControllers) {
        Write-Host ("  Intel GPU  : {0} (Driver {1})" -f $gpu.Name, $gpu.DriverVersion) -ForegroundColor Gray
    }
    if (-not $hasZe) {
        Write-Warning (
            "Level Zero loader (ze_loader.dll) was not found in System32. " +
            "For Intel Arc / Core Ultra XPU acceleration on Windows, ensure the latest Intel Graphics Driver is installed: " +
            "https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html"
        )
    }
}

$controllers = Get-VideoControllers
$resolvedBackend = Resolve-TargetBackend -Requested $Backend -Controllers $controllers
$uvTorchBackend = Resolve-UvTorchBackend -Target $resolvedBackend

Write-Host "Decision Forge Windows Installer" -ForegroundColor Cyan
Write-Host ("  Backend          : {0}" -f $resolvedBackend)
Write-Host ("  UV_TORCH_BACKEND : {0}" -f $uvTorchBackend)

if ($resolvedBackend -eq "xpu") {
    Show-IntelWindowsDriverHint -Controllers $controllers
} elseif ($resolvedBackend -eq "cuda" -and -not (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) {
    Write-Warning "NVIDIA GPU detected, but nvidia-smi.exe was not found. Install the latest NVIDIA Game Ready / Studio Driver: https://www.nvidia.com/drivers"
}

Ensure-Uv

uv venv --python $PythonVersion --allow-existing .venv
$venvPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"
$decideSetup = Join-Path (Get-Location) ".venv\Scripts\decide-setup.exe"

$env:UV_TORCH_BACKEND = $uvTorchBackend
# Note: pytorch-triton-xpu is Linux-only and intentionally NOT installed on Windows.
uv pip install --python $venvPython --reinstall-package torch -e . --torch-backend $uvTorchBackend

Write-Host "Running live PyTorch device smoke test..." -ForegroundColor Cyan
$smokeScript = @'
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
    _, cpu_msg = smoke_test_device("cpu", "float32")
    print(f"CPU fallback check -> {cpu_msg}")
'@
& $venvPython -c $smokeScript

if (Get-Command cargo -ErrorAction SilentlyContinue) {
    Write-Host "Building the terminal studio (tui/)..." -ForegroundColor Cyan
    cargo build --release --manifest-path tui/Cargo.toml
} else {
    Write-Host "cargo not found, so the terminal studio was skipped. Install Rust from https://rustup.rs and rerun this script to get it; until then 'decide' opens the prompt wizard." -ForegroundColor Yellow
}

Write-Host "Running decide-setup --check..." -ForegroundColor Cyan
if (Test-Path $decideSetup) {
    & $decideSetup --check
} else {
    & $venvPython -m decision_forge.bootstrap --check
}
