"""Cross-platform GGUF export via llama.cpp (`convert_hf_to_gguf.py` and optional `llama-quantize`)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from decision_forge.types import sanitize_name

_CONVERTER_NAMES = ("convert_hf_to_gguf.py", "convert-hf-to-gguf.py")
_DIRECT_OUTTYPES = frozenset({"f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"})


def _candidate_llama_dirs() -> list[Path]:
    dirs: list[Path] = []
    env_dir = os.environ.get("LLAMA_CPP_DIR")
    if env_dir:
        dirs.append(Path(env_dir).expanduser())

    home = Path.home()
    dirs.extend(
        [
            Path.cwd() / "llama.cpp",
            home / "llama.cpp",
            home / "src" / "llama.cpp",
            home / "dev" / "llama.cpp",
            home / "projects" / "llama.cpp",
            home / "Documents" / "llama.cpp",
            home / "tools" / "llama.cpp",
        ]
    )

    if sys.platform == "win32":
        dirs.extend(
            [
                Path("C:/llama.cpp"),
                home / "AppData" / "Local" / "llama.cpp",
            ]
        )
    else:
        dirs.extend(
            [
                Path("/opt/llama.cpp"),
                Path("/usr/local/share/llama.cpp"),
            ]
        )
        if sys.platform == "darwin":
            dirs.append(home / "Library" / "Application Support" / "llama.cpp")
            dirs.append(Path("/opt/homebrew/share/llama.cpp"))

    return dirs


def find_llama_cpp_converter() -> Path | None:
    """Locate `convert_hf_to_gguf.py` across `LLAMA_CPP_DIR`, `PATH`, and common clone directories."""
    for base in _candidate_llama_dirs():
        for name in _CONVERTER_NAMES:
            for sub in (base / name, base / "scripts" / name):
                if sub.is_file():
                    return sub.resolve()

    for cmd in (*_CONVERTER_NAMES, "llama-convert-hf-to-gguf"):
        found = shutil.which(cmd)
        if found:
            path = Path(found)
            if path.is_file():
                return path.resolve()

    for bin_name in ("llama-quantize", "llama-cli"):
        found_bin = shutil.which(bin_name)
        if found_bin:
            parent = Path(found_bin).resolve().parent
            for ancestor in (parent, parent.parent, parent.parent.parent):
                for name in _CONVERTER_NAMES:
                    candidate = ancestor / name
                    if candidate.is_file():
                        return candidate.resolve()

    return None


def _find_llama_quantize(converter: Path) -> Path | None:
    names = ("llama-quantize.exe", "llama-quantize") if sys.platform == "win32" else ("llama-quantize",)
    for name in names:
        found = shutil.which(name)
        if found and Path(found).is_file():
            return Path(found).resolve()
        for rel in (
            converter.parent / name,
            converter.parent / "build" / "bin" / name,
            converter.parent / "build" / "bin" / "Release" / name,
        ):
            if rel.is_file():
                return rel.resolve()
    return None


def _merge_lora_to_dir(adapter_dir: Path, target_dir: Path) -> None:
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoPeftModelForCausalLM.from_pretrained(str(adapter_dir))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(target_dir), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(target_dir))


def export_gguf(
    model_dir: Path,
    output_path: Path | None = None,
    outtype: str = "q8_0",
) -> tuple[bool, str]:
    """Export a saved Hugging Face causal decision model directory to GGUF via llama.cpp.

    Returns `(True, output_file_path)` on success, or `(False, diagnostic_message)` on failure.
    """
    model_path = Path(model_dir)
    if not model_path.is_dir():
        return False, f"Model directory not found: {model_path}"

    converter = find_llama_cpp_converter()
    if converter is None:
        return (
            False,
            "llama.cpp converter (convert_hf_to_gguf.py) not found. "
            "Set LLAMA_CPP_DIR or clone https://github.com/ggml-org/llama.cpp.",
        )

    outtype_norm = outtype.strip().lower()
    if not outtype_norm:
        return False, "outtype must be a non-empty quantization or dtype string (e.g. 'q8_0', 'f16')."

    quantizer: Path | None = None
    if outtype_norm not in _DIRECT_OUTTYPES:
        quantizer = _find_llama_quantize(converter)
        if quantizer is None:
            return (
                False,
                f"Outtype {outtype!r} requires `llama-quantize`, which was not found. "
                f"Direct converter supports: {', '.join(sorted(_DIRECT_OUTTYPES))}.",
            )

    target_output = (
        Path(output_path)
        if output_path is not None
        else model_path / f"{sanitize_name(model_path.name)}-{sanitize_name(outtype_norm)}.gguf"
    )
    target_output.parent.mkdir(parents=True, exist_ok=True)

    tmp_merge_dir: tempfile.TemporaryDirectory[str] | None = None
    tmp_quant_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        merged_sub = model_path / "merged"
        if (merged_sub / "config.json").is_file():
            source_dir = merged_sub
        elif (model_path / "config.json").is_file():
            source_dir = model_path
        elif (model_path / "adapter_config.json").is_file():
            tmp_merge_dir = tempfile.TemporaryDirectory(prefix="df_merge_")
            source_dir = Path(tmp_merge_dir.name)
            try:
                _merge_lora_to_dir(model_path, source_dir)
            except Exception as exc:
                return False, f"Failed to merge LoRA adapter before GGUF export: {exc}"
        else:
            return False, f"Missing config.json or adapter_config.json in {model_path}"

        if quantizer is None:
            cmd = [
                sys.executable,
                str(converter),
                str(source_dir),
                "--outfile",
                str(target_output),
                "--outtype",
                outtype_norm,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0 or not target_output.is_file():
                detail = (proc.stderr or proc.stdout or "").strip()
                return False, f"GGUF export failed (exit {proc.returncode}): {detail}"
            return True, str(target_output)

        tmp_quant_dir = tempfile.TemporaryDirectory(prefix="df_f16_")
        f16_path = Path(tmp_quant_dir.name) / "model-f16.gguf"
        conv_cmd = [
            sys.executable,
            str(converter),
            str(source_dir),
            "--outfile",
            str(f16_path),
            "--outtype",
            "f16",
        ]
        conv_proc = subprocess.run(conv_cmd, capture_output=True, text=True, check=False)
        if conv_proc.returncode != 0 or not f16_path.is_file():
            detail = (conv_proc.stderr or conv_proc.stdout or "").strip()
            return False, f"GGUF f16 conversion failed (exit {conv_proc.returncode}): {detail}"

        q_cmd = [str(quantizer), str(f16_path), str(target_output), outtype.upper()]
        q_proc = subprocess.run(q_cmd, capture_output=True, text=True, check=False)
        if q_proc.returncode != 0 or not target_output.is_file():
            detail = (q_proc.stderr or q_proc.stdout or "").strip()
            return False, f"llama-quantize failed (exit {q_proc.returncode}): {detail}"
        return True, str(target_output)
    finally:
        if tmp_merge_dir is not None:
            tmp_merge_dir.cleanup()
        if tmp_quant_dir is not None:
            tmp_quant_dir.cleanup()
