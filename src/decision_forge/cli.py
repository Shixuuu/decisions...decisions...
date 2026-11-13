"""Cross-platform CLI entrypoint (`decide`) for Decision Forge."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from decision_forge.bench.evaluator import run_decidebench
from decision_forge.bootstrap import check_environment_health
from decision_forge.bootstrap import main as bootstrap_main
from decision_forge.colab import generate_colab_notebook
from decision_forge.constants import DEFAULT_RUNS_DIR
from decision_forge.export import export_gguf
from decision_forge.hardware import available_execution_targets, detect_hardware
from decision_forge.hf_models import inspect_model_metadata
from decision_forge.serve import DecisionHTTPServer, serve_model
from decision_forge.tui import get_box_style, make_console
from decision_forge.tui.wizard import _load_decision_engine_compat, execute_session, run_wizard
from decision_forge.types import (
    CalibrationConfig,
    DatasetSpec,
    DecideBenchConfig,
    ExecutionTarget,
    ModelInfo,
    TrainConfig,
    WizardSessionConfig,
    load_json,
    make_run_id,
    sanitize_name,
)


def _build_session_from_args(args: argparse.Namespace, default_target: ExecutionTarget) -> WizardSessionConfig:
    if getattr(args, "config", None):
        cfg_data = load_json(Path(args.config))
        session = WizardSessionConfig.from_dict(cfg_data)
        if getattr(args, "model", None):
            session.model = _safe_inspect_cli_model(args.model)
        if getattr(args, "target", None):
            session.execution_target = args.target
        if getattr(args, "out", None) and session.execution_target != "colab":
            session.output_dir = str(args.out)
        return session

    model_repo = getattr(args, "model", None) or "Qwen/Qwen2.5-0.5B-Instruct"
    model_info = _safe_inspect_cli_model(model_repo)

    if getattr(args, "zero_shot", False):
        dataset_spec = DatasetSpec(mode="zero_shot", primitive=args.primitive)
    elif getattr(args, "general_mix", False):
        mix_raw = getattr(args, "mix_datasets", None) or ""
        mix_list = [x.strip() for x in mix_raw.split(",") if x.strip()] if mix_raw else []
        dataset_spec = DatasetSpec(
            mode="auto_general_mix",
            mix_datasets=mix_list,
            primitive=args.primitive,
            max_train_samples=getattr(args, "max_train_samples", 2000),
            max_val_samples=getattr(args, "max_val_samples", 400),
        )
    elif getattr(args, "local_file", None):
        dataset_spec = DatasetSpec(
            mode="local_file",
            local_path=str(Path(args.local_file).expanduser()),
            state_col=getattr(args, "state_col", None),
            label_col=getattr(args, "label_col", None),
            options_col=getattr(args, "options_col", None),
            question_col=getattr(args, "question_col", None),
            second_text_col=getattr(args, "second_text_col", None),
            primitive=args.primitive,
            max_train_samples=getattr(args, "max_train_samples", 2000),
            max_val_samples=getattr(args, "max_val_samples", 400),
        )
    elif getattr(args, "dataset", None):
        dataset_spec = DatasetSpec(
            mode="hf_dataset",
            repo_id=args.dataset,
            config_name=getattr(args, "dataset_config", None),
            split=getattr(args, "split", "train"),
            state_col=getattr(args, "state_col", None),
            label_col=getattr(args, "label_col", None),
            options_col=getattr(args, "options_col", None),
            question_col=getattr(args, "question_col", None),
            second_text_col=getattr(args, "second_text_col", None),
            primitive=args.primitive,
            max_train_samples=getattr(args, "max_train_samples", 2000),
            max_val_samples=getattr(args, "max_val_samples", 400),
        )
    else:
        dataset_spec = DatasetSpec(mode="zero_shot", primitive=args.primitive)

    method = "zero_shot" if dataset_spec.mode == "zero_shot" else args.method
    attention_mode = "encoder" if model_info.architecture_family == "encoder" else args.attention
    train_cfg = TrainConfig(
        method=method,
        attention_mode=attention_mode,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        grad_accum_steps=getattr(args, "grad_accum", 4),
        max_seq_len=args.max_seq_len,
        disable_thinking=(model_info.architecture_family != "encoder"),
        pride_debiasing=(method == "zero_shot"),
    )
    cal_cfg = CalibrationConfig(enabled=not getattr(args, "no_calibration", False))
    bench_cfg = DecideBenchConfig(
        enabled=bool(getattr(args, "eval_decidebench", False)),
        version=getattr(args, "bench_version", "v2"),
        limit=getattr(args, "bench_limit", None),
    )

    run_id = make_run_id(model_info.repo_id)
    out_arg = getattr(args, "out", None)
    if out_arg and default_target != "colab":
        output_dir = str(Path(out_arg))
    else:
        output_dir = str(Path(DEFAULT_RUNS_DIR) / sanitize_name(run_id))

    target: ExecutionTarget = getattr(args, "target", None) or default_target
    return WizardSessionConfig(
        run_id=run_id,
        output_dir=output_dir,
        model=model_info,
        dataset=dataset_spec,
        train=train_cfg,
        calibration=cal_cfg,
        decidebench=bench_cfg,
        export_gguf=bool(getattr(args, "export_gguf", False)),
        start_server_after=bool(getattr(args, "serve", False)),
        server_port=int(getattr(args, "port", 8765)),
        execution_target=target,
    )


def _safe_inspect_cli_model(repo_or_path: str) -> ModelInfo:
    cand = Path(repo_or_path).expanduser()
    if cand.is_dir():
        try:
            return inspect_model_metadata(repo_id=cand.name, local_dir=str(cand))
        except Exception:
            return ModelInfo(repo_id=cand.name, local_path=str(cand))
    try:
        return inspect_model_metadata(repo_id=repo_or_path)
    except Exception:
        low = repo_or_path.lower()
        arch = "encoder" if "bert" in low else "decoder"
        return ModelInfo(
            repo_id=repo_or_path,
            architecture_family=arch,  # type: ignore[arg-type]
            supports_thinking_toggle="qwen3" in low,
        )


def _cmd_doctor(args: argparse.Namespace) -> int:
    profile = detect_hardware()
    health = check_environment_health(profile)
    if args.json:
        print(json.dumps(health, indent=2))
        return 0

    console = make_console()
    box_style = get_box_style(console.file)
    from rich.panel import Panel
    from rich.table import Table

    summary = (
        f"[bold]OS:[/bold] {profile.os_distro} ({profile.os_name} / {profile.arch})\n"
        f"[bold]CPU:[/bold] {profile.cpu_model} ({profile.physical_cores}c / {profile.logical_cores}t)\n"
        f"[bold]RAM:[/bold] {profile.ram_available_gb:.1f} GB avail / {profile.ram_total_gb:.1f} GB total\n"
        f"[bold]PyTorch:[/bold] {profile.active_torch_version} (active={profile.active_torch_backend}, "
        f"recommended={profile.recommended_backend})\n"
        f"[bold]Healthy:[/bold] {health['healthy']} | [bold]Targets:[/bold] {', '.join(available_execution_targets(profile))}"
    )
    console.print(Panel(summary, title="Decision Forge Doctor", border_style="cyan", box=box_style))

    table = Table(title="Compute Devices", box=box_style)
    table.add_column("Vendor", style="bold")
    table.add_column("Device")
    table.add_column("Kind")
    table.add_column("Backend")
    table.add_column("Memory")
    table.add_column("Torch")
    table.add_column("Missing Runtime")
    for dev in profile.devices:
        mem = f"{dev.memory_gb:.1f} GB" if dev.memory_gb > 0 else "shared"
        table.add_row(
            dev.vendor,
            dev.name,
            dev.kind,
            dev.backend,
            mem,
            "yes" if dev.available_in_torch else "no",
            ", ".join(dev.missing_runtime) or "-",
        )
    console.print(table)

    if health.get("notes") or health.get("install_commands"):
        lines = list(health.get("notes", []))
        if health.get("install_commands"):
            lines.append("Recommended install commands:")
            lines.extend(f"  {cmd}" for cmd in health["install_commands"])
        console.print(Panel("\n".join(lines), title="Setup Recommendations", border_style="yellow", box=box_style))
    return 0


def _cmd_setup(args: argparse.Namespace) -> int:
    setup_argv: list[str] = []
    if args.check:
        setup_argv.append("--check")
    if args.install:
        setup_argv.append("--install")
    if args.backend:
        setup_argv.extend(["--backend", args.backend])
    if args.yes:
        setup_argv.append("--yes")
    if args.json:
        setup_argv.append("--json")
    return bootstrap_main(setup_argv)


def _cmd_train(args: argparse.Namespace) -> int:
    profile = detect_hardware()
    targets = available_execution_targets(profile)
    auto_target: ExecutionTarget = (
        "gpu" if "gpu" in targets else ("igpu" if "igpu" in targets else "cpu")
    )
    chosen_target: ExecutionTarget = args.target or auto_target
    session = _build_session_from_args(args, default_target=chosen_target)
    session.execution_target = chosen_target

    colab_out: Path | None = None
    if chosen_target == "colab" and args.out:
        colab_out = Path(args.out)

    console = make_console()
    execute_session(
        session,
        profile=profile,
        console=console,
        prompts=None,
        interactive_post_menu=False,
        colab_output_path=colab_out,
    )
    return 0


def _cmd_colab(args: argparse.Namespace) -> int:
    if not args.config and not args.model:
        args.model = "Qwen/Qwen2.5-0.5B-Instruct"
    session = _build_session_from_args(args, default_target="colab")
    session.execution_target = "colab"
    out_path = Path(args.out) if args.out else None
    nb_path = generate_colab_notebook(session, output_path=out_path)
    console = make_console()
    console.print(
        f"[green]Generated self-contained Google Colab notebook:[/green] {nb_path.resolve()}\n"
        "Upload to https://colab.research.google.com/ and run Runtime -> Run all."
    )
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    console = make_console()
    protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]
    cfg = DecideBenchConfig(
        enabled=True,
        version=args.version,
        protocols=protocols,  # type: ignore[arg-type]
        limit=args.limit,
        via_server=bool(args.via_server),
    )
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out) if args.out else model_dir

    ephemeral_server: DecisionHTTPServer | None = None
    ephemeral_thread: threading.Thread | None = None
    try:
        if args.via_server:
            if args.via_server.startswith(("http://", "https://")):
                predictor: Any = args.via_server
            else:
                engine = _load_decision_engine_compat(model_dir, device=args.device)
                ephemeral_server = DecisionHTTPServer(("127.0.0.1", 0), engine=engine)
                ephemeral_thread = threading.Thread(target=ephemeral_server.serve_forever, daemon=True)
                ephemeral_thread.start()
                _, bound_port = ephemeral_server.server_address[:2]
                predictor = f"http://127.0.0.1:{bound_port}"
        else:
            predictor = _load_decision_engine_compat(model_dir, device=args.device)

        summaries = run_decidebench(
            predictor=predictor,
            cfg=cfg,
            output_dir=out_dir,
            model_label=model_dir.name,
        )
        for s in summaries:
            console.print(
                f"[green]DecideBench {s.version} [{s.protocol}]:[/green] "
                f"acc={s.accuracy:.4f} (95% CI [{s.ci_low:.4f}, {s.ci_high:.4f}]) | "
                f"pair_acc={s.pair_accuracy:.4f} | p50={s.latency_p50_ms:.1f}ms | report={s.report_path}"
            )
    finally:
        if ephemeral_server is not None:
            ephemeral_server.shutdown()
            ephemeral_server.server_close()
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    serve_model(
        model_dir=Path(args.model_dir),
        host=args.host,
        port=args.port,
        max_batch=args.max_batch,
        device=args.device,
    )
    return 0


def _cmd_export_gguf(args: argparse.Namespace) -> int:
    console = make_console()
    ok, msg = export_gguf(
        model_dir=Path(args.model_dir),
        output_path=Path(args.out) if args.out else None,
        outtype=args.outtype,
    )
    if ok:
        console.print(f"[green]GGUF exported:[/green] {msg}")
        return 0
    console.print(f"[red]GGUF export failed:[/red] {msg}")
    return 1


def _add_shared_model_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Load base WizardSessionConfig JSON file.")
    parser.add_argument("--model", help="Hugging Face model repo ID or local model directory.")
    parser.add_argument("--dataset", help="Hugging Face dataset repo ID.")
    parser.add_argument("--dataset-config", help="Optional HF dataset config/subset name.")
    parser.add_argument("--split", default="train", help="HF dataset split (default: train).")
    parser.add_argument("--local-file", help="Path to local CSV/TSV/JSON/JSONL dataset.")
    parser.add_argument("--general-mix", action="store_true", help="Use curated Auto General Decision Mix.")
    parser.add_argument("--mix-datasets", help="Comma-separated recipe names for --general-mix.")
    parser.add_argument("--zero-shot", action="store_true", help="Zero-shot decision wrapper (no dataset SFT).")
    parser.add_argument("--state-col", help="Override state/text column name.")
    parser.add_argument("--second-text-col", help="Override second text column (NLI hypothesis).")
    parser.add_argument("--question-col", help="Override question column name.")
    parser.add_argument("--options-col", help="Override options/choices column name.")
    parser.add_argument("--label-col", help="Override label/answer column name.")
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--max-val-samples", type=int, default=400)
    parser.add_argument("--primitive", choices=("choice", "noul", "score"), default="choice")
    parser.add_argument("--method", choices=("lora", "full", "zero_shot"), default="lora")
    parser.add_argument("--attention", choices=("bidirectional", "set", "causal", "encoder"), default="bidirectional")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--eval-decidebench", action="store_true")
    parser.add_argument("--bench-version", choices=("v2", "v1", "both"), default="v2")
    parser.add_argument("--bench-limit", type=int, default=None)
    parser.add_argument("--out", help="Output run directory or .ipynb path.")


def build_parser() -> argparse.ArgumentParser:
    """Create the root ArgumentParser for the `decide` CLI."""
    parser = argparse.ArgumentParser(
        prog="decide",
        description="Decision Forge - Cross-platform local & Google Colab Decision Model studio.",
    )
    sub = parser.add_subparsers(dest="command")

    p_wiz = sub.add_parser("wizard", help="Launch interactive Configuration-First guided TUI wizard.")
    p_wiz.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR, help="Base directory for local run outputs.")

    p_doc = sub.add_parser("doctor", help="Inspect OS, CPU, RAM, GPU/iGPU, and PyTorch runtime health.")
    p_doc.add_argument("--json", action="store_true", help="Output JSON diagnostics.")

    p_setup = sub.add_parser("setup", help="Inspect and bootstrap optimal PyTorch & compute runtimes.")
    p_setup.add_argument("--check", action="store_true")
    p_setup.add_argument("--install", action="store_true")
    p_setup.add_argument("--backend", choices=("auto", "cuda", "rocm", "xpu", "mps", "cpu"), default=None)
    p_setup.add_argument("--yes", "-y", action="store_true")
    p_setup.add_argument("--json", action="store_true")

    p_train = sub.add_parser("train", help="Non-interactive decision model training / Colab export pipeline.")
    _add_shared_model_dataset_args(p_train)
    p_train.add_argument("--target", choices=("gpu", "igpu", "cpu", "colab"), default=None)
    p_train.add_argument("--export-gguf", action="store_true")
    p_train.add_argument("--serve", action="store_true")
    p_train.add_argument("--port", type=int, default=8765)

    p_colab = sub.add_parser("colab", help="Generate a self-contained Google Colab (.ipynb) notebook.")
    _add_shared_model_dataset_args(p_colab)

    p_eval = sub.add_parser("eval", help="Evaluate a trained model directory on DecideBench.")
    p_eval.add_argument("--model-dir", required=True, help="Path to trained/merged model directory.")
    p_eval.add_argument("--version", choices=("v2", "v1", "both"), default="v2")
    p_eval.add_argument("--protocols", default="few_shot,zero_shot")
    p_eval.add_argument("--limit", type=int, default=None)
    p_eval.add_argument(
        "--via-server",
        nargs="?",
        const="local",
        default=None,
        help="Evaluate over HTTP /v1/systemone (pass optional URL or omit value to launch ephemeral server).",
    )
    p_eval.add_argument("--device", default="auto")
    p_eval.add_argument("--out", help="Output directory for benchmark reports.")

    p_serve = sub.add_parser("serve", help="Start OpenAI-compatible /v1/systemone decision HTTP server.")
    p_serve.add_argument("--model-dir", required=True)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765)
    p_serve.add_argument("--max-batch", type=int, default=8)
    p_serve.add_argument("--device", default="auto")

    p_gguf = sub.add_parser("export-gguf", help="Export a saved decision model directory to GGUF via llama.cpp.")
    p_gguf.add_argument("--model-dir", required=True)
    p_gguf.add_argument("--out", help="Output .gguf file path.")
    p_gguf.add_argument("--outtype", default="q8_0")

    return parser


def _find_studio() -> Path | None:
    """Locate the `decide-tui` binary: $DECIDE_TUI_BIN, then a source checkout's build, then PATH."""
    name = "decide-tui.exe" if sys.platform == "win32" else "decide-tui"
    checkout_build = Path(__file__).resolve().parents[2] / "tui" / "target" / "release" / name
    for candidate in (os.environ.get("DECIDE_TUI_BIN"), checkout_build, shutil.which(name)):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def main(argv: list[str] | None = None) -> int:
    """Main entrypoint for the `decide` CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Bare `decide` opens the terminal studio, which drives this same package through
    # `decision_forge.bridge`. `decide wizard` keeps the prompt-by-prompt flow.
    if args.command is None and sys.stdin.isatty() and sys.stdout.isatty():
        studio = _find_studio()
        if studio is not None:
            return subprocess.call([str(studio)], env={**os.environ, "DECIDE_PYTHON": sys.executable})
        print(
            "The terminal studio is not built yet (cargo build --release --manifest-path tui/Cargo.toml). "
            "Opening the prompt wizard instead.",
            file=sys.stderr,
        )

    if args.command is None or args.command == "wizard":
        runs_dir = getattr(args, "runs_dir", DEFAULT_RUNS_DIR)
        run_wizard(runs_dir=runs_dir)
        return 0
    if args.command == "doctor":
        return _cmd_doctor(args)
    if args.command == "setup":
        return _cmd_setup(args)
    if args.command == "train":
        return _cmd_train(args)
    if args.command == "colab":
        return _cmd_colab(args)
    if args.command == "eval":
        return _cmd_eval(args)
    if args.command == "serve":
        return _cmd_serve(args)
    if args.command == "export-gguf":
        return _cmd_export_gguf(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
