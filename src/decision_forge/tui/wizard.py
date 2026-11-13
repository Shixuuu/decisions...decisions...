"""Cross-platform Configuration-First, Execution-Target-Last Guided TUI Wizard."""

from __future__ import annotations

import csv
import importlib
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from decision_forge.bootstrap import check_environment_health, ensure_optimal_backend
from decision_forge.colab import generate_colab_notebook
from decision_forge.constants import (
    DEFAULT_MODEL_SUGGESTIONS,
    DEFAULT_RUNS_DIR,
    GENERAL_MIX_RECIPES,
    OPTION_LETTERS,
    SCORE_DIGITS,
)
from decision_forge.export import export_gguf
from decision_forge.hardware import (
    apply_runtime_env,
    available_execution_targets,
    detect_hardware,
    estimate_model_fit,
    resolve_kernel_stack,
)
from decision_forge.hf_datasets import (
    infer_dataset_schema,
    load_decision_dataset,
    preview_dataset_rows,
    search_datasets,
)
from decision_forge.hf_models import (
    inspect_model_metadata,
    list_cached_models,
    pull_model,
    search_models,
)
from decision_forge.serve import serve_model
from decision_forge.tui import PromptDriver, get_box_style, get_icons, is_utf8_stream, make_console
from decision_forge.types import (
    CalibrationConfig,
    DatasetSpec,
    DecideBenchConfig,
    DecisionItem,
    DecisionOption,
    ExecutionTarget,
    HardwareProfile,
    KernelStackConfig,
    ModelInfo,
    TrainConfig,
    WizardSessionConfig,
    make_run_id,
    sanitize_name,
    save_json,
)


def _format_l4_fit(param_count_b: float | None) -> str:
    if param_count_b is None:
        return "unknown (parameter count unavailable)"
    lora_gb = param_count_b * 2.5 + 1.5
    qlora_gb = param_count_b * 0.85 + 1.5
    if lora_gb <= 24.0 * 0.75:
        return f"fits ({lora_gb:.1f}GB est. on 24.0GB L4 VRAM)"
    if lora_gb <= 24.0 * 0.95:
        return f"tight - use grad checkpointing ({lora_gb:.1f}GB est. on 24.0GB L4 VRAM)"
    if qlora_gb <= 24.0 * 0.85:
        return f"fits with 4-bit QLoRA (~{qlora_gb:.1f}GB est. on 24.0GB L4 VRAM)"
    return f"too_large ({lora_gb:.1f}GB est. > 24.0GB L4 VRAM)"


def _render_environment_banner(
    console: Console,
    profile: HardwareProfile,
    health: dict[str, Any],
) -> None:
    box_style = get_box_style(console.file)
    icons = get_icons(is_utf8_stream(console.file))
    gpu_summary_lines: list[str] = []
    for dev in profile.devices:
        if dev.kind == "cpu":
            continue
        status = f"{icons['ok']} torch active" if dev.available_in_torch else f"{icons['warn']} missing: {', '.join(dev.missing_runtime) or 'torch'}"
        mem = f"{dev.memory_gb:.1f} GB" if dev.memory_gb > 0 else "shared"
        gpu_summary_lines.append(f"  {icons['bullet']} [{dev.kind}] {dev.vendor} {dev.name} ({dev.backend}, {mem}) - {status}")

    if not gpu_summary_lines:
        gpu_summary_lines.append(f"  {icons['bullet']} None detected (local CPU or Google Colab available)")

    banner_body = "\n".join(
        [
            f"[bold]OS:[/bold] {profile.os_distro} ({profile.os_name} / {profile.arch})",
            f"[bold]CPU:[/bold] {profile.cpu_model} ({profile.physical_cores}c / {profile.logical_cores}t)",
            f"[bold]RAM:[/bold] {profile.ram_available_gb:.1f} GB available / {profile.ram_total_gb:.1f} GB total",
            f"[bold]PyTorch:[/bold] {profile.active_torch_version} (backend={profile.active_torch_backend}, recommended={profile.recommended_backend})",
            "[bold]Accelerators:[/bold]",
            *gpu_summary_lines,
        ]
    )
    console.print(Panel(banner_body, title="Decision Forge - Environment & Hardware Diagnostics", border_style="cyan", box=box_style))

    if profile.recommended_backend != "cpu" and not health.get("healthy", True):
        notes = list(health.get("notes", []))
        cmds = list(health.get("install_commands", []))
        hint_lines = notes + ([f"Suggested command: {cmds[0]}"] if cmds else [])
        console.print(
            Panel(
                "\n".join(hint_lines) or f"Runtime for '{profile.recommended_backend}' is not active in PyTorch.",
                title="Optional Local Accelerator Setup (Not Required for CPU or Colab)",
                border_style="yellow",
                box=box_style,
            )
        )


def _safe_inspect_model(
    repo_or_path: str,
    console: Console,
    profile: HardwareProfile | None = None,
) -> ModelInfo:
    from decision_forge.hardware import estimate_model_fit as _hw_estimate_model_fit

    candidate = Path(repo_or_path).expanduser()
    info: ModelInfo
    if candidate.is_dir():
        try:
            info = inspect_model_metadata(repo_id=candidate.name, local_dir=str(candidate))
        except Exception:
            info = ModelInfo(repo_id=candidate.name, local_path=str(candidate))
    else:
        try:
            info = inspect_model_metadata(repo_id=repo_or_path)
        except Exception as exc:
            console.print(f"[yellow]Metadata lookup unavailable offline ({exc}); using defaults for {repo_or_path}.[/yellow]")
            low = repo_or_path.lower()
            arch = "encoder" if "bert" in low else "decoder"
            info = ModelInfo(
                repo_id=repo_or_path,
                architecture_family=arch,  # type: ignore[arg-type]
                supports_thinking_toggle="qwen3" in low,
            )
    if profile is not None and not info.fit_by_target:
        info.fit_by_target = _hw_estimate_model_fit(info.param_count_b, profile)
    return info


def _step_select_model(
    console: Console,
    prompts: PromptDriver,
    profile: HardwareProfile,
) -> ModelInfo:
    box_style = get_box_style(console.file)
    console.print("\n[bold cyan]Step 1 of 5 - Model Search & Selection[/bold cyan]")

    cached = list_cached_models()
    table = Table(title="Suggested Decision Backbones", box=box_style)
    table.add_column("#", style="cyan", width=4)
    table.add_column("Hugging Face Repo ID", style="bold")
    table.add_column("Type")
    for idx, repo in enumerate(DEFAULT_MODEL_SUGGESTIONS, 1):
        kind = "encoder" if "bert" in repo.lower() else "decoder"
        table.add_row(str(idx), repo, kind)
    console.print(table)
    if cached:
        console.print(f"  [dim]{len(cached)} cached local model(s) available via 'c'.[/dim]")

    completers = list(DEFAULT_MODEL_SUGGESTIONS) + ["s", "search", "c", "cached"]
    raw = prompts.ask_text(
        "Pick [1-8], 's' to search HF Hub, 'c' for cached, or enter HF repo_id / local dir",
        default="1",
        completer_words=completers,
    )

    chosen_repo = DEFAULT_MODEL_SUGGESTIONS[0]
    local_dir: str | None = None

    if raw.isdigit() and 1 <= int(raw) <= len(DEFAULT_MODEL_SUGGESTIONS):
        chosen_repo = DEFAULT_MODEL_SUGGESTIONS[int(raw) - 1]
    elif raw.lower() in ("s", "search"):
        query = prompts.ask_text("Search HF models (e.g. Qwen, SmolLM2, ModernBERT)", default="Qwen")
        task = prompts.ask_text("Optional HF pipeline task filter (Enter for any)", default="")
        max_b_raw = prompts.ask_text("Optional max parameter size in B (e.g. 2.0, Enter for any)", default="")
        max_b: float | None = None
        if max_b_raw:
            try:
                max_b = float(max_b_raw)
            except ValueError:
                max_b = None
        from dataclasses import asdict
        from decision_forge.hardware import estimate_model_fit as _hw_estimate_model_fit

        fit_prof = asdict(profile)
        try:
            matches = search_models(query=query, task=task or None, max_params_b=max_b, limit=10, fit_profile=fit_prof)
        except Exception as exc:
            console.print(f"[yellow]HF model search failed ({exc}); falling back to direct repo input.[/yellow]")
            matches = []

        if matches:
            res_table = Table(title=f"HF Hub Search Results: '{query}'", box=box_style)
            res_table.add_column("#", style="cyan", width=4)
            res_table.add_column("Repo ID", style="bold", no_wrap=False, overflow="fold")
            res_table.add_column("Params")
            res_table.add_column("Arch")
            res_table.add_column("Fit")
            res_table.add_column("Downloads", justify="right")
            preferred_fit_key = "gpu" if profile.has_discrete_gpu else ("igpu" if profile.has_igpu else "cpu")
            for idx, m in enumerate(matches, 1):
                p_str = f"{m.param_count_b:.2f}B" if m.param_count_b is not None else "-"
                if not m.fit_by_target:
                    m.fit_by_target = _hw_estimate_model_fit(m.param_count_b, profile)
                fit_str = str(
                    m.fit_by_target.get(preferred_fit_key)
                    or m.fit_by_target.get("cpu")
                    or m.fit_by_target.get("colab_t4")
                    or m.fit_by_target.get("colab")
                    or "-"
                )
                if len(fit_str) > 28:
                    fit_str = fit_str[:25] + "..."
                res_table.add_row(str(idx), m.repo_id, p_str, m.architecture_family, fit_str, f"{m.downloads:,}")
            console.print(res_table)
            pick = prompts.ask_text("Pick result number or enter repo_id", default="1")
            if pick.isdigit() and 1 <= int(pick) <= len(matches):
                chosen = matches[int(pick) - 1]
                refreshed = _safe_inspect_model(chosen.repo_id, console, profile=profile)
                if refreshed.model_type or refreshed.param_count_b is not None or refreshed.transformers_version:
                    chosen = ModelInfo(
                        repo_id=refreshed.repo_id or chosen.repo_id,
                        revision=refreshed.revision or chosen.revision,
                        local_path=refreshed.local_path or chosen.local_path,
                        architecture_family=refreshed.architecture_family or chosen.architecture_family,
                        model_type=refreshed.model_type or chosen.model_type,
                        param_count_b=refreshed.param_count_b if refreshed.param_count_b is not None else chosen.param_count_b,
                        estimated_size_gb=(
                            refreshed.estimated_size_gb if refreshed.estimated_size_gb is not None else chosen.estimated_size_gb
                        ),
                        downloads=refreshed.downloads or chosen.downloads,
                        likes=refreshed.likes or chosen.likes,
                        license=refreshed.license or chosen.license,
                        pipeline_tag=refreshed.pipeline_tag or chosen.pipeline_tag,
                        has_chat_template=refreshed.has_chat_template or chosen.has_chat_template,
                        supports_thinking_toggle=refreshed.supports_thinking_toggle or chosen.supports_thinking_toggle,
                        requires_trust_remote_code=refreshed.requires_trust_remote_code or chosen.requires_trust_remote_code,
                        transformers_version=refreshed.transformers_version or chosen.transformers_version,
                        fit_by_target=refreshed.fit_by_target or chosen.fit_by_target,
                    )
                console.print(
                    f"[green]Selected:[/green] {chosen.repo_id} ({chosen.architecture_family}, "
                    f"chat_template={chosen.has_chat_template}, thinking_toggle={chosen.supports_thinking_toggle})"
                )
                return chosen
            chosen_repo = pick
        else:
            chosen_repo = prompts.ask_text("Enter HF repo_id directly", default=DEFAULT_MODEL_SUGGESTIONS[0])
    elif raw.lower() in ("c", "cached") and cached:
        c_table = Table(title="Cached Models", box=box_style)
        c_table.add_column("#", style="cyan", width=4)
        c_table.add_column("Repo ID", style="bold")
        c_table.add_column("Local Path")
        for idx, m in enumerate(cached, 1):
            c_table.add_row(str(idx), m.repo_id, m.local_path or "-")
        console.print(c_table)
        pick = prompts.ask_text("Pick cached model #", default="1")
        if pick.isdigit() and 1 <= int(pick) <= len(cached):
            chosen_cached = cached[int(pick) - 1]
            if not chosen_cached.fit_by_target:
                from decision_forge.hardware import estimate_model_fit as _hw_estimate_model_fit

                chosen_cached.fit_by_target = _hw_estimate_model_fit(chosen_cached.param_count_b, profile)
            return chosen_cached
        chosen_repo = pick
    else:
        chosen_repo = raw

    if Path(chosen_repo).expanduser().is_dir():
        local_dir = str(Path(chosen_repo).expanduser().resolve())
        info = _safe_inspect_model(local_dir, console, profile=profile)
    else:
        info = _safe_inspect_model(chosen_repo, console, profile=profile)

    if info.param_count_b is not None and info.estimated_size_gb is not None:
        size_str = f"{info.param_count_b:.2f}B (~{info.estimated_size_gb:.2f} GB)"
    elif info.param_count_b is not None:
        size_str = f"{info.param_count_b:.2f}B"
    else:
        size_str = "unknown"
    console.print(
        Panel(
            f"[bold]Repo:[/bold] {info.repo_id}\n"
            f"[bold]Architecture:[/bold] {info.architecture_family} (model_type={info.model_type or 'auto'})\n"
            f"[bold]Parameters:[/bold] {size_str} | [bold]Chat Template:[/bold] {info.has_chat_template} | "
            f"[bold]Thinking Toggle:[/bold] {info.supports_thinking_toggle}",
            title="Inspected Model Metadata (No Weights Downloaded Yet)",
            border_style="green",
            box=box_style,
        )
    )
    return info


def _read_local_preview_rows(path: Path, limit: int = 5) -> list[dict[str, Any]]:
    ext = path.suffix.lower()
    rows: list[dict[str, Any]] = []
    if ext in (".csv", ".tsv"):
        delim = "\t" if ext == ".tsv" else ","
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter=delim)
            for idx, row in enumerate(reader):
                rows.append(dict(row))
                if idx + 1 >= limit:
                    break
        return rows
    if ext == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        seq = obj if isinstance(obj, list) else obj.get("data", obj.get("items", []))
        return [dict(r) for r in seq[:limit] if isinstance(r, dict)]
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                rows.append(obj)
            if len(rows) >= limit:
                break
    return rows


def _render_preview_table(console: Console, rows: list[dict[str, Any]], title: str) -> None:
    if not rows:
        console.print(f"[yellow]No preview rows available for {title}.[/yellow]")
        return
    box_style = get_box_style(console.file)
    cols = list(rows[0].keys())[:6]
    table = Table(title=title, box=box_style)
    for col in cols:
        table.add_column(str(col), overflow="fold", max_width=40)
    for row in rows[:5]:
        vals = []
        for col in cols:
            s = str(row.get(col, ""))
            if len(s) > 64:
                s = s[:61] + "..."
            vals.append(s)
        table.add_row(*vals)
    console.print(table)


def _prompt_schema_overrides(
    console: Console,
    prompts: PromptDriver,
    inferred: dict[str, Any],
    base_spec: DatasetSpec,
) -> DatasetSpec:
    schema_type = str(inferred.get("schema_type") or base_spec.schema_type or "auto")
    state_col = inferred.get("state_col") or base_spec.state_col
    second_col = inferred.get("second_text_col") or base_spec.second_text_col
    q_col = inferred.get("question_col") or base_spec.question_col
    opts_col = inferred.get("options_col") or base_spec.options_col
    label_col = inferred.get("label_col") or base_spec.label_col

    console.print(
        f"[bold]Inferred Schema:[/bold] type={schema_type}, state={state_col}, "
        f"second={second_col}, question={q_col}, options={opts_col}, label={label_col}"
    )

    if prompts.ask_confirm("Override inferred columns or sample limits?", default=False):
        schema_type = prompts.ask_choice(
            "Schema type",
            [
                ("auto", "Auto-detect"),
                ("classification", "Classification (text + label)"),
                ("nli", "NLI (premise + hypothesis + label)"),
                ("multiple_choice", "Multiple-Choice (context/question + options + label)"),
            ],
            default=schema_type if schema_type in ("auto", "classification", "nli", "multiple_choice") else "auto",
        )
        state_col = prompts.ask_text("State/text column", default=str(state_col or "")) or None
        if schema_type == "nli":
            second_col = prompts.ask_text("Second text column (hypothesis)", default=str(second_col or "hypothesis")) or None
        if schema_type == "multiple_choice":
            q_col = prompts.ask_text("Question column (optional)", default=str(q_col or "")) or None
            opts_col = prompts.ask_text("Options column", default=str(opts_col or "options")) or None
        label_col = prompts.ask_text("Label/answer column", default=str(label_col or "label")) or None
        base_spec.default_question = prompts.ask_text("Default question prompt", default=base_spec.default_question)
        base_spec.max_train_samples = prompts.ask_int("Max training samples", default=base_spec.max_train_samples, minimum=1)
        base_spec.max_val_samples = prompts.ask_int("Max validation samples", default=base_spec.max_val_samples, minimum=1)
        base_spec.max_options_per_item = prompts.ask_int("Max options per item", default=base_spec.max_options_per_item, minimum=2, maximum=26)

    base_spec.schema_type = schema_type  # type: ignore[assignment]
    base_spec.state_col = state_col
    base_spec.second_text_col = second_col
    base_spec.question_col = q_col
    base_spec.options_col = opts_col
    base_spec.label_col = label_col
    return base_spec


def _step_select_dataset(console: Console, prompts: PromptDriver) -> DatasetSpec:
    box_style = get_box_style(console.file)
    console.print("\n[bold cyan]Step 2 of 5 - Dataset & Training Mode[/bold cyan]")

    mode_key = prompts.ask_choice(
        "Choose dataset mode",
        [
            ("hf_dataset", "Search Hugging Face Datasets + 5-row live preview + schema mapping"),
            ("local_file", "Local CSV / TSV / JSONL dataset + column mapping"),
            ("auto_general_mix", "General Decision Model (Auto Mix - no dataset needed; trains on curated multi-task mix)"),
            ("zero_shot", "General Decision Model (Zero-Shot - no dataset / no weight training; PriDe + calibration only)"),
        ],
        default="hf_dataset",
    )

    if mode_key == "zero_shot":
        return DatasetSpec(mode="zero_shot")

    if mode_key == "auto_general_mix":
        table = Table(title="Curated Canary-Checked General Mix Recipes", box=box_style)
        table.add_column("Name", style="bold")
        table.add_column("HF Repo ID")
        table.add_column("Primitive")
        table.add_column("Max Opts", justify="right")
        for rec in GENERAL_MIX_RECIPES:
            table.add_row(str(rec["name"]), str(rec["repo_id"]), str(rec["primitive"]), str(rec["max_options_per_item"]))
        console.print(table)
        mix_raw = prompts.ask_text("Mix recipe names (comma-separated, or 'all')", default="all")
        valid_names = {str(r["name"]) for r in GENERAL_MIX_RECIPES}
        if mix_raw.lower() == "all":
            chosen_mix = [str(r["name"]) for r in GENERAL_MIX_RECIPES]
        else:
            chosen_mix = [x.strip() for x in mix_raw.split(",") if x.strip() in valid_names] or [
                str(r["name"]) for r in GENERAL_MIX_RECIPES
            ]
        max_tr = prompts.ask_int("Max total training samples", default=2000, minimum=10)
        max_va = prompts.ask_int("Max total validation samples", default=400, minimum=10)
        return DatasetSpec(
            mode="auto_general_mix",
            mix_datasets=chosen_mix,
            max_train_samples=max_tr,
            max_val_samples=max_va,
        )

    if mode_key == "local_file":
        local_file: Path | None = None
        for _ in range(5):
            raw_path = prompts.ask_text("Path to local CSV, TSV, JSON, or JSONL file", default="", allow_empty=False)
            cand = Path(raw_path).expanduser()
            if cand.is_file():
                local_file = cand.resolve()
                break
            console.print(f"[yellow]File not found: {cand}[/yellow]")
        if local_file is None:
            raise FileNotFoundError("Valid local dataset path was not provided.")

        rows = _read_local_preview_rows(local_file, limit=5)
        _render_preview_table(console, rows, f"Local File Preview: {local_file.name}")
        inferred = infer_dataset_schema(rows) if rows else {}
        spec = DatasetSpec(mode="local_file", local_path=str(local_file))
        return _prompt_schema_overrides(console, prompts, inferred, spec)

    # hf_dataset
    ds_input = prompts.ask_text(
        "Search HF Datasets (query) or enter exact repo_id (e.g. fancyzhx/ag_news)",
        default="fancyzhx/ag_news",
    )
    if "/" not in ds_input or " " in ds_input:
        try:
            ds_hits = search_datasets(ds_input, limit=10)
        except Exception as exc:
            console.print(f"[yellow]HF dataset search unavailable ({exc}).[/yellow]")
            ds_hits = []
        if ds_hits:
            dtable = Table(title=f"HF Dataset Search: '{ds_input}'", box=box_style)
            dtable.add_column("#", style="cyan", width=4)
            dtable.add_column("Repo ID", style="bold")
            dtable.add_column("Downloads", justify="right")
            dtable.add_column("Tasks")
            for idx, d in enumerate(ds_hits, 1):
                dtable.add_row(str(idx), d["repo_id"], f"{d['downloads']:,}", ", ".join(d["task_categories"][:2]) or "-")
            console.print(dtable)
            pick = prompts.ask_text("Pick dataset # or enter repo_id", default="1")
            if pick.isdigit() and 1 <= int(pick) <= len(ds_hits):
                repo_id = ds_hits[int(pick) - 1]["repo_id"]
            else:
                repo_id = pick
        else:
            repo_id = prompts.ask_text("Enter HF dataset repo_id", default="fancyzhx/ag_news")
    else:
        repo_id = ds_input

    cfg_name = prompts.ask_text("Optional dataset config/subset name (Enter for default)", default="") or None
    split = prompts.ask_text("Dataset split", default="train")

    preview_rows: list[dict[str, Any]] = []
    try:
        preview_rows = preview_dataset_rows(repo_id, config_name=cfg_name, split=split, limit=5)
        _render_preview_table(console, preview_rows, f"Live 5-Row Preview: {repo_id} ({split})")
    except Exception as exc:
        console.print(f"[yellow]Live preview skipped ({exc}); continuing with manual/auto schema.[/yellow]")

    inferred = infer_dataset_schema(preview_rows) if preview_rows else {}
    spec = DatasetSpec(
        mode="hf_dataset",
        repo_id=repo_id,
        config_name=cfg_name,
        split=split,
    )
    return _prompt_schema_overrides(console, prompts, inferred, spec)


def _step_configure_training_and_eval(
    console: Console,
    prompts: PromptDriver,
    model: ModelInfo,
    dataset: DatasetSpec,
) -> tuple[DatasetSpec, TrainConfig, CalibrationConfig, DecideBenchConfig, bool, bool, int]:
    console.print("\n[bold cyan]Step 3 of 5 - Training, Architecture, Calibration & DecideBench Config[/bold cyan]")

    primitive = prompts.ask_choice(
        "Decision primitive focus",
        [
            ("choice", "Discrete multiple-choice (A..Z)"),
            ("noul", "Binary decision (No / Yes)"),
            ("score", "Ordinal rating scale (1..9)"),
        ],
        default=dataset.primitive,
    )
    dataset.primitive = primitive  # type: ignore[assignment]

    if model.architecture_family == "encoder":
        console.print("  [green]Encoder backbone detected:[/green] attention_mode automatically set to 'encoder' (native bidirectional).")
        attention_mode = "encoder"
    else:
        attention_mode = prompts.ask_choice(
            "Attention mode",
            [
                ("bidirectional", "Causal-mask lifting across context & options (recommended)"),
                ("set", "Set-structured order-invariant attention"),
                ("causal", "Standard autoregressive causal attention"),
            ],
            default="bidirectional",
        )

    if dataset.mode == "zero_shot":
        method = "zero_shot"
    else:
        method = prompts.ask_choice(
            "Training method",
            [
                ("lora", "LoRA parameter-efficient SFT (recommended)"),
                ("full", "Full-parameter SFT"),
                ("zero_shot", "Zero-shot decision wrapper (no weight updates)"),
            ],
            default="lora",
        )

    train_cfg = TrainConfig(
        method=method,  # type: ignore[arg-type]
        attention_mode=attention_mode,  # type: ignore[arg-type]
    )

    if method == "lora":
        train_cfg.lora_r = prompts.ask_int("LoRA rank (r)", default=16, minimum=1)
        train_cfg.lora_alpha = prompts.ask_int("LoRA alpha", default=32, minimum=1)
        train_cfg.lora_target_scope = prompts.ask_choice(  # type: ignore[assignment]
            "LoRA target scope",
            [("attn_mlp", "Attention + MLP linear layers"), ("attn", "Attention layers only")],
            default="attn_mlp",
        )

    if method in ("lora", "full"):
        default_lr = 2e-4 if method == "lora" else 2e-5
        train_cfg.epochs = prompts.ask_int("Training epochs", default=2, minimum=1)
        train_cfg.learning_rate = prompts.ask_float("Learning rate", default=default_lr, minimum=1e-7)
        train_cfg.batch_size = prompts.ask_int("Micro batch size", default=4, minimum=1)
        train_cfg.grad_accum_steps = prompts.ask_int("Gradient accumulation steps", default=4, minimum=1)
        train_cfg.max_seq_len = prompts.ask_int("Max sequence length", default=512, minimum=64)
        train_cfg.shuffle_options = prompts.ask_confirm("Shuffle option order per training step?", default=True)
        train_cfg.pride_debiasing = False
    else:
        train_cfg.pride_debiasing = prompts.ask_confirm("Enable PriDe cyclic option-order debiasing?", default=True)
        if train_cfg.pride_debiasing:
            train_cfg.pride_permutations = prompts.ask_int("PriDe permutation count", default=3, minimum=2, maximum=8)
        train_cfg.max_seq_len = prompts.ask_int("Max sequence length", default=512, minimum=64)

    if model.architecture_family != "encoder":
        train_cfg.disable_thinking = prompts.ask_confirm("Disable verbose <think> reasoning tokens?", default=True)
    else:
        train_cfg.disable_thinking = False

    cal_enabled = prompts.ask_confirm("Enable post-hoc temperature scaling calibration?", default=True)
    cal_cfg = CalibrationConfig(enabled=cal_enabled)
    if cal_enabled:
        cal_cfg.per_bucket = prompts.ask_confirm("Fit per-bucket temperatures (noul / choice:small / mid / large)?", default=True)
        cal_cfg.order_flip_eval_samples = prompts.ask_int("Order-flip consistency probe samples (0 to skip)", default=64, minimum=0)

    bench_enabled = prompts.ask_confirm("Evaluate on DecideBench after training/wrapping?", default=True)
    bench_cfg = DecideBenchConfig(enabled=bench_enabled)
    if bench_enabled:
        bench_cfg.version = prompts.ask_choice(  # type: ignore[assignment]
            "DecideBench version",
            [("v2", "DecideBench v2 (default)"), ("v1", "DecideBench v1"), ("both", "Both v2 and v1")],
            default="v2",
        )
        proto_choice = prompts.ask_choice(
            "DecideBench evaluation protocols",
            [("both", "few_shot + zero_shot"), ("few_shot", "few_shot only"), ("zero_shot", "zero_shot only")],
            default="both",
        )
        bench_cfg.protocols = ["few_shot", "zero_shot"] if proto_choice == "both" else [proto_choice]  # type: ignore[list-item]
        limit_raw = prompts.ask_text("Optional DecideBench item limit per version (Enter for full)", default="")
        if limit_raw.isdigit() and int(limit_raw) > 0:
            bench_cfg.limit = int(limit_raw)

    export_gguf_flag = prompts.ask_confirm("Export GGUF after local run?", default=False)
    start_server_flag = prompts.ask_confirm("Start /v1/systemone server after local run?", default=False)
    server_port = prompts.ask_int("Server port", default=8765, minimum=1024, maximum=65535) if start_server_flag else 8765

    return dataset, train_cfg, cal_cfg, bench_cfg, export_gguf_flag, start_server_flag, server_port


def _render_review_and_fit_matrix(
    console: Console,
    session: WizardSessionConfig,
    profile: HardwareProfile,
) -> None:
    box_style = get_box_style(console.file)
    console.print("\n[bold cyan]Step 4 of 5 - Configuration Review & Multi-Target Fit Matrix[/bold cyan]")

    ds = session.dataset
    tr = session.train
    cal = session.calibration
    db = session.decidebench

    ds_desc = ds.mode
    if ds.mode == "hf_dataset":
        ds_desc = f"hf_dataset ({ds.repo_id}, split={ds.split})"
    elif ds.mode == "local_file":
        ds_desc = f"local_file ({ds.local_path})"
    elif ds.mode == "auto_general_mix":
        ds_desc = f"auto_general_mix ({', '.join(ds.mix_datasets)})"

    summary_lines = [
        f"[bold]Run ID:[/bold] {session.run_id}",
        f"[bold]Output Dir:[/bold] {session.output_dir}",
        f"[bold]Model:[/bold] {session.model.repo_id} (arch={session.model.architecture_family})",
        f"[bold]Dataset:[/bold] {ds_desc} | [bold]Primitive:[/bold] {ds.primitive}",
        f"[bold]Training:[/bold] method={tr.method}, attention={tr.attention_mode}, epochs={tr.epochs}, lr={tr.learning_rate}, bs={tr.batch_size}x{tr.grad_accum_steps}, max_seq_len={tr.max_seq_len}",
        f"[bold]Calibration:[/bold] enabled={cal.enabled}, per_bucket={cal.per_bucket}, order_flip_samples={cal.order_flip_eval_samples}",
        f"[bold]DecideBench:[/bold] enabled={db.enabled}, version={db.version}, protocols={db.protocols}, limit={db.limit}",
        f"[bold]Post-Run:[/bold] export_gguf={session.export_gguf}, start_server={session.start_server_after} (port={session.server_port})",
    ]
    console.print(Panel("\n".join(summary_lines), title="Chosen Configuration Summary", border_style="cyan", box=box_style))

    fit = estimate_model_fit(session.model.param_count_b, profile)
    fit_table = Table(title="Estimated Model Fit Across Targets", box=box_style)
    fit_table.add_column("Target", style="bold")
    fit_table.add_column("Estimated Fit")
    if profile.has_discrete_gpu:
        fit_table.add_row("Local GPU", fit.get("gpu", "unknown"))
    if profile.has_igpu:
        fit_table.add_row("Local iGPU", fit.get("igpu", "unknown"))
    fit_table.add_row("Local CPU", fit.get("cpu", "unknown"))
    fit_table.add_row("Google Colab T4 (15GB)", fit.get("colab_t4", "unknown"))
    fit_table.add_row("Google Colab L4 (24GB)", _format_l4_fit(session.model.param_count_b))
    fit_table.add_row("Google Colab A100 (40GB)", fit.get("colab_a100", "unknown"))
    console.print(fit_table)


def _step_choose_execution_target(
    console: Console,
    prompts: PromptDriver,
    profile: HardwareProfile,
) -> ExecutionTarget:
    console.print("\n[bold cyan]Step 5 of 5 (LAST STEP) - Choose Where to Train / Execute[/bold cyan]")
    targets = available_execution_targets(profile)
    label_map: dict[ExecutionTarget, str] = {
        "gpu": "Local Discrete GPU (CUDA / ROCm)",
        "igpu": "Local Integrated GPU (Intel Arc XPU / AMD / Apple MPS)",
        "cpu": "Local CPU",
        "colab": "Export Ready-Made Google Colab Notebook (.ipynb)",
    }
    options = [(t, label_map[t]) for t in targets]
    default_target: ExecutionTarget = (
        "gpu" if "gpu" in targets else ("igpu" if "igpu" in targets else "colab")
    )
    chosen = prompts.ask_choice("Select execution target", options, default=default_target)
    return chosen  # type: ignore[return-value]


def _load_decision_engine_compat(model_dir: Path | str, device: str = "auto") -> Any:
    for mod_name in ("decision_forge.engine", "decision_forge.infer", "decision_forge.inference"):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        cls = getattr(mod, "DecisionEngine", None)
        if cls is None:
            continue
        if hasattr(cls, "from_pretrained"):
            sig = inspect.signature(cls.from_pretrained)
            if "device" in sig.parameters:
                return cls.from_pretrained(model_dir, device=device)
            return cls.from_pretrained(model_dir)
        sig = inspect.signature(cls)
        if "device" in sig.parameters:
            return cls(model_dir, device=device)
        return cls(model_dir)
    raise ImportError("Could not import DecisionEngine from decision_forge.engine")


def _call_train_decision_model(
    session: WizardSessionConfig,
    train_items: list[DecisionItem],
    val_items: list[DecisionItem],
    kernel: KernelStackConfig,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    trainer_mod = importlib.import_module("decision_forge.trainer")
    fn = getattr(trainer_mod, "train_decision_model")
    sig = inspect.signature(fn)
    params = sig.parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    if has_var_kw and "session" not in params:
        result = fn(
            model_info=session.model,
            train_items=train_items,
            val_items=val_items,
            train_cfg=session.train,
            calib_cfg=session.calibration,
            kernel=kernel,
            output_dir=Path(session.output_dir),
        )
    else:
        kwargs: dict[str, Any] = {}
        if "session" in params:
            kwargs["session"] = session
        if "model_info" in params:
            kwargs["model_info"] = session.model
        if "train_items" in params:
            kwargs["train_items"] = train_items
        if "val_items" in params:
            kwargs["val_items"] = val_items
        if "train_cfg" in params:
            kwargs["train_cfg"] = session.train
        if "calib_cfg" in params:
            kwargs["calib_cfg"] = session.calibration
        elif "cal_cfg" in params:
            kwargs["cal_cfg"] = session.calibration
        if "kernel" in params:
            kwargs["kernel"] = kernel
        if "output_dir" in params:
            kwargs["output_dir"] = Path(session.output_dir)
        if "progress_cb" in params:
            kwargs["progress_cb"] = progress
        result = fn(**kwargs)

    if isinstance(result, (str, Path)):
        return Path(result)
    if isinstance(result, dict):
        for key in ("merged_dir", "model_dir", "output_dir"):
            if key in result and result[key]:
                return Path(result[key])
    merged = Path(session.output_dir) / "merged"
    return merged if merged.is_dir() else Path(session.output_dir)


def _interactive_probe_loop(console: Console, prompts: PromptDriver, engine: Any) -> None:
    console.print("\n[bold cyan]Interactive Decision Probe (leave state empty to exit)[/bold cyan]")
    while True:
        state = prompts.ask_text("State / Context (Enter to finish)", default="")
        if not state:
            break
        question = prompts.ask_text("Question", default="Which option is best?")
        primitive = prompts.ask_choice(
            "Primitive",
            [("choice", "Multiple choice"), ("noul", "Binary decision"), ("score", "Ordinal 1..9")],
            default="choice",
        )
        raw_opts = prompts.ask_text("Comma-separated options", default="Approve, Decline")
        opt_texts = [o.strip() for o in raw_opts.split(",") if o.strip()] or ["Approve", "Decline"]
        keys = SCORE_DIGITS if primitive == "score" else OPTION_LETTERS
        options = [DecisionOption(key=keys[i], description=t) for i, t in enumerate(opt_texts[: len(keys)])]
        item = DecisionItem(
            id="interactive_probe",
            state=state,
            question=question,
            options=options,
            primitive=primitive,  # type: ignore[arg-type]
        )
        if hasattr(engine, "predict_one"):
            pred = engine.predict_one(item)
        elif hasattr(engine, "predict"):
            pred = engine.predict(item)
        elif hasattr(engine, "predict_item"):
            pred = engine.predict_item(item)
        else:
            pred = engine.predict_batch([item])[0]
        console.print(
            Panel(
                f"[bold]Choice:[/bold] {pred.choice} | [bold]Confidence:[/bold] {pred.confidence:.4f} | "
                f"[bold]Temp:[/bold] {pred.temperature:.3f} | [bold]Latency:[/bold] {pred.latency_ms:.1f} ms\n"
                f"[bold]Probabilities:[/bold] {pred.probabilities}",
                title="Decision Prediction",
                border_style="green",
                box=get_box_style(console.file),
            )
        )


def execute_session(
    session: WizardSessionConfig,
    profile: HardwareProfile | None = None,
    console: Console | None = None,
    prompts: PromptDriver | None = None,
    interactive_post_menu: bool = False,
    colab_output_path: Path | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute the configured session on the chosen target (Colab export or local compute).

    `progress` receives a `plan` event naming the phases this run will go through, a `phase`
    event as each one starts, and the trainer's `step` events.
    """
    out_console = console if console is not None else make_console()
    box_style = get_box_style(out_console.file)
    target = session.execution_target or "colab"

    def emit(event: str, **fields: Any) -> None:
        if progress is not None:
            progress({"event": event, **fields})

    if target == "colab":
        emit("plan", phases=["notebook"])
        emit("phase", name="notebook")
        nb_path = generate_colab_notebook(session, output_path=colab_output_path)
        out_console.print(
            Panel(
                f"[bold green]Ready-made Google Colab notebook created:[/bold green] {nb_path.resolve()}\n\n"
                "Next steps:\n"
                "  1. Open https://colab.research.google.com/ and upload this .ipynb file.\n"
                "  2. Select Runtime -> Change runtime type -> T4 / L4 / A100 GPU.\n"
                "  3. Click Runtime -> Run all.",
                title="Colab Notebook Exported",
                border_style="green",
                box=box_style,
            )
        )
        return {"target": "colab", "notebook_path": str(nb_path), "session": session}

    trains = session.train.method != "zero_shot"
    emit(
        "plan",
        phases=["weights"]
        + (["dataset"] if session.dataset.mode != "zero_shot" else [])
        + ["train"]
        + (["calibrate"] if session.calibration.enabled and session.dataset.mode != "zero_shot" else [])
        + (["bench"] if session.decidebench.enabled else [])
        + (["export"] if session.export_gguf else []),
    )

    prof = profile if profile is not None else detect_hardware()
    out_dir = Path(session.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "session_config.json", session)

    kernel = resolve_kernel_stack(
        target,
        prof,
        param_count_b=session.model.param_count_b,
        batch_size=session.train.batch_size,
    )

    if target in ("gpu", "igpu") and kernel.backend != "cpu" and kernel.backend != prof.active_torch_backend:
        out_console.print(
            f"[yellow]Selected target '{target}' requires PyTorch backend '{kernel.backend}', "
            f"while current PyTorch has '{prof.active_torch_backend}'.[/yellow]"
        )
        installed_ok = False
        if prompts is not None and prompts.ask_confirm(
            f"Run ensure_optimal_backend for '{kernel.backend}' now?",
            default=False,
        ):
            installed_ok = ensure_optimal_backend(
                prof,
                auto_install=True,
                yes=True,
                target_backend=kernel.backend,
                console=out_console,
            )
            prof = detect_hardware()
            kernel = resolve_kernel_stack(
                target,
                prof,
                param_count_b=session.model.param_count_b,
                batch_size=session.train.batch_size,
            )
        if not installed_ok and kernel.backend != prof.active_torch_backend:
            out_console.print("[yellow]Falling back cleanly to local CPU execution.[/yellow]")
            kernel = resolve_kernel_stack(
                "cpu",
                prof,
                param_count_b=session.model.param_count_b,
                batch_size=session.train.batch_size,
            )

    apply_runtime_env(kernel)

    emit("phase", name="weights")
    if not session.model.local_path or not Path(session.model.local_path).exists():
        out_console.print(f"[cyan]Pulling model weights for {session.model.repo_id}...[/cyan]")
        session.model = pull_model(session.model.repo_id, revision=session.model.revision)

    if session.dataset.mode != "zero_shot":
        emit("phase", name="dataset")
    out_console.print(f"[cyan]Loading dataset ({session.dataset.mode})...[/cyan]")
    train_items, val_items, ds_meta = load_decision_dataset(session.dataset, seed=session.train.seed)
    save_json(out_dir / "dataset_meta.json", ds_meta)
    if train_items or val_items:
        emit("phase", name="dataset", detail=f"{len(train_items):,} train · {len(val_items):,} validation")

    emit("phase", name="train", detail=f"{session.train.method} on {kernel.device_str}" if trains else "zero shot wrap")

    out_console.print(f"[cyan]Running trainer ({session.train.method} / {session.train.attention_mode} on {kernel.device_str})...[/cyan]")
    model_dir = _call_train_decision_model(session, train_items, val_items, kernel, progress)

    bench_summaries = []
    if session.decidebench.enabled:
        from decision_forge.bench.evaluator import run_decidebench

        emit("phase", name="bench")
        engine = _load_decision_engine_compat(model_dir, device=kernel.device_str)
        bench_summaries = run_decidebench(
            predictor=engine,
            cfg=session.decidebench,
            output_dir=out_dir,
            model_label=session.run_id,
        )
        for s in bench_summaries:
            out_console.print(
                f"[green]DecideBench {s.version} [{s.protocol}]:[/green] "
                f"acc={s.accuracy:.4f} (95% CI [{s.ci_low:.4f}, {s.ci_high:.4f}]) | pair_acc={s.pair_accuracy:.4f}"
            )

    if session.export_gguf:
        emit("phase", name="export")
        ok, msg = export_gguf(model_dir)
        style = "green" if ok else "yellow"
        out_console.print(f"[{style}]GGUF export: {msg}[/{style}]")

    if session.start_server_after:
        serve_model(model_dir, port=session.server_port, device=kernel.device_str)

    if interactive_post_menu and prompts is not None:
        while True:
            post_choice = prompts.ask_choice(
                "\nPost-run actions",
                [
                    ("probe", "Interactive decision probe"),
                    ("serve", "Start /v1/systemone HTTP server"),
                    ("gguf", "Export GGUF"),
                    ("colab", "Also export ready-made Google Colab Notebook (.ipynb)"),
                    ("exit", "Exit wizard"),
                ],
                default="exit",
            )
            if post_choice == "exit":
                break
            if post_choice == "probe":
                engine = _load_decision_engine_compat(model_dir, device=kernel.device_str)
                _interactive_probe_loop(out_console, prompts, engine)
            elif post_choice == "serve":
                serve_model(model_dir, port=session.server_port, device=kernel.device_str)
            elif post_choice == "gguf":
                ok, msg = export_gguf(model_dir)
                out_console.print(f"GGUF export ({'ok' if ok else 'failed'}): {msg}")
            elif post_choice == "colab":
                nb_path = generate_colab_notebook(session)
                out_console.print(f"[green]Colab notebook exported:[/green] {nb_path}")

    return {
        "target": target,
        "model_dir": str(model_dir),
        "bench_summaries": bench_summaries,
        "session": session,
    }


def run_wizard(
    console: Console | None = None,
    prompts: PromptDriver | None = None,
    execute: bool = True,
    interactive_post_menu: bool = True,
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
) -> WizardSessionConfig:
    """Run the Configuration-First, Execution-Target-Last interactive Decision Forge wizard."""
    out_console = console if console is not None else make_console()
    driver = prompts if prompts is not None else PromptDriver(console=out_console)

    # 0) Environment & Diagnostics Banner (non-blocking optional setup)
    profile = detect_hardware()
    health = check_environment_health(profile)
    _render_environment_banner(out_console, profile, health)
    if profile.recommended_backend != "cpu" and not health.get("healthy", True):
        if driver.ask_confirm(
            "Install/repair missing local GPU/iGPU runtime now? (Skip if choosing CPU or Colab later)",
            default=False,
        ):
            ensure_optimal_backend(profile, auto_install=True, yes=True, console=out_console)
            profile = detect_hardware()

    # 1) Step 1 - Model Search & Selection (metadata only, no multi-GB eager pull)
    model_info = _step_select_model(out_console, driver, profile)

    # 2) Step 2 - Dataset & Training Mode
    dataset_spec = _step_select_dataset(out_console, driver)

    # 3) Step 3 - Training, Architecture, Calibration & DecideBench Configuration
    (
        dataset_spec,
        train_cfg,
        cal_cfg,
        bench_cfg,
        export_gguf_flag,
        start_server_flag,
        server_port,
    ) = _step_configure_training_and_eval(out_console, driver, model_info, dataset_spec)

    run_id = make_run_id(model_info.repo_id)
    out_dir = Path(runs_dir) / sanitize_name(run_id)

    session = WizardSessionConfig(
        run_id=run_id,
        output_dir=str(out_dir),
        model=model_info,
        dataset=dataset_spec,
        train=train_cfg,
        calibration=cal_cfg,
        decidebench=bench_cfg,
        export_gguf=export_gguf_flag,
        start_server_after=start_server_flag,
        server_port=server_port,
        execution_target=None,
    )

    # 4) Configuration Review & Multi-Target Fit Matrix
    _render_review_and_fit_matrix(out_console, session, profile)

    # 5) LAST STEP - Choose Where to Train / Execute
    chosen_target = _step_choose_execution_target(out_console, driver, profile)
    session.execution_target = chosen_target

    colab_out: Path | None = None
    if chosen_target == "colab":
        default_nb = str(out_dir / f"{sanitize_name(run_id)}_colab.ipynb")
        nb_str = driver.ask_text("Output Colab notebook path (.ipynb)", default=default_nb)
        colab_out = Path(nb_str)

    # 6) Execute Selected Target
    if execute:
        execute_session(
            session,
            profile=profile,
            console=out_console,
            prompts=driver,
            interactive_post_menu=interactive_post_menu,
            colab_output_path=colab_out,
        )

    return session
