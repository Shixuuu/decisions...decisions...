"""JSON-lines backend for the ratatui front end (`decide-tui`).

The front end owns the screen; this process owns every decision about models, datasets,
hardware, and training. One JSON object per line in each direction:

    request   {"id": 3, "method": "search_models", "params": {"query": "qwen"}}
    reply     {"id": 3, "ok": true, "result": ...}  or  {"id": 3, "ok": false, "error": "..."}
    progress  {"id": 9, "event": {"event": "step", ...}}   (only while a `run` request is open)

Requests run on a thread pool, so a slow search never blocks typing and replies can arrive out
of order. The protocol owns the original stdout; anything a library prints goes to stderr.
Searches are remembered for the life of the process, so retyping a query answers at once.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import httpx

from decision_forge.constants import GENERAL_MIX_RECIPES
from decision_forge.types import HardwareProfile

DATASETS_SERVER = "https://datasets-server.huggingface.co"
TARGET_LABELS = {"gpu": "Local GPU", "igpu": "Integrated GPU", "cpu": "Local CPU", "colab": "Colab notebook"}
FIT_LABELS = {"gpu": "GPU", "igpu": "iGPU", "cpu": "CPU", "colab_t4": "Colab T4", "colab_a100": "Colab A100"}
# Hub tasks whose models load as a causal decoder or a text encoder. Vision, audio, and
# robotics checkpoints match name searches too and cannot be trained here.
TRAINABLE_PIPELINES = {"text-generation", "fill-mask", "text-classification", "feature-extraction", None}
PREVIEW_ROWS = 5
PREVIEW_CELL_CHARS = 120

_profile_lock = threading.Lock()
_profile_cache: HardwareProfile | None = None


def _profile() -> HardwareProfile:
    global _profile_cache
    with _profile_lock:
        if _profile_cache is None:
            from decision_forge.hardware import detect_hardware

            _profile_cache = detect_hardware()
        return _profile_cache


def _ready_targets() -> list[dict[str, Any]]:
    from decision_forge.hardware import available_execution_targets, resolve_kernel_stack

    profile = _profile()
    targets = []
    for key in available_execution_targets(profile):
        backend = resolve_kernel_stack(key, profile).backend
        ready = key in ("cpu", "colab") or backend in ("cpu", profile.active_torch_backend)
        targets.append(
            {
                "key": key,
                "label": TARGET_LABELS[key],
                "ready": ready,
                "note": "" if ready else f"PyTorch has no {backend} runtime yet. Run decide setup --install",
            }
        )
    return targets


def _fit_level(text: str) -> str:
    if text.startswith("fits with 4-bit"):
        return "qlora"
    for level in ("fits", "tight", "too_large"):
        if text.startswith(level):
            return level
    return "unknown"


def _fits(param_count_b: float | None) -> list[dict[str, str]]:
    """Memory fit per target, in the order the targets are offered."""
    from decision_forge.hardware import estimate_model_fit

    return [
        {
            "key": key,
            "label": FIT_LABELS.get(key, key),
            "level": _fit_level(text),
            "detail": text.partition("(")[2].rstrip(")"),
        }
        for key, text in estimate_model_fit(param_count_b, _profile()).items()
        if not text.startswith("unavailable")
    ]


def hello(_params: dict[str, Any]) -> dict[str, Any]:
    profile = _profile()
    targets = _ready_targets()
    return {
        "machine": f"{profile.cpu_model} · {profile.ram_total_gb:.0f} GB",
        "targets": targets,
        "mix": [
            {"name": str(r["name"]), "repo_id": str(r["repo_id"]), "primitive": str(r["primitive"])}
            for r in GENERAL_MIX_RECIPES
        ],
    }


def search_models(params: dict[str, Any]) -> list[dict[str, Any]]:
    return _search_models(str(params.get("query", "")).strip())


@functools.lru_cache(maxsize=256)
def _search_models(query: str) -> list[dict[str, Any]]:
    from decision_forge import hf_models

    found = hf_models.search_models(query=query, task=None if query else "text-generation", limit=40)
    found = [info for info in found if info.pipeline_tag in TRAINABLE_PIPELINES][:20]
    local = next((t["key"] for t in _ready_targets() if t["ready"]), "cpu")
    hits = []
    for info in found:
        fits = _fits(info.param_count_b)
        hits.append(
            {
                "info": asdict(info),
                "fits": fits,
                "fit": next((f["level"] for f in fits if f["key"] == local), "unknown"),
            }
        )
    return hits


def search_datasets(params: dict[str, Any]) -> list[dict[str, Any]]:
    return _search_datasets(str(params.get("query", "")).strip())


@functools.lru_cache(maxsize=256)
def _search_datasets(query: str) -> list[dict[str, Any]]:
    from decision_forge import hf_datasets

    found = hf_datasets.search_datasets(query, task_category=None if query else "text-classification", limit=20)
    return [
        {
            "repo_id": d["repo_id"],
            "downloads": d["downloads"],
            "likes": d["likes"],
            "tasks": d["task_categories"][:3],
            "gated": d["gated"],
        }
        for d in found
    ]


def _preview(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from decision_forge.hf_datasets import infer_dataset_schema

    rows = rows[:PREVIEW_ROWS]
    schema = infer_dataset_schema(rows) if rows else {}
    columns = list(rows[0].keys()) if rows else []
    return {
        "columns": columns,
        "rows": [[" ".join(str(row.get(c, "")).split())[:PREVIEW_CELL_CHARS] for c in columns] for row in rows],
        "schema": {
            key: schema.get(key)
            for key in ("schema_type", "state_col", "second_text_col", "question_col", "options_col", "label_col")
        },
    }


def preview_dataset(params: dict[str, Any]) -> dict[str, Any]:
    """First rows of a Hub dataset, choosing a subset and split when the caller names none."""
    repo_id = str(params["repo_id"])
    with httpx.Client(timeout=10.0, follow_redirects=True) as client:
        resp = client.get(f"{DATASETS_SERVER}/splits", params={"dataset": repo_id})
        if resp.status_code != 200:
            raise RuntimeError(resp.json().get("error", f"No preview available (HTTP {resp.status_code})"))
        pairs = [(str(s["config"]), str(s["split"])) for s in resp.json().get("splits", [])]
        if not pairs:
            raise RuntimeError("This dataset exposes no splits to preview.")
        configs = list(dict.fromkeys(config for config, _ in pairs))
        config = params.get("config") or ("default" if "default" in configs else configs[0])
        splits = [split for c, split in pairs if c == config]
        split = "train" if "train" in splits else splits[0]

        resp = client.get(
            f"{DATASETS_SERVER}/first-rows", params={"dataset": repo_id, "config": config, "split": split}
        )
        if resp.status_code != 200:
            raise RuntimeError(resp.json().get("error", f"No preview available (HTTP {resp.status_code})"))
        rows = [entry["row"] for entry in resp.json().get("rows", [])]
    return {"repo_id": repo_id, "config": config, "configs": configs, "split": split, **_preview(rows)}


def preview_file(params: dict[str, Any]) -> dict[str, Any]:
    from decision_forge.hf_datasets import _read_local_rows

    path = Path(str(params["path"])).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"No file at {path}")
    rows = _read_local_rows(path)
    return {"path": str(path.resolve()), "total_rows": len(rows), **_preview(rows)}


def run(params: dict[str, Any], emit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Run one session. `params` are `decide train` arguments keyed by their argparse names."""
    from rich.console import Console

    from decision_forge import cli
    from decision_forge.tui.wizard import execute_session
    from decision_forge.types import load_json

    args = cli.build_parser().parse_args(["train"])
    unknown = set(params) - set(vars(args))
    if unknown:
        raise ValueError(f"Unknown run parameters: {sorted(unknown)}")
    vars(args).update(params)

    session = cli._build_session_from_args(args, default_target=args.target)
    outcome = execute_session(
        session,
        profile=_profile(),
        console=Console(file=sys.stderr, no_color=True, force_terminal=False),
        progress=emit,
    )

    result: dict[str, Any] = {"target": outcome["target"], "output_dir": session.output_dir}
    if outcome["target"] == "colab":
        result["notebook_path"] = os.path.relpath(outcome["notebook_path"])
        return result

    result["model_dir"] = outcome["model_dir"]
    metrics_file = Path(session.output_dir) / "train_metrics.json"
    if metrics_file.is_file():
        metrics = load_json(metrics_file)
        metrics.pop("loss_history", None)
        result["metrics"] = metrics
    result["bench"] = [
        {
            "version": s.version,
            "protocol": s.protocol,
            "accuracy": s.accuracy,
            "ci_low": s.ci_low,
            "ci_high": s.ci_high,
            "pair_accuracy": s.pair_accuracy,
        }
        for s in outcome["bench_summaries"]
    ]
    return result


METHODS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "hello": hello,
    "search_models": search_models,
    "search_datasets": search_datasets,
    "preview_dataset": preview_dataset,
    "preview_file": preview_file,
}


def main() -> int:
    # Keep the protocol on the real stdout and send every stray print to stderr.
    channel = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    write_lock = threading.Lock()

    def send(message: dict[str, Any]) -> None:
        line = json.dumps(message, ensure_ascii=False, default=str)
        with write_lock:
            channel.write(line + "\n")
            channel.flush()

    def handle(request: dict[str, Any]) -> None:
        request_id = request.get("id")
        try:
            method = request["method"]
            params = request.get("params") or {}
            if method == "run":
                result = run(params, emit=lambda event: send({"id": request_id, "event": event}))
            else:
                result = METHODS[method](params)
            send({"id": request_id, "ok": True, "result": result})
        except BaseException as exc:  # noqa: BLE001 - every failure must reach the screen
            # These three carry messages written for people; anything else needs its type to make sense.
            plain = isinstance(exc, (FileNotFoundError, RuntimeError, ValueError)) and str(exc)
            send({"id": request_id, "ok": False, "error": str(exc) if plain else f"{type(exc).__name__}: {exc}"})

    with ThreadPoolExecutor(max_workers=6) as pool:
        for line in sys.stdin:
            if line.strip():
                pool.submit(handle, json.loads(line))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
