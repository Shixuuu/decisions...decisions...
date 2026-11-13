"""Hugging Face Hub model catalog search, lightweight inspection, snapshot pull, and token validation."""

from __future__ import annotations

import inspect
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import HfApi, hf_hub_download, snapshot_download, try_to_load_from_cache

from decision_forge.constants import APP_CACHE_DIR_NAME, OPTION_LETTERS
from decision_forge.types import (
    ArchitectureFamily,
    HardwareProfile,
    ModelInfo,
    load_json,
    sanitize_name,
    save_json,
)

ENCODER_PIPELINE_TAGS = frozenset({"text-classification", "fill-mask", "feature-extraction"})
ENCODER_MODEL_TYPES = frozenset(
    {
        "bert",
        "roberta",
        "modernbert",
        "deberta",
        "deberta-v2",
        "distilbert",
        "electra",
        "xlm-roberta",
        "albert",
    }
)
CUSTOM_DECISION_PATTERN = re.compile(r"(?:^|[^a-z0-9])(d1|decider|clef|laya)(?:$|[^a-z0-9])")
PARAM_NAME_PATTERN = re.compile(r"(?:^|[-_/\sxX])(\d+(?:\.\d+)?)\s*([bBmM])(?=$|[-_/\s.])")
DEFAULT_TARGET_BUDGETS_GB: dict[str, float] = {"colab": 15.0}
# Exports for other runtimes that sit beside the PyTorch weights and often outweigh them several times over.
FOREIGN_WEIGHT_PATTERNS = [
    "*.onnx",
    "*.onnx_data",
    "onnx/*",
    "openvino/*",
    "*.gguf",
    "*.h5",
    "*.msgpack",
    "*.tflite",
    "*.ot",
    "*.pb",
    "*.mlpackage/*",
]


def get_cache_dir() -> Path:
    """Return the Decision Forge cache directory."""
    raw = os.environ.get("DECISION_FORGE_CACHE")
    if raw:
        return Path(raw)
    return Path.cwd() / APP_CACHE_DIR_NAME


def _registry_path() -> Path:
    return get_cache_dir() / "models.json"


def _registry_key(repo_id: str, revision: str) -> str:
    return f"{sanitize_name(repo_id)}--{sanitize_name(revision)}"


def _load_registry() -> dict[str, dict[str, Any]]:
    path = _registry_path()
    if not path.is_file():
        return {}
    try:
        data = load_json(path)
    except Exception:
        return {}
    if isinstance(data, dict):
        if "models" in data and isinstance(data["models"], dict):
            return dict(data["models"])
        return {k: v for k, v in data.items() if isinstance(v, dict)}
    return {}


def _extract_license(hub_info: Any) -> str | None:
    card_data = getattr(hub_info, "card_data", None)
    if card_data is not None:
        lic = card_data.get("license") if isinstance(card_data, dict) else getattr(card_data, "license", None)
        if isinstance(lic, list) and lic:
            return ", ".join(str(x) for x in lic)
        if isinstance(lic, str) and lic.strip():
            return lic.strip()
    for tag in getattr(hub_info, "tags", None) or []:
        tag_str = str(tag)
        if tag_str.lower().startswith("license:"):
            return tag_str.split(":", 1)[1].strip() or None
    return None


def extract_param_count_b(
    repo_id: str,
    safetensors: Any = None,
    config: dict[str, Any] | None = None,
) -> float | None:
    """Extract parameter count in billions from safetensors metadata, config, or repo name."""
    total_params: int | float | None = None
    if safetensors is not None:
        if isinstance(safetensors, dict):
            total_params = safetensors.get("total")
            if not total_params and isinstance(safetensors.get("parameters"), dict):
                total_params = sum(
                    v for v in safetensors["parameters"].values() if isinstance(v, (int, float))
                )
        else:
            total_params = getattr(safetensors, "total", None)
            if not total_params:
                params_map = getattr(safetensors, "parameters", None)
                if isinstance(params_map, dict):
                    total_params = sum(
                        v for v in params_map.values() if isinstance(v, (int, float))
                    )
    if isinstance(total_params, (int, float)) and total_params > 0:
        return round(float(total_params) / 1e9, 3)

    if isinstance(config, dict):
        cfg_params = config.get("num_parameters")
        if isinstance(cfg_params, (int, float)) and cfg_params > 0:
            return round(float(cfg_params) / 1e9, 3)

    matches = PARAM_NAME_PATTERN.findall(repo_id)
    if matches:
        num_str, unit = matches[-1]
        value = float(num_str)
        if unit.lower() == "m":
            return round(value / 1000.0, 3)
        return round(value, 3)
    return None


def infer_architecture_family(
    repo_id: str,
    tags: list[str] | tuple[str, ...] | None = None,
    pipeline_tag: str | None = None,
    model_type: str = "",
    config: dict[str, Any] | None = None,
) -> ArchitectureFamily:
    """Infer architecture family ('custom_decision', 'encoder', or 'decoder')."""
    tags_lower = {str(t).strip().lower() for t in (tags or [])}
    model_type_lower = (model_type or "").strip().lower()
    if not model_type_lower and isinstance(config, dict):
        model_type_lower = str(config.get("model_type", "") or "").strip().lower()

    if "decision-model" in tags_lower:
        return "custom_decision"
    if any(kw in tags_lower for kw in ("d1", "decider", "clef", "laya")):
        return "custom_decision"
    if CUSTOM_DECISION_PATTERN.search(f"{repo_id} {model_type_lower}".lower()):
        return "custom_decision"

    pipe_lower = (pipeline_tag or "").strip().lower()
    if pipe_lower in ENCODER_PIPELINE_TAGS or model_type_lower in ENCODER_MODEL_TYPES:
        return "encoder"

    return "decoder"


def estimate_model_size_gb(param_count_b: float | None) -> float | None:
    """Estimate BF16 checkpoint size in GB (~2.1 GB per 1B parameters)."""
    if param_count_b is None or param_count_b <= 0:
        return None
    return round(param_count_b * 2.1, 3)


def _classify_memory_fit(
    param_count_b: float | None,
    estimated_size_gb: float | None,
    available_gb: float,
) -> str:
    if estimated_size_gb is None or available_gb <= 0:
        return "unknown"
    if estimated_size_gb * 1.5 <= available_gb:
        return "fits_bf16"
    if param_count_b is not None and (param_count_b * 0.7 + 1.5) <= available_gb:
        return "fits_qlora"
    if estimated_size_gb <= available_gb:
        return "tight"
    return "too_large"


def estimate_model_fit(
    param_count_b: float | None,
    estimated_size_gb: float | None,
    fit_profile: dict[str, Any] | HardwareProfile | Callable[..., Any] | None = None,
) -> dict[str, str]:
    """Compute per-target fit status using a hardware profile, budget map, or callback."""
    if fit_profile is None:
        return {}

    if callable(fit_profile):
        try:
            result = fit_profile(param_count_b=param_count_b, estimated_size_gb=estimated_size_gb)
        except TypeError:
            result = fit_profile(param_count_b, estimated_size_gb)
        return dict(result) if isinstance(result, dict) else {}

    if isinstance(fit_profile, dict) and callable(fit_profile.get("estimate_model_fit")):
        fn = fit_profile["estimate_model_fit"]
        try:
            result = fn(param_count_b=param_count_b, estimated_size_gb=estimated_size_gb)
        except TypeError:
            result = fn(param_count_b, estimated_size_gb)
        return dict(result) if isinstance(result, dict) else {}

    budgets: dict[str, float] = {}
    if isinstance(fit_profile, HardwareProfile):
        profile_dict = asdict(fit_profile)
    elif isinstance(fit_profile, dict):
        profile_dict = fit_profile
    else:
        return {}

    if "targets" in profile_dict and isinstance(profile_dict["targets"], dict):
        for target, mem in profile_dict["targets"].items():
            if isinstance(mem, (int, float)):
                budgets[str(target)] = float(mem)
    elif any(k in profile_dict for k in ("devices", "ram_available_gb", "ram_total_gb")):
        gpu_mems: list[float] = []
        igpu_mems: list[float] = []
        for dev in profile_dict.get("devices") or []:
            if not isinstance(dev, dict):
                continue
            mem = float(dev.get("memory_gb", 0.0) or 0.0)
            kind = dev.get("kind")
            if kind == "discrete_gpu":
                gpu_mems.append(mem)
            elif kind == "igpu":
                igpu_mems.append(mem)
        if gpu_mems:
            budgets["gpu"] = max(gpu_mems)
        if igpu_mems:
            budgets["igpu"] = max(igpu_mems)
        cpu_ram = float(
            profile_dict.get("ram_available_gb") or profile_dict.get("ram_total_gb") or 0.0
        )
        if cpu_ram > 0:
            budgets["cpu"] = cpu_ram
        budgets.setdefault("colab", DEFAULT_TARGET_BUDGETS_GB["colab"])
    else:
        for target in ("gpu", "igpu", "cpu", "colab"):
            val = profile_dict.get(target)
            if isinstance(val, (int, float)):
                budgets[target] = float(val)

    return {
        target: _classify_memory_fit(param_count_b, estimated_size_gb, mem_gb)
        for target, mem_gb in budgets.items()
    }


def _supports_list_models_direction(api: HfApi) -> bool:
    try:
        sig = inspect.signature(api.list_models)
        return "direction" in sig.parameters
    except (TypeError, ValueError):
        return False


def _detect_chat_and_thinking(
    repo_id: str,
    model_type: str,
    tokenizer_config: dict[str, Any] | None,
    architecture_family: ArchitectureFamily,
) -> tuple[bool, bool]:
    chat_template_raw = (tokenizer_config or {}).get("chat_template")
    if isinstance(chat_template_raw, (list, dict)):
        chat_template_str = json.dumps(chat_template_raw)
    elif isinstance(chat_template_raw, str):
        chat_template_str = chat_template_raw
    else:
        chat_template_str = ""

    if tokenizer_config is not None:
        has_chat_template = bool(chat_template_str.strip())
    elif architecture_family == "encoder":
        has_chat_template = False
    else:
        lower_repo = repo_id.lower()
        has_chat_template = any(tok in lower_repo for tok in ("instruct", "chat", "qwen3", "r1"))

    lower_repo = repo_id.lower()
    lower_mt = (model_type or "").lower()
    lower_tpl = chat_template_str.lower()
    supports_thinking = (
        "enable_thinking" in lower_tpl
        or "<think>" in lower_tpl
        or lower_mt in {"qwen3", "qwen3_moe"}
        or "qwen3" in lower_repo
        or "deepseek-r1" in lower_repo
        or "deepseek_r1" in lower_repo
    )
    return has_chat_template, supports_thinking


def _detect_remote_code_and_version(
    config: dict[str, Any] | None,
    tags: list[str] | None = None,
) -> tuple[bool, str]:
    cfg = config if isinstance(config, dict) else {}
    low_tags = {str(t).lower() for t in (tags or [])}
    auto_map = cfg.get("auto_map")
    requires_remote = (
        (isinstance(auto_map, dict) and bool(auto_map))
        or ("custom_code" in low_tags)
        or ("trust-remote-code" in low_tags)
        or ("trust_remote_code" in low_tags)
    )
    tf_ver = str(cfg.get("transformers_version", "") or "").strip()
    return requires_remote, tf_ver


def _is_non_transformers_weights_repo(
    repo_id: str,
    tags: list[str],
    model_type: str,
    safetensors: Any,
) -> bool:
    del model_type
    low_tags = {str(t).lower() for t in tags}
    if safetensors is not None or "safetensors" in low_tags:
        return False
    repo_low = repo_id.lower()
    non_hf_tags = {"gguf", "mlx", "onnx", "coreml", "tflite", "exl2"}
    if low_tags & non_hf_tags:
        return True
    return any(
        marker in repo_low
        for marker in ("-gguf", "_gguf", "/gguf", "-localai", "-mlx", "-onnx", "-exl2")
    )


def search_models(
    query: str,
    task: str | None = None,
    max_params_b: float | None = None,
    limit: int = 12,
    fit_profile: dict[str, Any] | None = None,
) -> list[ModelInfo]:
    """Search Hugging Face Hub models and convert matches into ModelInfo records."""
    api = HfApi()
    fetch_limit = max(limit * 2, 1)
    kwargs: dict[str, Any] = {
        "search": query or None,
        "filter": task or None,
        "sort": "downloads",
        "limit": fetch_limit,
        "expand": ["safetensors", "downloads", "likes", "tags", "pipeline_tag", "config"],
    }
    if _supports_list_models_direction(api):
        kwargs["direction"] = -1

    allow_non_transformers = any(
        tok in (query or "").lower() for tok in ("gguf", "mlx", "onnx", "localai")
    )
    results: list[ModelInfo] = []
    fallback_results: list[ModelInfo] = []
    for raw in api.list_models(**kwargs):
        repo_id = str(getattr(raw, "id", "") or "")
        if not repo_id:
            continue
        config = getattr(raw, "config", None)
        config_dict = config if isinstance(config, dict) else {}
        model_type = str(config_dict.get("model_type", "") or "")
        tags = [str(t) for t in (getattr(raw, "tags", None) or [])]
        pipeline_tag = getattr(raw, "pipeline_tag", None)
        safetensors = getattr(raw, "safetensors", None)

        param_count_b = extract_param_count_b(repo_id, safetensors=safetensors, config=config_dict)
        if max_params_b is not None and param_count_b is not None and param_count_b > max_params_b:
            continue

        estimated_size_gb = estimate_model_size_gb(param_count_b)
        arch = infer_architecture_family(
            repo_id=repo_id,
            tags=tags,
            pipeline_tag=pipeline_tag,
            model_type=model_type,
            config=config_dict,
        )
        fit_by_target = estimate_model_fit(param_count_b, estimated_size_gb, fit_profile)
        raw_tok_cfg = config_dict.get("tokenizer_config")
        tok_cfg_dict = raw_tok_cfg if isinstance(raw_tok_cfg, dict) else None
        has_chat, supports_thinking = _detect_chat_and_thinking(
            repo_id=repo_id,
            model_type=model_type,
            tokenizer_config=tok_cfg_dict,
            architecture_family=arch,
        )
        requires_remote, tf_ver = _detect_remote_code_and_version(config_dict, tags)

        info = ModelInfo(
            repo_id=repo_id,
            revision="main",
            local_path=None,
            architecture_family=arch,
            model_type=model_type,
            param_count_b=param_count_b,
            estimated_size_gb=estimated_size_gb,
            downloads=int(getattr(raw, "downloads", 0) or 0),
            likes=int(getattr(raw, "likes", 0) or 0),
            license=_extract_license(raw),
            pipeline_tag=pipeline_tag,
            has_chat_template=has_chat,
            supports_thinking_toggle=supports_thinking,
            requires_trust_remote_code=requires_remote,
            transformers_version=tf_ver,
            fit_by_target=fit_by_target,
        )
        if not allow_non_transformers and _is_non_transformers_weights_repo(
            repo_id, tags, model_type, safetensors
        ):
            fallback_results.append(info)
            continue
        results.append(info)
        if len(results) >= limit:
            break

    return results if results else fallback_results[:limit]


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = load_json(path)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _load_lightweight_json(
    repo_id: str,
    filename: str,
    revision: str,
    local_root: Path | None,
) -> dict[str, Any] | None:
    if local_root is not None:
        local_file = local_root / filename
        data = _read_json_if_exists(local_file)
        if data is not None:
            return data

    cached = try_to_load_from_cache(repo_id=repo_id, filename=filename, revision=revision)
    if isinstance(cached, str):
        data = _read_json_if_exists(Path(cached))
        if data is not None:
            return data

    if local_root is not None:
        return None

    try:
        downloaded = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        return _read_json_if_exists(Path(downloaded))
    except Exception:
        return None


def inspect_model_metadata(
    repo_id: str,
    revision: str = "main",
    local_dir: str | None = None,
) -> ModelInfo:
    """Inspect model metadata using local files or tiny config downloads (never full weights)."""
    resolved_local: Path | None = None
    if local_dir:
        candidate = Path(local_dir)
        if candidate.is_dir():
            resolved_local = candidate
    else:
        reg = _load_registry()
        entry = reg.get(_registry_key(repo_id, revision))
        if entry and entry.get("local_path"):
            candidate = Path(str(entry["local_path"]))
            if candidate.is_dir():
                resolved_local = candidate

    config_json = _load_lightweight_json(repo_id, "config.json", revision, resolved_local)
    tok_json = _load_lightweight_json(repo_id, "tokenizer_config.json", revision, resolved_local)

    if resolved_local is not None and tok_json is not None and not tok_json.get("chat_template"):
        jinja_path = resolved_local / "chat_template.jinja"
        if jinja_path.is_file():
            tok_json = {**tok_json, "chat_template": jinja_path.read_text(encoding="utf-8")}

    hub_info: Any = None
    if resolved_local is None:
        try:
            hub_info = HfApi().model_info(repo_id=repo_id, revision=revision)
        except Exception:
            hub_info = None

    hub_config = getattr(hub_info, "config", None)
    merged_config: dict[str, Any] = {}
    if isinstance(hub_config, dict):
        merged_config.update(hub_config)
    if isinstance(config_json, dict):
        merged_config.update(config_json)

    model_type = str(merged_config.get("model_type", "") or "")
    tags = [str(t) for t in (getattr(hub_info, "tags", None) or [])]
    pipeline_tag = getattr(hub_info, "pipeline_tag", None)
    safetensors = getattr(hub_info, "safetensors", None)

    param_count_b = extract_param_count_b(repo_id, safetensors=safetensors, config=merged_config)
    estimated_size_gb = estimate_model_size_gb(param_count_b)
    arch = infer_architecture_family(
        repo_id=repo_id,
        tags=tags,
        pipeline_tag=pipeline_tag,
        model_type=model_type,
        config=merged_config,
    )
    has_chat, supports_thinking = _detect_chat_and_thinking(
        repo_id=repo_id,
        model_type=model_type,
        tokenizer_config=tok_json,
        architecture_family=arch,
    )
    requires_remote, tf_ver = _detect_remote_code_and_version(merged_config, tags)

    return ModelInfo(
        repo_id=repo_id,
        revision=revision,
        local_path=str(resolved_local) if resolved_local is not None else None,
        architecture_family=arch,
        model_type=model_type,
        param_count_b=param_count_b,
        estimated_size_gb=estimated_size_gb,
        downloads=int(getattr(hub_info, "downloads", 0) or 0),
        likes=int(getattr(hub_info, "likes", 0) or 0),
        license=_extract_license(hub_info) if hub_info is not None else None,
        pipeline_tag=pipeline_tag,
        has_chat_template=has_chat,
        supports_thinking_toggle=supports_thinking,
        requires_trust_remote_code=requires_remote,
        transformers_version=tf_ver,
        fit_by_target={},
    )


def pull_model(repo_id: str, revision: str = "main") -> ModelInfo:
    """Download a model snapshot to HF cache and record it in Decision Forge's registry."""
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    local_path = str(
        snapshot_download(repo_id=repo_id, revision=revision, ignore_patterns=FOREIGN_WEIGHT_PATTERNS)
    )

    info = inspect_model_metadata(repo_id=repo_id, revision=revision, local_dir=local_path)
    info.local_path = local_path

    registry = _load_registry()
    registry[_registry_key(repo_id, revision)] = asdict(info)
    save_json(_registry_path(), registry)
    return info


def list_cached_models() -> list[ModelInfo]:
    """List models registered in Decision Forge's local models.json cache."""
    registry = _load_registry()
    models: list[ModelInfo] = []
    for entry in registry.values():
        try:
            models.append(ModelInfo.from_dict(entry))
        except (TypeError, KeyError):
            continue
    return models


def _collect_special_token_ids(tokenizer: Any) -> set[int]:
    special_ids: set[int] = set()
    raw_special = getattr(tokenizer, "all_special_ids", None)
    if raw_special is not None:
        for tok_id in raw_special:
            if isinstance(tok_id, int) and not isinstance(tok_id, bool):
                special_ids.add(tok_id)
    for attr in (
        "unk_token_id",
        "pad_token_id",
        "eos_token_id",
        "bos_token_id",
        "cls_token_id",
        "sep_token_id",
        "mask_token_id",
    ):
        val = getattr(tokenizer, attr, None)
        if isinstance(val, int) and not isinstance(val, bool) and val >= 0:
            special_ids.add(val)
    return special_ids


def _is_valid_token_id(tok_id: Any, special_ids: set[int]) -> bool:
    return (
        isinstance(tok_id, int)
        and not isinstance(tok_id, bool)
        and tok_id >= 0
        and tok_id not in special_ids
    )


def validate_option_tokens(
    tokenizer: Any,
    codes: str | list[str] = OPTION_LETTERS,
) -> dict[str, int]:
    """Resolve each option code to a single unique non-special tokenizer token ID."""
    code_list = list(codes)
    special_ids = _collect_special_token_ids(tokenizer)
    mapping: dict[str, int] = {}
    used_ids: dict[int, str] = {}

    for code in code_list:
        resolved_id: int | None = None

        encode_fn = getattr(tokenizer, "encode", None)
        if callable(encode_fn):
            try:
                ids = encode_fn(code, add_special_tokens=False)
                if isinstance(ids, (list, tuple)) and len(ids) == 1 and _is_valid_token_id(ids[0], special_ids):
                    resolved_id = ids[0]
            except Exception:
                resolved_id = None

        if resolved_id is None:
            convert_fn = getattr(tokenizer, "convert_tokens_to_ids", None)
            if callable(convert_fn):
                try:
                    tid = convert_fn(code)
                    if _is_valid_token_id(tid, special_ids):
                        resolved_id = tid
                except Exception:
                    resolved_id = None

        if resolved_id is None and callable(encode_fn):
            try:
                space_ids = encode_fn(" " + code, add_special_tokens=False)
                if (
                    isinstance(space_ids, (list, tuple))
                    and len(space_ids) == 1
                    and _is_valid_token_id(space_ids[0], special_ids)
                ):
                    resolved_id = space_ids[0]
            except Exception:
                resolved_id = None

        if resolved_id is None:
            raise ValueError(
                f"Cannot resolve option code {code!r} to a single non-special token ID."
            )
        if resolved_id in used_ids:
            raise ValueError(
                f"Option code {code!r} collided with {used_ids[resolved_id]!r} on token ID {resolved_id}."
            )

        used_ids[resolved_id] = code
        mapping[code] = resolved_id

    return mapping
