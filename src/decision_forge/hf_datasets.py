"""Hugging Face dataset discovery, preview, schema inference, canary guard, and loaders."""

from __future__ import annotations

import csv
import hashlib
import inspect
import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import datasets
import httpx
from huggingface_hub import HfApi

from decision_forge.constants import (
    DECIDEBENCH_CANARY_GUID,
    GENERAL_MIX_RECIPES,
    OPTION_LETTERS,
    SCORE_DIGITS,
)
from decision_forge.types import DatasetSpec, DecisionItem, DecisionOption, PrimitiveKind

DATASETS_SERVER_FIRST_ROWS_URL = "https://datasets-server.huggingface.co/first-rows"

CLASSIFICATION_TEXT_COLS = (
    "text",
    "sentence",
    "review",
    "content",
    "document",
    "email",
    "message",
    "body",
    "headline",
    "title",
    "utterance",
    "input",
    "prompt",
    "query",
)
CLASSIFICATION_LABEL_COLS = (
    "label",
    "category",
    "intent",
    "sentiment",
    "class",
    "topic",
    "target",
    "y",
)
MC_CONTEXT_COLS = (
    "context",
    "passage",
    "article",
    "state",
    "text",
    "story",
    "background",
    "document",
    "scenario",
)
MC_QUESTION_COLS = ("question", "query", "stem", "prompt")
MC_OPTIONS_COLS = ("choices", "options", "candidates", "answers", "endings")
MC_LABEL_COLS = ("answerkey", "answer_key", "answer", "gold_idx", "label", "target", "correct")
NLI_LABEL_COLS = ("label", "gold_label", "target", "category")
CANONICAL_NLI_ORDER = ("entailment", "neutral", "contradiction")


def _supports_list_datasets_direction(api: HfApi) -> bool:
    try:
        return "direction" in inspect.signature(api.list_datasets).parameters
    except (TypeError, ValueError):
        return False


def _deterministic_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def search_datasets(
    query: str,
    task_category: str | None = None,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Search Hugging Face Hub datasets by query and optional task category."""
    api = HfApi()
    kwargs: dict[str, Any] = {
        "search": query or None,
        "task_categories": task_category or None,
        "sort": "downloads",
        "limit": limit,
        "expand": ["downloads", "likes", "tags", "description"],
    }
    if _supports_list_datasets_direction(api):
        kwargs["direction"] = -1

    results: list[dict[str, Any]] = []
    for ds in api.list_datasets(**kwargs):
        repo_id = str(getattr(ds, "id", "") or "")
        if not repo_id:
            continue
        tags = [str(t) for t in (getattr(ds, "tags", None) or [])]
        task_cats = [
            t.split(":", 1)[1]
            for t in tags
            if t.lower().startswith("task_categories:") and ":" in t
        ]
        results.append(
            {
                "repo_id": repo_id,
                "downloads": int(getattr(ds, "downloads", 0) or 0),
                "likes": int(getattr(ds, "likes", 0) or 0),
                "tags": tags,
                "task_categories": task_cats,
                "description": str(getattr(ds, "description", "") or ""),
                "gated": bool(getattr(ds, "gated", False)),
            }
        )
    return results


def preview_dataset_rows(
    repo_id: str,
    config_name: str | None = None,
    split: str = "train",
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Fetch first rows via datasets-server API with fast timeout, falling back to streaming."""
    if limit <= 0:
        return []

    params = {
        "dataset": repo_id,
        "config": config_name or "default",
        "split": split,
    }
    try:
        with httpx.Client(timeout=5.0, follow_redirects=True) as client:
            resp = client.get(DATASETS_SERVER_FIRST_ROWS_URL, params=params)
            resp.raise_for_status()
            payload = resp.json()
        raw_rows = payload.get("rows", []) if isinstance(payload, dict) else []
        parsed: list[dict[str, Any]] = []
        for entry in raw_rows[:limit]:
            if isinstance(entry, dict) and isinstance(entry.get("row"), dict):
                parsed.append(dict(entry["row"]))
            elif isinstance(entry, dict):
                parsed.append(dict(entry))
        if parsed:
            return parsed
    except Exception:
        pass

    ds = datasets.load_dataset(repo_id, name=config_name, split=split, streaming=True)
    if isinstance(ds, Mapping):
        ds = ds[split] if split in ds else next(iter(ds.values()))
    return [dict(row) for row in itertools.islice(ds, limit)]


def _extract_names_from_feature_obj(feat: Any) -> list[str] | None:
    if isinstance(feat, datasets.ClassLabel):
        return [str(n) for n in feat.names]
    names = getattr(feat, "names", None)
    if isinstance(names, (list, tuple)) and names:
        return [str(n) for n in names]
    if isinstance(feat, dict):
        if feat.get("_type") == "ClassLabel" and isinstance(feat.get("names"), (list, tuple)):
            return [str(n) for n in feat["names"]]
        if isinstance(feat.get("type"), dict):
            return _extract_names_from_feature_obj(feat["type"])
    return None


def _extract_class_label_names(
    features: Any,
    label_col: str | None,
) -> tuple[str | None, list[str] | None]:
    if features is None:
        return label_col, None

    feature_map: dict[str, Any] = {}
    if isinstance(features, Mapping):
        feature_map = {str(k): v for k, v in features.items()}
    elif isinstance(features, list):
        for entry in features:
            if isinstance(entry, dict) and "name" in entry:
                feature_map[str(entry["name"])] = entry.get("type", entry)

    if label_col and label_col in feature_map:
        names = _extract_names_from_feature_obj(feature_map[label_col])
        if names is not None:
            return label_col, names

    if label_col:
        lower_map = {k.lower(): k for k in feature_map}
        matched_key = lower_map.get(label_col.lower())
        if matched_key is not None:
            names = _extract_names_from_feature_obj(feature_map[matched_key])
            if names is not None:
                return matched_key, names

    if not label_col:
        for col_name, feat in feature_map.items():
            names = _extract_names_from_feature_obj(feat)
            if names is not None:
                return col_name, names

    return label_col, None


def _collect_columns(rows: list[dict[str, Any]], features: Any) -> list[str]:
    cols: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                cols.append(key)
    if isinstance(features, Mapping):
        for key in features:
            skey = str(key)
            if skey not in seen:
                seen.add(skey)
                cols.append(skey)
    elif isinstance(features, list):
        for entry in features:
            if isinstance(entry, dict) and "name" in entry:
                skey = str(entry["name"])
                if skey not in seen:
                    seen.add(skey)
                    cols.append(skey)
    return cols


def _match_first_col(lower_map: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def infer_dataset_schema(
    rows: list[dict[str, Any]],
    features: Any = None,
) -> dict[str, Any]:
    """Infer dataset schema type (classification, nli, multiple_choice) and columns."""
    columns = _collect_columns(rows, features)
    lower_map = {c.lower(): c for c in columns}

    # 1. NLI: premise + hypothesis (+ label)
    premise_col = lower_map.get("premise")
    hypothesis_col = lower_map.get("hypothesis")
    if premise_col and hypothesis_col:
        label_col = _match_first_col(lower_map, NLI_LABEL_COLS)
        label_col, label_names = _extract_class_label_names(features, label_col)
        return {
            "schema_type": "nli",
            "state_col": premise_col,
            "second_text_col": hypothesis_col,
            "question_col": None,
            "options_col": None,
            "label_col": label_col,
            "label_names": label_names,
            "columns": columns,
        }

    # 2. Multiple choice: optional context/passage/article + question + choices/options + answerKey/answer/label
    options_col = _match_first_col(lower_map, MC_OPTIONS_COLS)
    question_col = _match_first_col(lower_map, MC_QUESTION_COLS)
    context_col = _match_first_col(lower_map, MC_CONTEXT_COLS)
    if context_col == question_col:
        context_col = None
    mc_label_col = _match_first_col(lower_map, MC_LABEL_COLS)
    if options_col and (question_col or context_col):
        mc_label_col, label_names = _extract_class_label_names(features, mc_label_col)
        return {
            "schema_type": "multiple_choice",
            "state_col": context_col,
            "second_text_col": None,
            "question_col": question_col,
            "options_col": options_col,
            "label_col": mc_label_col,
            "label_names": label_names,
            "columns": columns,
        }

    # 3. Classification: text/sentence/review/content/document/email + label/category
    state_col = _match_first_col(lower_map, CLASSIFICATION_TEXT_COLS)
    label_col = _match_first_col(lower_map, CLASSIFICATION_LABEL_COLS)
    label_col, label_names = _extract_class_label_names(features, label_col)

    if state_col is None and rows:
        for col in columns:
            if col == label_col:
                continue
            sample_vals = [r.get(col) for r in rows[:5] if r.get(col) is not None]
            if sample_vals and all(isinstance(v, str) for v in sample_vals):
                state_col = col
                break

    if label_names is None and label_col and rows:
        raw_labels = [r.get(label_col) for r in rows if r.get(label_col) is not None]
        if raw_labels and all(isinstance(v, str) for v in raw_labels):
            label_names = sorted({str(v) for v in raw_labels})

    return {
        "schema_type": "classification",
        "state_col": state_col,
        "second_text_col": None,
        "question_col": None,
        "options_col": None,
        "label_col": label_col,
        "label_names": label_names,
        "columns": columns,
    }


def filter_canary_items(items: list[DecisionItem]) -> tuple[list[DecisionItem], int]:
    """Strip any DecisionItem containing DECIDEBENCH_CANARY_GUID (case-insensitive)."""
    needle = DECIDEBENCH_CANARY_GUID.lower()
    kept: list[DecisionItem] = []
    stripped = 0
    for item in items:
        fields = [item.state, item.question]
        for opt in item.options:
            fields.append(opt.key)
            fields.append(opt.description)
        if any(needle in (val or "").lower() for val in fields):
            stripped += 1
        else:
            kept.append(item)
    return kept, stripped


def compute_dataset_diagnostics(items: list[DecisionItem]) -> dict[str, Any]:
    """Compute summary counts, class/option balance, imbalance ratio, and warnings."""
    total = len(items)
    if total == 0:
        return {
            "total_count": 0,
            "class_balance": {},
            "gold_key_balance": {},
            "imbalance_ratio": 0.0,
            "avg_option_count": 0.0,
            "min_option_count": 0,
            "max_option_count": 0,
            "sources": {},
            "warnings": ["Dataset is empty (0 items)."],
        }

    class_counts: Counter[str] = Counter()
    gold_key_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    option_lengths: list[int] = []
    invalid_gold_count = 0
    few_options_count = 0

    for item in items:
        n_opts = len(item.options)
        option_lengths.append(n_opts)
        if n_opts < 2:
            few_options_count += 1
        if 0 <= item.gold_idx < n_opts:
            gold_opt = item.options[item.gold_idx]
            label_key = gold_opt.description.strip() or gold_opt.key
            class_counts[label_key] += 1
            gold_key_counts[gold_opt.key] += 1
        else:
            invalid_gold_count += 1
        if item.source:
            source_counts[item.source] += 1

    avg_option_count = round(sum(option_lengths) / total, 3)
    if len(class_counts) >= 2:
        counts_list = list(class_counts.values())
        imbalance_ratio = round(max(counts_list) / max(1, min(counts_list)), 3)
    elif len(class_counts) == 1:
        imbalance_ratio = 1.0
    else:
        imbalance_ratio = 0.0

    warnings: list[str] = []
    if total < 20:
        warnings.append("Small dataset (< 20 items); metrics and calibration may be noisy.")
    if few_options_count > 0:
        warnings.append(f"{few_options_count} item(s) have fewer than 2 options.")
    if invalid_gold_count > 0:
        warnings.append(f"{invalid_gold_count} item(s) have missing or out-of-range gold_idx.")
    if len(class_counts) == 1 and total > 1:
        warnings.append("Only 1 gold class is represented across items.")
    if imbalance_ratio >= 5.0:
        warnings.append(f"High class imbalance detected (ratio={imbalance_ratio:.2f}).")
    if total >= 10 and avg_option_count >= 2.0 and gold_key_counts:
        top_key_count = max(gold_key_counts.values())
        if top_key_count / total > 0.6:
            warnings.append(
                "Gold option key distribution is skewed; enable shuffle_options or PRIDE debiasing."
            )

    return {
        "total_count": total,
        "class_balance": dict(class_counts),
        "gold_key_balance": dict(gold_key_counts),
        "imbalance_ratio": imbalance_ratio,
        "avg_option_count": avg_option_count,
        "min_option_count": min(option_lengths),
        "max_option_count": max(option_lengths),
        "sources": dict(source_counts),
        "warnings": warnings,
    }


def _option_codes_for_primitive(primitive: PrimitiveKind, max_options: int) -> str:
    alphabet = SCORE_DIGITS if primitive == "score" else OPTION_LETTERS
    cap = min(max(max_options, 2), len(alphabet))
    return alphabet[:cap]


def _sample_options_with_gold(
    all_descriptions: list[str],
    gold_description: str,
    codes: str,
    seed: int,
    item_key: str,
) -> tuple[list[DecisionOption], int]:
    cap = len(codes)
    if len(all_descriptions) <= cap:
        selected = list(all_descriptions)
        gold_idx = selected.index(gold_description)
    else:
        rng = random.Random(_deterministic_seed(seed, item_key))
        distractors = [d for d in all_descriptions if d != gold_description]
        chosen_distractors = rng.sample(distractors, k=min(cap - 1, len(distractors)))
        selected = [gold_description, *chosen_distractors]
        rng.shuffle(selected)
        gold_idx = selected.index(gold_description)

    options = [DecisionOption(key=codes[i], description=desc) for i, desc in enumerate(selected)]
    return options, gold_idx


def _parse_mc_choices(raw_choices: Any) -> tuple[list[str], list[str] | None]:
    if isinstance(raw_choices, str):
        stripped = raw_choices.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                raw_choices = json.loads(stripped)
            except json.JSONDecodeError:
                pass

    if isinstance(raw_choices, dict):
        texts = raw_choices.get("text") or raw_choices.get("texts") or raw_choices.get("options") or []
        labels = raw_choices.get("label") or raw_choices.get("labels") or raw_choices.get("key")
        descs = [str(t) for t in texts]
        keys = [str(k) for k in labels] if isinstance(labels, (list, tuple)) else None
        return descs, keys

    if isinstance(raw_choices, (list, tuple)):
        if not raw_choices:
            return [], None
        if isinstance(raw_choices[0], dict):
            descs = []
            keys = []
            for entry in raw_choices:
                desc = entry.get("description", entry.get("text", entry.get("label", "")))
                descs.append(str(desc))
                key = entry.get("key", entry.get("label"))
                if key is not None:
                    keys.append(str(key))
            return descs, keys if len(keys) == len(descs) else None
        return [str(x) for x in raw_choices], None

    return [], None


def _resolve_mc_gold_idx(
    raw_answer: Any,
    descriptions: list[str],
    raw_keys: list[str] | None,
) -> int:
    if raw_answer is None or not descriptions:
        return -1

    if raw_keys is not None:
        ans_str = str(raw_answer).strip()
        for idx, k in enumerate(raw_keys):
            if k.strip().lower() == ans_str.lower():
                return idx

    if isinstance(raw_answer, int) and not isinstance(raw_answer, bool):
        return raw_answer if 0 <= raw_answer < len(descriptions) else -1

    ans_str = str(raw_answer).strip()
    if ans_str.isdigit():
        idx = int(ans_str)
        if 0 <= idx < len(descriptions):
            return idx

    upper = ans_str.upper()
    if len(upper) == 1 and upper in OPTION_LETTERS:
        letter_idx = OPTION_LETTERS.index(upper)
        if 0 <= letter_idx < len(descriptions):
            return letter_idx

    for idx, desc in enumerate(descriptions):
        if desc.strip() == ans_str:
            return idx

    return -1


def _resolve_class_order(
    rows: list[dict[str, Any]],
    label_col: str | None,
    label_names: list[str] | None,
    schema_type: str,
) -> tuple[list[str], dict[Any, str]]:
    if label_names:
        mapping: dict[Any, str] = {}
        for idx, name in enumerate(label_names):
            mapping[idx] = name
            mapping[str(idx)] = name
            mapping[name] = name
        return list(label_names), mapping

    if not label_col:
        return [], {}

    raw_vals = [r.get(label_col) for r in rows if r.get(label_col) is not None]
    if not raw_vals:
        return [], {}

    if all(isinstance(v, int) and not isinstance(v, bool) for v in raw_vals):
        unique_ints = sorted({int(v) for v in raw_vals if int(v) >= 0})
        classes = [str(v) for v in unique_ints]
        mapping = {v: str(v) for v in unique_ints}
        mapping.update({str(v): str(v) for v in unique_ints})
        return classes, mapping

    unique_strs = sorted({str(v).strip() for v in raw_vals if str(v).strip()})
    if schema_type == "nli" and set(unique_strs).issubset(set(CANONICAL_NLI_ORDER)):
        classes = [c for c in CANONICAL_NLI_ORDER if c in unique_strs]
    else:
        classes = unique_strs
    return classes, {c: c for c in classes}


def _rows_to_decision_items(
    rows: list[dict[str, Any]],
    schema: dict[str, Any],
    spec: DatasetSpec,
    seed: int,
    source_name: str,
) -> list[DecisionItem]:
    schema_type = schema["schema_type"]
    state_col = spec.state_col or schema.get("state_col")
    second_col = spec.second_text_col or schema.get("second_text_col")
    question_col = spec.question_col or schema.get("question_col")
    options_col = spec.options_col or schema.get("options_col")
    label_col = spec.label_col or schema.get("label_col")
    label_names: list[str] | None = schema.get("label_names")

    codes = _option_codes_for_primitive(spec.primitive, spec.max_options_per_item)
    items: list[DecisionItem] = []

    if schema_type == "multiple_choice":
        for idx, row in enumerate(rows):
            descs, raw_keys = _parse_mc_choices(row.get(options_col)) if options_col else ([], None)
            if len(descs) < 2:
                continue
            raw_ans = row.get(label_col) if label_col else None
            gold_idx = _resolve_mc_gold_idx(raw_ans, descs, raw_keys)
            if gold_idx < 0 and label_col is not None:
                continue

            item_id = str(row.get("id") or f"{source_name}:{idx}")
            state = str(row.get(state_col, "") or "") if state_col else ""
            question = (
                str(row.get(question_col, "") or "") if question_col else spec.default_question
            ) or spec.default_question

            if len(descs) > len(codes):
                if 0 <= gold_idx < len(descs):
                    gold_desc = descs[gold_idx]
                    options, final_gold_idx = _sample_options_with_gold(
                        descs, gold_desc, codes, seed=seed, item_key=item_id
                    )
                else:
                    trimmed = descs[: len(codes)]
                    options = [
                        DecisionOption(key=codes[i], description=d) for i, d in enumerate(trimmed)
                    ]
                    final_gold_idx = -1
            else:
                options = [
                    DecisionOption(key=codes[i], description=d) for i, d in enumerate(descs)
                ]
                final_gold_idx = gold_idx

            items.append(
                DecisionItem(
                    id=item_id,
                    state=state,
                    question=question,
                    options=options,
                    gold_idx=final_gold_idx,
                    primitive=spec.primitive,
                    source=source_name,
                    category=str(row.get("category", source_name) or source_name),
                )
            )
        return items

    all_classes, label_map = _resolve_class_order(rows, label_col, label_names, schema_type)
    if len(all_classes) < 2:
        return []

    for idx, row in enumerate(rows):
        raw_label = row.get(label_col) if label_col else None
        if raw_label is None:
            continue
        if isinstance(raw_label, int) and not isinstance(raw_label, bool) and raw_label < 0:
            continue
        gold_desc = label_map.get(raw_label)
        if gold_desc is None:
            gold_desc = label_map.get(str(raw_label).strip())
        if gold_desc is None:
            continue

        if schema_type == "nli" and state_col and second_col:
            premise = str(row.get(state_col, "") or "")
            hypothesis = str(row.get(second_col, "") or "")
            state = f"Premise: {premise}\nHypothesis: {hypothesis}"
        else:
            state = str(row.get(state_col, "") or "") if state_col else json.dumps(row, ensure_ascii=False)

        question = (
            str(row.get(question_col, "") or "") if question_col else spec.default_question
        ) or spec.default_question

        item_id = str(row.get("id") or f"{source_name}:{idx}")
        options, gold_idx = _sample_options_with_gold(
            all_classes,
            gold_desc,
            codes=codes,
            seed=seed,
            item_key=item_id,
        )
        items.append(
            DecisionItem(
                id=item_id,
                state=state,
                question=question,
                options=options,
                gold_idx=gold_idx,
                primitive=spec.primitive,
                source=source_name,
                category=source_name,
            )
        )

    return items


def _read_local_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Local dataset file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]

    if suffix in (".jsonl", ".ndjson"):
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                obj = json.loads(stripped)
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected JSON object on line {line_no} of {path}")
                rows.append(obj)
        return rows

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(r) for r in payload if isinstance(r, dict)]
        if isinstance(payload, dict):
            for key in ("data", "items", "rows", "examples"):
                if isinstance(payload.get(key), list):
                    return [dict(r) for r in payload[key] if isinstance(r, dict)]
        raise ValueError(f"Unsupported JSON root structure in {path}")

    raise ValueError(f"Unsupported local dataset extension {suffix!r}; expected .csv, .jsonl, or .json")


def _split_train_val(
    items: list[DecisionItem],
    spec: DatasetSpec,
    seed: int,
    already_shuffled: bool = False,
) -> tuple[list[DecisionItem], list[DecisionItem]]:
    if not items:
        return [], []

    pool = list(items)
    if not already_shuffled:
        random.Random(seed).shuffle(pool)

    total = len(pool)
    if total == 1 or spec.val_ratio <= 0 or spec.max_val_samples <= 0:
        return pool[: spec.max_train_samples], []

    val_count = min(spec.max_val_samples, max(1, round(total * spec.val_ratio)))
    val_count = min(val_count, total - 1)
    val_items = pool[:val_count]
    train_items = pool[val_count : val_count + spec.max_train_samples]
    return train_items, val_items


def _load_hf_rows_and_features(
    repo_id: str,
    config_name: str | None,
    split: str,
    target_rows: int,
    seed: int,
) -> tuple[list[dict[str, Any]], Any]:
    use_streaming = "[" not in split
    ds: Any = None
    if use_streaming:
        try:
            ds = datasets.load_dataset(repo_id, name=config_name, split=split, streaming=True)
        except TypeError:
            ds = datasets.load_dataset(repo_id, name=config_name, split=split)
        except ValueError as exc:
            if "streaming" in str(exc).lower():
                ds = datasets.load_dataset(repo_id, name=config_name, split=split)
            else:
                raise
    else:
        ds = datasets.load_dataset(repo_id, name=config_name, split=split)

    if isinstance(ds, Mapping):
        ds = ds[split] if split in ds else next(iter(ds.values()))

    features = getattr(ds, "features", None)

    if hasattr(ds, "select") and hasattr(ds, "__len__"):
        total_len = len(ds)
        if hasattr(ds, "shuffle"):
            try:
                ds = ds.shuffle(seed=seed)
            except Exception:
                pass
        take_n = min(total_len, target_rows)
        ds = ds.select(range(take_n))
        return [dict(r) for r in ds], features

    if hasattr(ds, "shuffle"):
        try:
            ds = ds.shuffle(seed=seed, buffer_size=min(5000, max(target_rows * 2, 1000)))
        except TypeError:
            pass
        except Exception:
            pass

    rows = [dict(r) for r in itertools.islice(ds, target_rows)]
    return rows, features


def _build_schema_for_spec(
    rows: list[dict[str, Any]],
    features: Any,
    spec: DatasetSpec,
) -> dict[str, Any]:
    inferred = infer_dataset_schema(rows, features=features)
    if spec.schema_type != "auto":
        inferred["schema_type"] = spec.schema_type
    if spec.state_col:
        inferred["state_col"] = spec.state_col
    if spec.second_text_col:
        inferred["second_text_col"] = spec.second_text_col
    if spec.question_col:
        inferred["question_col"] = spec.question_col
    if spec.options_col:
        inferred["options_col"] = spec.options_col
    if spec.label_col:
        inferred["label_col"] = spec.label_col
        _, explicit_names = _extract_class_label_names(features, spec.label_col)
        if explicit_names is not None:
            inferred["label_names"] = explicit_names
    return inferred


def _load_single_hf_items(
    spec: DatasetSpec,
    target_rows: int,
    seed: int,
    source_name: str,
) -> tuple[list[DecisionItem], int, dict[str, Any]]:
    if not spec.repo_id:
        raise ValueError("DatasetSpec.repo_id is required when mode == 'hf_dataset'.")

    rows, features = _load_hf_rows_and_features(
        repo_id=spec.repo_id,
        config_name=spec.config_name,
        split=spec.split,
        target_rows=target_rows,
        seed=seed,
    )
    schema = _build_schema_for_spec(rows, features, spec)
    raw_items = _rows_to_decision_items(
        rows=rows,
        schema=schema,
        spec=spec,
        seed=seed,
        source_name=source_name,
    )
    clean_items, canary_stripped = filter_canary_items(raw_items)
    return clean_items, canary_stripped, schema


def _select_mix_recipes(mix_datasets: list[str]) -> list[dict[str, object]]:
    if not mix_datasets:
        return list(GENERAL_MIX_RECIPES)
    wanted = {m.strip().lower() for m in mix_datasets if m.strip()}
    selected = [
        dict(recipe)
        for recipe in GENERAL_MIX_RECIPES
        if str(recipe["name"]).lower() in wanted or str(recipe["repo_id"]).lower() in wanted
    ]
    if not selected:
        raise ValueError(f"No GENERAL_MIX_RECIPES matched mix_datasets={mix_datasets!r}.")
    return selected


def load_decision_dataset(
    spec: DatasetSpec,
    seed: int = 42,
) -> tuple[list[DecisionItem], list[DecisionItem], dict[str, Any]]:
    """Load, adapt, canary-filter, and split a dataset according to DatasetSpec."""
    if spec.mode == "zero_shot":
        return [], [], {"mode": "zero_shot"}

    if spec.mode == "local_file":
        if not spec.local_path:
            raise ValueError("DatasetSpec.local_path is required when mode == 'local_file'.")
        path = Path(spec.local_path)
        rows = _read_local_rows(path)
        schema = _build_schema_for_spec(rows, features=None, spec=spec)
        raw_items = _rows_to_decision_items(
            rows=rows,
            schema=schema,
            spec=spec,
            seed=seed,
            source_name=path.stem,
        )
        clean_items, canary_stripped = filter_canary_items(raw_items)
        train_items, val_items = _split_train_val(clean_items, spec, seed=seed)
        diagnostics = compute_dataset_diagnostics(train_items + val_items)
        return train_items, val_items, {
            "mode": "local_file",
            "local_path": str(path),
            "schema": schema,
            "canary_stripped": canary_stripped,
            "diagnostics": diagnostics,
        }

    if spec.mode == "hf_dataset":
        target_total = max(1, spec.max_train_samples + spec.max_val_samples)
        fetch_target = target_total + min(64, max(8, target_total // 10))
        clean_items, canary_stripped, schema = _load_single_hf_items(
            spec=spec,
            target_rows=fetch_target,
            seed=seed,
            source_name=spec.repo_id or "hf_dataset",
        )
        train_items, val_items = _split_train_val(clean_items, spec, seed=seed)
        diagnostics = compute_dataset_diagnostics(train_items + val_items)
        return train_items, val_items, {
            "mode": "hf_dataset",
            "repo_id": spec.repo_id,
            "config_name": spec.config_name,
            "split": spec.split,
            "schema": schema,
            "canary_stripped": canary_stripped,
            "diagnostics": diagnostics,
        }

    if spec.mode == "auto_general_mix":
        recipes = _select_mix_recipes(spec.mix_datasets)
        target_total = max(1, spec.max_train_samples + spec.max_val_samples)
        per_recipe_target = max(1, math.ceil(target_total / len(recipes)))
        fetch_per_recipe = per_recipe_target + min(32, max(8, per_recipe_target // 5))

        recipe_buckets: list[list[DecisionItem]] = []
        recipes_loaded: list[str] = []
        failed_recipes: dict[str, str] = {}
        total_canary_stripped = 0

        for idx, recipe in enumerate(recipes):
            recipe_name = str(recipe["name"])
            recipe_max_opts = int(recipe.get("max_options_per_item", spec.max_options_per_item))
            recipe_spec = DatasetSpec(
                mode="hf_dataset",
                repo_id=str(recipe["repo_id"]),
                config_name=recipe.get("config_name"),  # type: ignore[arg-type]
                split=str(recipe.get("split", "train")),
                schema_type=str(recipe.get("schema_type", "auto")),  # type: ignore[arg-type]
                state_col=recipe.get("state_col"),  # type: ignore[arg-type]
                second_text_col=recipe.get("second_text_col"),  # type: ignore[arg-type]
                label_col=recipe.get("label_col"),  # type: ignore[arg-type]
                default_question=str(recipe.get("default_question", spec.default_question)),
                primitive=str(recipe.get("primitive", spec.primitive)),  # type: ignore[arg-type]
                max_train_samples=fetch_per_recipe,
                max_val_samples=0,
                val_ratio=0.0,
                max_options_per_item=min(recipe_max_opts, spec.max_options_per_item),
            )
            try:
                items, stripped, _ = _load_single_hf_items(
                    spec=recipe_spec,
                    target_rows=fetch_per_recipe,
                    seed=seed + idx + 1,
                    source_name=recipe_name,
                )
                total_canary_stripped += stripped
                if items:
                    random.Random(seed + idx + 1).shuffle(items)
                    recipe_buckets.append(items)
                    recipes_loaded.append(recipe_name)
                else:
                    failed_recipes[recipe_name] = "Loaded 0 valid items."
            except Exception as exc:
                failed_recipes[recipe_name] = str(exc)

        if not recipe_buckets:
            raise RuntimeError(f"All auto_general_mix recipes failed: {failed_recipes}")

        # Interleave round-robin across succeeded recipes for balanced coverage up to target_total
        balanced_pool: list[DecisionItem] = []
        max_bucket_len = max(len(b) for b in recipe_buckets)
        for pos in range(max_bucket_len):
            for bucket in recipe_buckets:
                if pos < len(bucket):
                    balanced_pool.append(bucket[pos])
                    if len(balanced_pool) >= target_total:
                        break
            if len(balanced_pool) >= target_total:
                break

        balanced_pool, extra_stripped = filter_canary_items(balanced_pool)
        total_canary_stripped += extra_stripped

        rng = random.Random(seed)
        rng.shuffle(balanced_pool)
        train_items, val_items = _split_train_val(
            balanced_pool, spec, seed=seed, already_shuffled=True
        )
        diagnostics = compute_dataset_diagnostics(train_items + val_items)
        return train_items, val_items, {
            "mode": "auto_general_mix",
            "recipes_loaded": recipes_loaded,
            "failed_recipes": failed_recipes,
            "canary_stripped": total_canary_stripped,
            "diagnostics": diagnostics,
        }

    raise ValueError(f"Unsupported DatasetSpec.mode: {spec.mode!r}")
