"""Load, cache, validate, and build few-shot pools for DecideBench v2 and v1."""

from __future__ import annotations

import json
import logging
import random
import warnings
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import httpx

from decision_forge.constants import (
    APP_CACHE_DIR_NAME,
    DECIDEBENCH_GITHUB_RAW,
    DECIDEBENCH_REPO_ID,
    DECIDEBENCH_V1_FAMILIES,
    DECIDEBENCH_V2_FAMILIES,
)
from decision_forge.types import DecisionItem, DecisionOption

logger = logging.getLogger(__name__)

TemplateKey = tuple[str, str, tuple[tuple[str, str], ...]]


def template_key(item: DecisionItem) -> TemplateKey:
    """Return the canonical DecideBench template key matching `decidebench/fewshot.py`."""
    return (
        item.category,
        item.question,
        tuple((o.key, o.description) for o in item.options),
    )


def _normalize_options(raw_options: Any) -> list[DecisionOption]:
    """Parse DecideBench options from list-of-dicts or HuggingFace columnar dict-of-lists."""
    if isinstance(raw_options, dict):
        keys = raw_options.get("key") or []
        descs = raw_options.get("description") or [""] * len(keys)
        return [DecisionOption(key=str(k), description=str(d)) for k, d in zip(keys, descs, strict=False)]
    if isinstance(raw_options, (list, tuple)):
        out: list[DecisionOption] = []
        for entry in raw_options:
            if isinstance(entry, DecisionOption):
                out.append(entry)
            elif isinstance(entry, dict):
                out.append(DecisionOption(key=str(entry["key"]), description=str(entry.get("description", ""))))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                out.append(DecisionOption(key=str(entry[0]), description=str(entry[1])))
        return out
    return []


def parse_decidebench_row(
    obj: dict[str, Any],
    *,
    source: str = "decidebench_v2",
    is_example: bool = False,
) -> DecisionItem:
    """Convert a raw DecideBench JSONL or HuggingFace dataset row into a `DecisionItem`."""
    options = _normalize_options(obj.get("options"))
    keys = [o.key for o in options]
    gold_raw = obj.get("gold")
    if isinstance(gold_raw, int) and 0 <= gold_raw < len(options):
        gold_idx = gold_raw
    elif isinstance(gold_raw, str) and gold_raw in keys:
        gold_idx = keys.index(gold_raw)
    elif isinstance(obj.get("gold_idx"), int) and 0 <= obj["gold_idx"] < len(options):
        gold_idx = int(obj["gold_idx"])
    else:
        gold_idx = -1

    pair_id = str(obj.get("pair_id") or obj.get("template_id") or "")
    difficulty = str(obj.get("difficulty") or ("example" if is_example else ""))
    return DecisionItem(
        id=str(obj.get("id", "")),
        state=str(obj.get("state", "")),
        question=str(obj.get("question", "")),
        options=options,
        gold_idx=gold_idx,
        primitive="choice",
        source=source,
        category=str(obj.get("category", "")),
        pair_id=pair_id,
        difficulty=difficulty,
        examples=[],
    )


def _item_to_record(item: DecisionItem, *, is_example: bool = False) -> dict[str, Any]:
    gold = item.options[item.gold_idx].key if 0 <= item.gold_idx < len(item.options) else ""
    rec: dict[str, Any] = {
        "id": item.id,
        "category": item.category,
        "state": item.state,
        "question": item.question,
        "options": [{"key": o.key, "description": o.description} for o in item.options],
        "gold": gold,
    }
    if is_example:
        rec["template_id"] = item.pair_id
    else:
        rec["pair_id"] = item.pair_id
        rec["difficulty"] = item.difficulty
    return rec


def validate_decidebench_pairs(items: list[DecisionItem]) -> tuple[list[DecisionItem], list[str]]:
    """Validate item fields and contrastive pair integrity (`pair_id` + `a`/`b`), stripping corrupt pairs."""
    problems: list[str] = []
    seen_ids: set[str] = set()
    by_pair: dict[str, list[DecisionItem]] = defaultdict(list)

    for it in items:
        if not it.id:
            problems.append("Skipping item with empty id")
            continue
        if it.id in seen_ids:
            problems.append(f"{it.id}: duplicate id; stripping duplicate")
            continue
        seen_ids.add(it.id)

        if not 2 <= len(it.options) <= 10:
            problems.append(f"{it.id}: expected 2-10 options, got {len(it.options)}")
            continue
        if len(set(it.keys)) != len(it.keys):
            problems.append(f"{it.id}: duplicate option keys")
            continue
        if not 0 <= it.gold_idx < len(it.options):
            problems.append(f"{it.id}: invalid gold_idx {it.gold_idx}")
            continue
        if not it.pair_id or it.id not in (f"{it.pair_id}a", f"{it.pair_id}b"):
            problems.append(f"{it.id}: id must be pair_id + 'a'/'b' (got pair_id={it.pair_id!r})")
            continue
        by_pair[it.pair_id].append(it)

    valid_items: list[DecisionItem] = []
    for pid in sorted(by_pair.keys()):
        members = sorted(by_pair[pid], key=lambda m: m.id)
        if len(members) != 2:
            problems.append(f"{pid}: expected 2 halves, found {len(members)}; stripping pair")
            continue
        a, b = members
        if a.id != f"{pid}a" or b.id != f"{pid}b":
            problems.append(f"{pid}: halves must be {pid}a and {pid}b; stripping pair")
            continue
        if a.question != b.question or a.options != b.options:
            problems.append(f"{pid}: halves differ in question or options; stripping pair")
            continue
        if a.gold_idx == b.gold_idx:
            problems.append(f"{pid}: halves share identical gold; stripping pair")
            continue
        if a.state == b.state:
            problems.append(f"{pid}: halves share identical state; stripping pair")
            continue
        valid_items.extend((a, b))

    for msg in problems:
        logger.warning("DecideBench validation: %s", msg)
        warnings.warn(f"DecideBench validation: {msg}", UserWarning, stacklevel=2)

    return valid_items, problems


def select_balanced_pairs(items: list[DecisionItem], limit: int | None) -> list[DecisionItem]:
    """Select whole contrastive pairs deterministically and balanced across task families."""
    if limit is None:
        return list(items)
    if limit <= 0 or not items:
        return []

    target_pairs = max(1, limit // 2)
    pairs_by_id: dict[str, list[DecisionItem]] = defaultdict(list)
    for it in items:
        pairs_by_id[it.pair_id].append(it)

    pairs_by_cat: dict[str, list[tuple[DecisionItem, DecisionItem]]] = defaultdict(list)
    for pid in sorted(pairs_by_id.keys()):
        members = sorted(pairs_by_id[pid], key=lambda m: m.id)
        if len(members) == 2:
            pairs_by_cat[members[0].category].append((members[0], members[1]))

    categories = sorted(pairs_by_cat.keys())
    if not categories:
        return []

    selected: list[DecisionItem] = []
    pair_count = 0
    round_idx = 0
    while pair_count < target_pairs:
        added_this_round = False
        for cat in categories:
            cat_pairs = pairs_by_cat[cat]
            if round_idx < len(cat_pairs):
                a, b = cat_pairs[round_idx]
                selected.extend((a, b))
                pair_count += 1
                added_this_round = True
                if pair_count >= target_pairs:
                    break
        if not added_this_round:
            break
        round_idx += 1

    return selected


def _load_jsonl_items(
    path: Path,
    *,
    source: str,
    is_example: bool,
) -> list[DecisionItem]:
    items: list[DecisionItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(parse_decidebench_row(json.loads(line), source=source, is_example=is_example))
    return items


def _write_jsonl_items(path: Path, items: list[DecisionItem], *, is_example: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(_item_to_record(it, is_example=is_example), ensure_ascii=False) for it in items]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _download_from_hf(version: Literal["v2", "v1"]) -> tuple[list[DecisionItem], list[DecisionItem]]:
    from datasets import DatasetDict, load_dataset  # Lazy import to avoid heavy startup when cached

    test_cfg = "default" if version == "v2" else "v1"
    ex_cfg = "examples" if version == "v2" else "v1-examples"
    source = f"decidebench_{version}"

    ds_test = load_dataset(DECIDEBENCH_REPO_ID, name=test_cfg, split="test")
    if isinstance(ds_test, DatasetDict):
        ds_test = ds_test["test"] if "test" in ds_test else next(iter(ds_test.values()))

    ds_ex = load_dataset(DECIDEBENCH_REPO_ID, name=ex_cfg, split="train")
    if isinstance(ds_ex, DatasetDict):
        ds_ex = ds_ex["train"] if "train" in ds_ex else next(iter(ds_ex.values()))

    test_items = [parse_decidebench_row(dict(row), source=source, is_example=False) for row in ds_test]
    ex_items = [parse_decidebench_row(dict(row), source=f"{source}_examples", is_example=True) for row in ds_ex]
    return test_items, ex_items


def _download_from_github_raw(version: Literal["v2", "v1"]) -> tuple[list[DecisionItem], list[DecisionItem]]:
    families = DECIDEBENCH_V2_FAMILIES if version == "v2" else DECIDEBENCH_V1_FAMILIES
    source = f"decidebench_{version}"
    test_items: list[DecisionItem] = []
    ex_items: list[DecisionItem] = []

    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        for fam in families:
            t_resp = client.get(f"{DECIDEBENCH_GITHUB_RAW}/data/{version}/{fam}.jsonl")
            t_resp.raise_for_status()
            for line in t_resp.text.splitlines():
                if line.strip():
                    test_items.append(parse_decidebench_row(json.loads(line), source=source, is_example=False))

            e_resp = client.get(f"{DECIDEBENCH_GITHUB_RAW}/data/{version}/examples/{fam}.jsonl")
            e_resp.raise_for_status()
            for line in e_resp.text.splitlines():
                if line.strip():
                    ex_items.append(
                        parse_decidebench_row(json.loads(line), source=f"{source}_examples", is_example=True)
                    )

    return test_items, ex_items


def build_example_pool(
    items: list[DecisionItem],
    examples: list[DecisionItem],
) -> dict[TemplateKey, list[DecisionItem]]:
    """Build the `template_key -> worked examples` pool matching `decidebench/fewshot.py`."""
    ids_by_key: dict[TemplateKey, str] = {}
    counts: dict[str, int] = {}
    for it in items:
        k = template_key(it)
        if k not in ids_by_key:
            counts[it.category] = counts.get(it.category, 0) + 1
            ids_by_key[k] = f"{it.category}-t{counts[it.category]:02d}"

    key_by_tid = {tid: k for k, tid in ids_by_key.items()}
    pool: dict[TemplateKey, list[DecisionItem]] = {k: [] for k in ids_by_key}

    for ex in examples:
        k = key_by_tid.get(ex.pair_id) if ex.pair_id in key_by_tid else template_key(ex)
        pool.setdefault(k, []).append(ex)

    return pool


def attach_fewshot_examples(
    items: list[DecisionItem],
    pool: dict[TemplateKey, list[DecisionItem]],
    protocol: Literal["few_shot", "zero_shot"],
) -> list[DecisionItem]:
    """Return copies of `items` with deterministic worked examples attached for `few_shot`, or empty for `zero_shot`."""
    if protocol == "zero_shot":
        return [replace(it, examples=[]) for it in items]
    if protocol != "few_shot":
        raise ValueError(f"Unsupported protocol: {protocol!r}")

    out: list[DecisionItem] = []
    for it in items:
        if it.difficulty == "example":
            out.append(replace(it, examples=[]))
            continue

        shots = list(pool.get(template_key(it), []))
        random.Random(it.id).shuffle(shots)
        serialized: list[dict[str, Any]] = []
        for s in shots:
            if isinstance(s, dict):
                serialized.append(dict(s))
            else:
                gold_key = s.options[s.gold_idx].key if 0 <= s.gold_idx < len(s.options) else ""
                serialized.append(
                    {
                        "id": s.id,
                        "state": s.state,
                        "gold": gold_key,
                        "gold_idx": s.gold_idx,
                        "answer": gold_key,
                    }
                )
        out.append(replace(it, examples=serialized))
    return out


def load_decidebench(
    version: Literal["v2", "v1"] = "v2",
    limit: int | None = None,
    cache_dir: Path | None = None,
) -> tuple[list[DecisionItem], dict[TemplateKey, list[DecisionItem]]]:
    """Load and validate DecideBench (`v2` or `v1`) items and worked-example pool.

    Loads from `cache_dir` when available; otherwise downloads `choyiny/decidebench` from
    Hugging Face Hub with automatic fallback to GitHub raw JSONL files, validates contrastive
    pair integrity, and applies pair-preserving balanced `limit` when requested.
    """
    if version not in ("v2", "v1"):
        raise ValueError(f"Unsupported DecideBench version: {version!r}")

    root_cache = Path(cache_dir) if cache_dir is not None else (Path.home() / APP_CACHE_DIR_NAME / "decidebench")
    ver_cache = root_cache / version
    test_cache_file = ver_cache / "test.jsonl"
    ex_cache_file = ver_cache / "examples.jsonl"
    source = f"decidebench_{version}"

    if test_cache_file.is_file() and ex_cache_file.is_file():
        raw_items = _load_jsonl_items(test_cache_file, source=source, is_example=False)
        raw_examples = _load_jsonl_items(ex_cache_file, source=f"{source}_examples", is_example=True)
    else:
        try:
            raw_items, raw_examples = _download_from_hf(version)
            if not raw_items:
                raise ValueError("Empty HuggingFace DecideBench split")
        except Exception as hf_exc:
            logger.info("HF load_dataset fallback to GitHub raw for %s (%s)", version, hf_exc)
            raw_items, raw_examples = _download_from_github_raw(version)

        _write_jsonl_items(test_cache_file, raw_items, is_example=False)
        _write_jsonl_items(ex_cache_file, raw_examples, is_example=True)

    valid_items, _ = validate_decidebench_pairs(raw_items)
    pool = build_example_pool(valid_items, raw_examples)
    selected_items = select_balanced_pairs(valid_items, limit)
    return selected_items, pool
