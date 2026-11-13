"""Prompt rendering, option code mapping, option shuffling, and token packing."""

from __future__ import annotations

import random
import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import torch

from decision_forge.constants import OPTION_LETTERS, SCORE_DIGITS
from decision_forge.types import DecisionItem, DecisionOption

_NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
_PREFIX_NUM_RE = re.compile(r"^\d+\s*[-:.)]")
_KEY_SCORE_RE = re.compile(r"^(?:score|rating|level)[_\s-]*\d+$", re.IGNORECASE)


def _is_numeric_like_option(opt: DecisionOption) -> bool:
    key = opt.key.strip()
    desc = opt.description.strip()
    if _NUM_RE.match(key) or _KEY_SCORE_RE.match(key):
        return True
    if _NUM_RE.match(desc) or _PREFIX_NUM_RE.match(desc):
        return True
    return False


def get_option_codes(item: DecisionItem) -> list[str]:
    """Return canonical single-character option codes for a DecisionItem."""
    n = len(item.options)
    if n == 0:
        return []
    if n > len(OPTION_LETTERS):
        raise ValueError(f"Too many options ({n}); maximum supported is {len(OPTION_LETTERS)}.")

    if item.primitive == "noul":
        if n == 2:
            return ["A", "B"]
        return list(OPTION_LETTERS[:n])

    if item.primitive == "score":
        if n <= len(SCORE_DIGITS) and all(_is_numeric_like_option(o) for o in item.options):
            return list(SCORE_DIGITS[:n])
        return list(OPTION_LETTERS[:n])

    return list(OPTION_LETTERS[:n])


def _format_example(idx: int, ex: dict[str, Any]) -> str:
    """Format a single worked example dict without leaking target labels."""
    lines: list[str] = [f"Example {idx}:"]
    ex_state = ex.get("state") or ex.get("context") or ex.get("input") or ex.get("text")
    if ex_state:
        lines.append(f"State:\n{str(ex_state).strip()}")

    ex_question = ex.get("question")
    if ex_question:
        lines.append(f"QUESTION: {str(ex_question).strip()}")

    ex_options = ex.get("options")
    codes_for_ex: list[str] = []
    if isinstance(ex_options, list) and ex_options:
        lines.append("Options:")
        for opt_idx, opt in enumerate(ex_options):
            code = OPTION_LETTERS[opt_idx] if opt_idx < len(OPTION_LETTERS) else str(opt_idx + 1)
            codes_for_ex.append(code)
            if isinstance(opt, DecisionOption):
                opt_text = opt.description.strip() or opt.key.strip()
            elif isinstance(opt, Mapping):
                opt_text = str(opt.get("description") or opt.get("text") or opt.get("key") or "").strip()
            else:
                opt_text = str(opt).strip()
            lines.append(f"{code}) {opt_text}")

    ans = (
        ex.get("answer")
        if ex.get("answer") is not None
        else ex.get("code")
        if ex.get("code") is not None
        else ex.get("choice")
        if ex.get("choice") is not None
        else ex.get("output")
        if ex.get("output") is not None
        else ex.get("label")
    )
    if ans is None and isinstance(ex.get("gold_idx"), int):
        gidx = ex["gold_idx"]
        if 0 <= gidx < len(codes_for_ex):
            ans = codes_for_ex[gidx]
    if ans is not None:
        lines.append(f"Answer: {ans}")
    return "\n".join(lines)


def build_prompt_text(
    item: DecisionItem,
    option_codes: list[str] | None = None,
    include_examples: bool = True,
) -> str:
    """Render state, optional worked examples, question, and option lines."""
    codes = list(option_codes) if option_codes is not None else get_option_codes(item)
    if len(codes) != len(item.options):
        raise ValueError(
            f"option_codes length ({len(codes)}) must match item.options length ({len(item.options)})."
        )

    sections: list[str] = []
    if include_examples and item.examples:
        ex_blocks = [_format_example(i + 1, ex) for i, ex in enumerate(item.examples)]
        sections.append("Examples:\n" + "\n\n".join(ex_blocks))

    if item.state and item.state.strip():
        sections.append(f"State:\n{item.state.strip()}")

    sections.append(f"QUESTION: {item.question.strip()}")

    option_lines = ["Options:"]
    for code, opt in zip(codes, item.options):
        opt_text = opt.description.strip() or opt.key.strip()
        option_lines.append(f"{code}) {opt_text}")
    sections.append("\n".join(option_lines))

    sections.append("Reply with the option code only.")
    return "\n\n".join(sections)


def _unpack_encoded(encoded: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize BatchEncoding, dict, tensor, or sequence outputs into 1D tensors."""
    if isinstance(encoded, Mapping) or (
        hasattr(encoded, "__getitem__")
        and not isinstance(encoded, (torch.Tensor, list, tuple, str, bytes))
    ):
        raw_ids = encoded["input_ids"]
        raw_mask = encoded["attention_mask"] if "attention_mask" in encoded else None
    elif isinstance(encoded, torch.Tensor):
        raw_ids = encoded
        raw_mask = None
    elif isinstance(encoded, (list, tuple)):
        raw_ids = torch.tensor(encoded, dtype=torch.long)
        raw_mask = None
    else:
        raise TypeError(f"Unsupported tokenizer output type: {type(encoded)!r}")

    ids = raw_ids if isinstance(raw_ids, torch.Tensor) else torch.tensor(raw_ids, dtype=torch.long)
    if ids.dim() == 2 and ids.shape[0] == 1:
        ids = ids.squeeze(0)
    elif ids.dim() != 1:
        ids = ids.reshape(-1)
    ids = ids.to(dtype=torch.long)

    if raw_mask is None:
        mask = torch.ones_like(ids, dtype=torch.long)
    else:
        mask = raw_mask if isinstance(raw_mask, torch.Tensor) else torch.tensor(raw_mask, dtype=torch.long)
        if mask.dim() == 2 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        elif mask.dim() != 1:
            mask = mask.reshape(-1)
        mask = mask.to(dtype=torch.long)

    return ids, mask


def _encode_with_tokenizer(
    tokenizer: Any,
    prompt_text: str,
    disable_thinking: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode prompt using chat template if available, else plain tokenizer call."""
    has_template = bool(getattr(tokenizer, "chat_template", None))
    if has_template:
        messages = [{"role": "user", "content": prompt_text}]
        kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        if disable_thinking:
            try:
                encoded = tokenizer.apply_chat_template(
                    messages,
                    enable_thinking=False,
                    **kwargs,
                )
            except TypeError:
                encoded = tokenizer.apply_chat_template(messages, **kwargs)
        else:
            encoded = tokenizer.apply_chat_template(messages, **kwargs)

        if isinstance(encoded, str):
            encoded = tokenizer(encoded, return_tensors="pt")
        return _unpack_encoded(encoded)

    encoded = tokenizer(prompt_text, return_tensors="pt")
    return _unpack_encoded(encoded)


def _tail_truncate(
    ids: torch.Tensor,
    mask: torch.Tensor,
    max_seq_len: int,
    bos_token_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncate from the left to preserve options, instruction, and answer slot."""
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
    if ids.numel() <= max_seq_len:
        return ids, mask

    if (
        bos_token_id is not None
        and ids.numel() > 0
        and int(ids[0].item()) == int(bos_token_id)
        and max_seq_len > 1
    ):
        keep = max_seq_len - 1
        return (
            torch.cat([ids[:1], ids[-keep:]], dim=0),
            torch.cat([mask[:1], mask[-keep:]], dim=0),
        )
    return ids[-max_seq_len:], mask[-max_seq_len:]


def render_model_inputs(
    tokenizer: Any,
    item: DecisionItem,
    disable_thinking: bool = True,
    max_seq_len: int = 512,
    option_codes: list[str] | None = None,
) -> dict[str, Any]:
    """Render prompt tokens without appending any ground-truth token."""
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
    codes = list(option_codes) if option_codes is not None else get_option_codes(item)
    prompt = build_prompt_text(item, option_codes=codes, include_examples=True)
    ids, mask = _encode_with_tokenizer(tokenizer, prompt, disable_thinking=disable_thinking)

    if ids.numel() > max_seq_len and item.examples:
        prompt_no_ex = build_prompt_text(item, option_codes=codes, include_examples=False)
        ids, mask = _encode_with_tokenizer(tokenizer, prompt_no_ex, disable_thinking=disable_thinking)

    bos_id = getattr(tokenizer, "bos_token_id", None)
    ids, mask = _tail_truncate(ids, mask, max_seq_len=max_seq_len, bos_token_id=bos_id)

    if ids.numel() == 0:
        raise ValueError("Tokenizer returned empty input_ids for prompt.")

    return {
        "input_ids": ids,
        "attention_mask": mask,
        "answer_pos": int(ids.numel() - 1),
        "option_codes": codes,
    }


def render_set_inputs(
    tokenizer: Any,
    state: str,
    questions: list[DecisionItem],
    disable_thinking: bool = True,
    max_seq_len: int = 1024,
) -> dict[str, Any]:
    """Pack shared state followed by independent question+answer_slot segments."""
    if not questions:
        raise ValueError("questions must be non-empty.")
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

    state_text = f"State:\n{state.strip()}\n" if state.strip() else "State:\n"
    state_encoded = tokenizer(state_text, return_tensors="pt")
    state_ids, _ = _unpack_encoded(state_encoded)

    bos_id = getattr(tokenizer, "bos_token_id", None)
    has_state_bos = (
        bos_id is not None and state_ids.numel() > 0 and int(state_ids[0].item()) == int(bos_id)
    )

    q_chunks: list[torch.Tensor] = []
    option_codes_per_q: list[list[str]] = []
    for q in questions:
        q_only = replace(q, state="")
        rendered = render_model_inputs(
            tokenizer,
            q_only,
            disable_thinking=disable_thinking,
            max_seq_len=max_seq_len,
        )
        q_ids = rendered["input_ids"]
        if (
            has_state_bos
            and bos_id is not None
            and q_ids.numel() > 1
            and int(q_ids[0].item()) == int(bos_id)
        ):
            q_ids = q_ids[1:]
        q_chunks.append(q_ids)
        option_codes_per_q.append(rendered["option_codes"])

    q_total = sum(int(chunk.numel()) for chunk in q_chunks)
    total_len = int(state_ids.numel()) + q_total
    if total_len > max_seq_len:
        state_budget = max_seq_len - q_total
        if state_budget >= 0:
            state_ids = state_ids[:state_budget]
        else:
            min_state = min(int(state_ids.numel()), max(1, max_seq_len // (len(q_chunks) + 1)))
            state_ids = state_ids[:min_state]
            remaining = max_seq_len - int(state_ids.numel())
            per_q_budget = max(1, remaining // len(q_chunks))
            trimmed_q: list[torch.Tensor] = []
            for idx, q_ids in enumerate(q_chunks):
                budget = per_q_budget + (1 if idx < (remaining % len(q_chunks)) else 0)
                budget = max(1, budget)
                trimmed_q.append(q_ids[-budget:] if q_ids.numel() > budget else q_ids)
            q_chunks = trimmed_q

    chunks: list[torch.Tensor] = []
    segment_spans: list[tuple[int, int, str]] = []
    answer_positions: list[int] = []
    cursor = 0

    if state_ids.numel() > 0:
        chunks.append(state_ids)
        end = cursor + int(state_ids.numel())
        segment_spans.append((cursor, end, "state"))
        cursor = end

    for idx, q_ids in enumerate(q_chunks):
        chunks.append(q_ids)
        end = cursor + int(q_ids.numel())
        segment_spans.append((cursor, end, f"q_{idx}"))
        answer_positions.append(end - 1)
        cursor = end

    all_ids = torch.cat(chunks, dim=0)
    all_mask = torch.ones_like(all_ids, dtype=torch.long)

    return {
        "input_ids": all_ids,
        "attention_mask": all_mask,
        "segment_spans": segment_spans,
        "answer_positions": answer_positions,
        "option_codes_per_q": option_codes_per_q,
    }


def shuffle_item_options(item: DecisionItem, rng: random.Random) -> DecisionItem:
    """Randomly permute item.options and update gold_idx consistently."""
    n = len(item.options)
    if n <= 1:
        return replace(item, options=list(item.options))

    perm = list(range(n))
    rng.shuffle(perm)
    new_options = [item.options[i] for i in perm]
    if 0 <= item.gold_idx < n:
        new_gold_idx = perm.index(item.gold_idx)
    else:
        new_gold_idx = item.gold_idx

    return replace(item, options=new_options, gold_idx=new_gold_idx)
