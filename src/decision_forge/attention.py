"""4D attention mask lifting (bidirectional, set, causal) and behavioral probing."""

from __future__ import annotations

import math
from typing import Literal

import torch

MaskMode = Literal["bidirectional", "set", "causal"]


def build_4d_attention_mask(
    attention_mask_2d: torch.Tensor,
    mode: MaskMode,
    dtype: torch.dtype,
    segment_spans: list[list[tuple[int, int, str]]] | None = None,
) -> torch.Tensor:
    """Build a (B, 1, L, L) additive float attention mask from a 2D (B, L) pad mask."""
    if not dtype.is_floating_point:
        raise ValueError(f"4D attention mask dtype must be floating point, got {dtype}.")

    if attention_mask_2d.dim() == 1:
        mask_2d = attention_mask_2d.unsqueeze(0)
    elif attention_mask_2d.dim() == 2:
        mask_2d = attention_mask_2d
    else:
        raise ValueError(f"Expected 1D or 2D attention_mask, got shape {tuple(attention_mask_2d.shape)}.")

    batch_size, seq_len = mask_2d.shape
    device = mask_2d.device
    valid = mask_2d.bool()
    min_val = torch.finfo(dtype).min

    if mode == "bidirectional":
        allowed = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    elif mode == "causal":
        tril = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)).unsqueeze(0)
        allowed = tril & valid.unsqueeze(-1) & valid.unsqueeze(-2)
    elif mode == "set":
        if segment_spans is None:
            raise ValueError("segment_spans must be provided when mode='set'.")
        if len(segment_spans) != batch_size:
            raise ValueError(
                f"segment_spans length ({len(segment_spans)}) must match batch size ({batch_size})."
            )

        allowed = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool, device=device)
        for b in range(batch_size):
            state_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            all_q_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            q_groups: dict[str, torch.Tensor] = {}

            for start, end, label in segment_spans[b]:
                s = max(0, min(seq_len, int(start)))
                e = max(s, min(seq_len, int(end)))
                if s >= e:
                    continue
                if label == "state":
                    state_mask[s:e] = True
                else:
                    all_q_mask[s:e] = True
                    group = q_groups.setdefault(
                        label, torch.zeros(seq_len, dtype=torch.bool, device=device)
                    )
                    group[s:e] = True

            state_keys = state_mask | all_q_mask
            allowed[b] |= state_mask.unsqueeze(-1) & state_keys.unsqueeze(-2)

            for q_mask in q_groups.values():
                q_keys = state_mask | q_mask
                allowed[b] |= q_mask.unsqueeze(-1) & q_keys.unsqueeze(-2)

            allowed[b] &= valid[b].unsqueeze(-1) & valid[b].unsqueeze(-2)
    else:
        raise ValueError(f"Unsupported 4D attention mask mode: {mode!r}")

    mask_4d = torch.full((batch_size, 1, seq_len, seq_len), min_val, dtype=dtype, device=device)
    mask_4d.masked_fill_(allowed.unsqueeze(1), 0.0)
    return mask_4d


def _resolve_probe_mask_dtype(
    model: torch.nn.Module,
    device: torch.device,
    requested_dtype: torch.dtype,
) -> torch.dtype:
    if torch.is_autocast_enabled(device.type):
        ac_dtype = torch.get_autocast_dtype(device.type)
        if ac_dtype.is_floating_point:
            return ac_dtype
    for param in model.parameters():
        if param.is_floating_point():
            return param.dtype
    return requested_dtype if requested_dtype.is_floating_point else torch.float32


def _extract_hidden_for_probe(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask_4d: torch.Tensor,
) -> torch.Tensor:
    if not hasattr(model, "peft_config") and hasattr(model, "model") and callable(model.model):
        try:
            backbone_out = model.model(input_ids=input_ids, attention_mask=attention_mask_4d)
            if hasattr(backbone_out, "last_hidden_state") and backbone_out.last_hidden_state is not None:
                return backbone_out.last_hidden_state
        except Exception:
            pass

    try:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask_4d,
            output_hidden_states=True,
        )
    except TypeError:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask_4d,
        )
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state
    if hasattr(out, "hidden_states") and out.hidden_states:
        return out.hidden_states[-1]
    if hasattr(out, "logits") and out.logits is not None:
        return out.logits
    raise RuntimeError("Model output did not expose last_hidden_state, hidden_states, or logits.")


def probe_bidirectional_support(
    model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[bool, str]:
    """Behaviorally probe whether a model honors a 4D bidirectional attention mask."""
    was_training = model.training
    model.eval()
    try:
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
        if not isinstance(vocab_size, int) or vocab_size < 2:
            emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
            if emb is not None and hasattr(emb, "weight"):
                vocab_size = int(emb.weight.shape[0])
            else:
                vocab_size = 32000

        if vocab_size >= 6:
            x1 = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
            x2 = torch.tensor([[1, 2, 3, 5]], dtype=torch.long, device=device)
        else:
            x1 = torch.tensor(
                [[0, min(1, vocab_size - 1), min(2, vocab_size - 1), 0]],
                dtype=torch.long,
                device=device,
            )
            x2 = torch.tensor(
                [[0, min(1, vocab_size - 1), min(2, vocab_size - 1), vocab_size - 1]],
                dtype=torch.long,
                device=device,
            )

        mask_dtype = _resolve_probe_mask_dtype(model, device, dtype)
        attn_2d = torch.ones((1, 4), dtype=torch.long, device=device)
        mask_4d = build_4d_attention_mask(attn_2d, "bidirectional", mask_dtype)

        with torch.no_grad():
            h1 = _extract_hidden_for_probe(model, x1, mask_4d)
            h2 = _extract_hidden_for_probe(model, x2, mask_4d)

        diff = float((h1[:, 0] - h2[:, 0]).abs().max().item())
        if not math.isfinite(diff):
            return False, f"bidirectional probe produced non-finite difference ({diff})"
        if diff > 1e-5:
            return True, f"bidirectional probe passed (pos0_diff={diff:.6g})"
        return False, f"model ignored 4D bidirectional mask (pos0_diff={diff:.6g} <= 1e-5)"
    except Exception as exc:
        return False, f"bidirectional probe raised {type(exc).__name__}: {exc}"
    finally:
        model.train(was_training)
