"""Single-pass DecisionEngine inference, PriDe cyclic debiasing, set-batching, and /v1/systemone."""

from __future__ import annotations

import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from decision_forge.attention import build_4d_attention_mask, probe_bidirectional_support
from decision_forge.calibration import bucket_for_item
from decision_forge.constants import OPTION_LETTERS, SCORE_DIGITS
from decision_forge.prompts import render_model_inputs, render_set_inputs
from decision_forge.types import (
    ArchitectureFamily,
    CalibrationReport,
    DecisionConfig,
    DecisionItem,
    DecisionOption,
    DecisionPrediction,
    PrimitiveKind,
    load_json,
)

_ENCODER_MODEL_TYPES = {
    "bert",
    "roberta",
    "xlm-roberta",
    "deberta",
    "deberta-v2",
    "modernbert",
    "electra",
    "albert",
    "distilbert",
    "mpnet",
}
_LEADING_FLOAT_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)")


def _encode_no_special(tokenizer: Any, text: str) -> list[int]:
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(text)
    if isinstance(ids, torch.Tensor):
        return [int(x) for x in ids.reshape(-1).tolist()]
    return [int(x) for x in ids]


def resolve_label_token_ids(tokenizer: Any, codes: list[str]) -> list[int]:
    """Resolve option codes to unique single vocabulary token IDs."""
    unk_id = getattr(tokenizer, "unk_token_id", None)
    resolved: list[int] = []
    seen: set[int] = set()

    for code in codes:
        tid: int | None = None
        for candidate in (code, " " + code):
            ids = _encode_no_special(tokenizer, candidate)
            if len(ids) == 1 and ids[0] != unk_id:
                tid = int(ids[0])
                break

        if tid is None and hasattr(tokenizer, "convert_tokens_to_ids"):
            for tok in (code, f"\u2581{code}", f"\u0120{code}"):
                cand_id = tokenizer.convert_tokens_to_ids(tok)
                if (
                    isinstance(cand_id, int)
                    and cand_id >= 0
                    and cand_id != unk_id
                ):
                    tid = int(cand_id)
                    break

        if tid is None:
            raise ValueError(
                f"Failed to resolve option code {code!r} to a single non-unk token ID."
            )
        if tid in seen:
            raise ValueError(
                f"Option code {code!r} resolved to duplicate token ID {tid}."
            )
        seen.add(tid)
        resolved.append(tid)

    return resolved


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(dtype: str | torch.dtype, device: torch.device) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    norm = str(dtype).lower().strip()
    if norm in ("float32", "fp32"):
        return torch.float32
    if norm in ("bfloat16", "bf16"):
        return torch.bfloat16
    if norm in ("float16", "fp16", "half"):
        return torch.float16
    if norm == "auto":
        if device.type == "cpu":
            return torch.float32
        if device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        if device.type == "xpu":
            return torch.bfloat16
        if device.type == "mps":
            return torch.float16
        return torch.float32
    raise ValueError(f"Unsupported dtype specification: {dtype!r}")


def _load_hf_model(
    cls_or_auto: Any,
    source: str,
    torch_dtype: torch.dtype,
    attn_implementation: str | None = None,
    trust_remote_code: bool = False,
) -> Any:
    kwargs: dict[str, Any] = {"dtype": torch_dtype}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    try:
        return cls_or_auto.from_pretrained(source, **kwargs)
    except TypeError:
        # Compatibility fallback for older transformers (<5.x) expecting torch_dtype
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = torch_dtype
        try:
            return cls_or_auto.from_pretrained(source, **kwargs)
        except (TypeError, ValueError):
            kwargs.pop("attn_implementation", None)
            return cls_or_auto.from_pretrained(source, **kwargs)
    except (ValueError, NotImplementedError) as exc:
        if not trust_remote_code and ("trust_remote_code" in str(exc) or "custom code" in str(exc).lower()):
            kwargs["trust_remote_code"] = True
            return _load_hf_model(
                cls_or_auto,
                source,
                torch_dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )
        if attn_implementation == "sdpa":
            kwargs["attn_implementation"] = "eager"
            try:
                return cls_or_auto.from_pretrained(source, **kwargs)
            except (TypeError, ValueError, NotImplementedError):
                kwargs.pop("attn_implementation", None)
                return cls_or_auto.from_pretrained(source, **kwargs)
        kwargs.pop("attn_implementation", None)
        return cls_or_auto.from_pretrained(source, **kwargs)


def _log_softmax_1d(arr: np.ndarray) -> np.ndarray:
    max_v = float(np.max(arr))
    shifted = arr - max_v
    return shifted - float(np.log(np.sum(np.exp(shifted))))


def _softmax_1d(arr: np.ndarray) -> np.ndarray:
    return np.exp(_log_softmax_1d(arr))


def _score_option_values(options: list[DecisionOption]) -> list[float]:
    values: list[float] = []
    for idx, opt in enumerate(options):
        key_match = _LEADING_FLOAT_RE.match(opt.key)
        if key_match:
            values.append(float(key_match.group(1)))
            continue
        desc_match = _LEADING_FLOAT_RE.match(opt.description)
        if desc_match:
            values.append(float(desc_match.group(1)))
            continue
        values.append(float(idx + 1))
    return values


def _compute_score_stats(
    options: list[DecisionOption],
    probs: np.ndarray,
) -> tuple[float, float]:
    values = _score_option_values(options)
    expected = float(sum(float(p) * v for p, v in zip(probs, values)))

    paired = sorted(zip(values, probs), key=lambda pair: pair[0])
    cum = 0.0
    median = paired[-1][0]
    for val, prob in paired:
        cum += float(prob)
        if cum >= 0.5 - 1e-12:
            median = float(val)
            break
    return expected, median


class DecisionEngine:
    """Zero-generation single-pass inference engine for Decision Models."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: DecisionConfig,
        calibration: CalibrationReport | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.calibration = calibration
        self.device = torch.device(device)
        self.dtype = dtype

        if getattr(self.tokenizer, "pad_token_id", None) is None:
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_id is not None:
                self.tokenizer.pad_token_id = eos_id
                if getattr(self.tokenizer, "eos_token", None) is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token_id = 0

        self.model.eval()
        try:
            first_param = next(self.model.parameters(), None)
            if first_param is not None and first_param.device != self.device:
                self.model.to(self.device)
        except Exception:
            pass

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        device: str = "auto",
        dtype: str = "auto",
    ) -> DecisionEngine:
        """Load merged model, adapter+base, or raw HF checkpoint for zero-shot inference."""
        target_device = _resolve_device(device)
        target_dtype = _resolve_dtype(dtype, target_device)

        raw_path = Path(model_dir)
        load_dir: Path | str = model_dir
        parent_dir: Path | None = None

        if raw_path.exists():
            if (
                (raw_path / "merged" / "decision_config.json").is_file()
                and not (raw_path / "config.json").is_file()
                and not (raw_path / "adapter_config.json").is_file()
            ):
                parent_dir = raw_path
                load_dir = raw_path / "merged"
            else:
                load_dir = raw_path

        config_obj: DecisionConfig | None = None
        calibration_obj: CalibrationReport | None = None
        is_local = isinstance(load_dir, Path) and load_dir.exists()

        if is_local:
            assert isinstance(load_dir, Path)
            cfg_file = load_dir / "decision_config.json"
            if cfg_file.is_file():
                config_obj = DecisionConfig.from_dict(load_json(cfg_file))
            elif parent_dir is not None and (parent_dir / "decision_config.json").is_file():
                config_obj = DecisionConfig.from_dict(load_json(parent_dir / "decision_config.json"))

            cal_file = load_dir / "calibration.json"
            if cal_file.is_file():
                calibration_obj = CalibrationReport.from_dict(load_json(cal_file))
            elif parent_dir is not None and (parent_dir / "calibration.json").is_file():
                calibration_obj = CalibrationReport.from_dict(load_json(parent_dir / "calibration.json"))

        source_str = str(load_dir)
        is_adapter = isinstance(load_dir, Path) and (load_dir / "adapter_config.json").is_file() and not (
            load_dir / "config.json"
        ).is_file()

        if is_local and is_adapter:
            assert isinstance(load_dir, Path)
            if config_obj is None and (load_dir.parent / "decision_config.json").is_file():
                config_obj = DecisionConfig.from_dict(load_json(load_dir.parent / "decision_config.json"))
            if calibration_obj is None and (load_dir.parent / "calibration.json").is_file():
                calibration_obj = CalibrationReport.from_dict(load_json(load_dir.parent / "calibration.json"))

        adapter_base: str | None = None
        if is_adapter and isinstance(load_dir, Path):
            adapter_cfg = load_json(load_dir / "adapter_config.json")
            adapter_base = adapter_cfg.get("base_model_name_or_path") or (
                config_obj.base_model if config_obj is not None else None
            )
            if not adapter_base:
                raise ValueError(f"Adapter directory {load_dir} has no base_model_name_or_path.")

        tok_source = source_str
        if is_adapter and isinstance(load_dir, Path) and not (
            (load_dir / "tokenizer.json").is_file()
            or (load_dir / "tokenizer_config.json").is_file()
            or (load_dir / "vocab.json").is_file()
        ):
            tok_source = adapter_base or source_str

        hf_cfg_source = adapter_base if is_adapter and adapter_base else source_str
        trust_remote = bool(config_obj.requires_trust_remote_code) if config_obj is not None else False
        try:
            hf_cfg = (
                AutoConfig.from_pretrained(hf_cfg_source, trust_remote_code=True)
                if trust_remote
                else AutoConfig.from_pretrained(hf_cfg_source)
            )
        except ValueError as exc:
            if not trust_remote and ("trust_remote_code" in str(exc) or "custom code" in str(exc).lower()):
                trust_remote = True
                hf_cfg = AutoConfig.from_pretrained(hf_cfg_source, trust_remote_code=True)
            else:
                raise
        if getattr(hf_cfg, "auto_map", None):
            trust_remote = True

        try:
            tokenizer = (
                AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
                if trust_remote
                else AutoTokenizer.from_pretrained(tok_source)
            )
        except ValueError as exc:
            if not trust_remote and ("trust_remote_code" in str(exc) or "custom code" in str(exc).lower()):
                trust_remote = True
                tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
            else:
                raise
        if getattr(tokenizer, "pad_token_id", None) is None:
            if getattr(tokenizer, "eos_token_id", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.pad_token_id = 0

        model_type = str(getattr(hf_cfg, "model_type", "")).lower()
        arch_family: ArchitectureFamily = (
            config_obj.architecture_family
            if config_obj is not None
            else ("encoder" if model_type in _ENCODER_MODEL_TYPES else "decoder")
        )

        desired_attn_mode = config_obj.effective_attention_mode if config_obj else "bidirectional"
        attn_impl = "sdpa" if desired_attn_mode in ("bidirectional", "set") else None

        has_encoder_head = isinstance(load_dir, Path) and (load_dir / "encoder_lm_head.pt").is_file()
        if has_encoder_head or arch_family == "encoder":
            from decision_forge.trainer import EncoderDecisionWrapper

            backbone = _load_hf_model(
                AutoModel,
                hf_cfg_source,
                target_dtype,
                attn_implementation=attn_impl,
                trust_remote_code=trust_remote,
            )
            model = EncoderDecisionWrapper(backbone)
            if has_encoder_head and isinstance(load_dir, Path):
                state_dict = torch.load(load_dir / "encoder_lm_head.pt", map_location="cpu", weights_only=True)
                model.lm_head.load_state_dict(state_dict)
            if is_adapter:
                model = PeftModel.from_pretrained(model, source_str)
        elif is_adapter:
            assert adapter_base is not None
            base_model = _load_hf_model(
                AutoModelForCausalLM,
                adapter_base,
                target_dtype,
                attn_implementation=attn_impl,
                trust_remote_code=trust_remote,
            )
            model = PeftModel.from_pretrained(base_model, source_str)
        else:
            model = _load_hf_model(
                AutoModelForCausalLM,
                source_str,
                target_dtype,
                attn_implementation=attn_impl,
                trust_remote_code=trust_remote,
            )

        model.to(target_device)
        model.eval()

        if config_obj is None:
            if arch_family == "encoder":
                eff_mode = "encoder"
            else:
                supported, _ = probe_bidirectional_support(model, target_device, target_dtype)
                eff_mode = "bidirectional" if supported else "causal"
            config_obj = DecisionConfig(
                model_name=Path(source_str).name if Path(source_str).exists() else source_str,
                base_model=source_str,
                architecture_family=arch_family,
                attention_mode="bidirectional" if arch_family != "encoder" else "encoder",
                effective_attention_mode=eff_mode,
                train_method="zero_shot",
                primitives=["choice", "noul", "score"],
                option_tokens=list(OPTION_LETTERS[:10]) + list(SCORE_DIGITS),
                disable_thinking=True,
                pride_debiasing=False,
                pride_permutations=3,
                max_seq_len=512,
                hardware_backend=target_device.type,
                requires_trust_remote_code=trust_remote,
            )

        return cls(
            model=model,
            tokenizer=tokenizer,
            config=config_obj,
            calibration=calibration_obj,
            device=str(target_device),
            dtype=target_dtype,
        )

    def _mask_dtype(self) -> torch.dtype:
        for param in self.model.parameters():
            if param.is_floating_point():
                return param.dtype
        return self.dtype if self.dtype.is_floating_point else torch.float32

    def _item_temperature(self, item: DecisionItem) -> float:
        if self.calibration is None:
            return 1.0
        bucket = bucket_for_item(item)
        temp = float(
            self.calibration.bucket_temperatures.get(bucket, self.calibration.global_temperature)
        )
        return max(temp, 1e-6)

    def _forward_option_logits_batch(
        self,
        items: list[DecisionItem],
    ) -> tuple[list[np.ndarray], list[int]]:
        if not items:
            return [], []

        rendered = [
            render_model_inputs(
                self.tokenizer,
                it,
                disable_thinking=self.config.disable_thinking,
                max_seq_len=self.config.max_seq_len,
            )
            for it in items
        ]
        label_ids_list = [resolve_label_token_ids(self.tokenizer, r["option_codes"]) for r in rendered]
        input_counts = [int(r["input_ids"].numel()) for r in rendered]

        max_len = max(input_counts)
        batch_size = len(items)
        pad_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)

        padded_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=self.device)
        padded_mask_2d = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)

        for b, r in enumerate(rendered):
            seq_len = int(r["input_ids"].numel())
            padded_ids[b, :seq_len] = r["input_ids"].to(self.device)
            padded_mask_2d[b, :seq_len] = r["attention_mask"].to(self.device)

        if self.config.effective_attention_mode in ("bidirectional", "set") and (
            self.config.architecture_family != "encoder"
        ):
            attn_arg: torch.Tensor = build_4d_attention_mask(
                padded_mask_2d,
                "bidirectional",
                self._mask_dtype(),
            )
        else:
            attn_arg = padded_mask_2d

        with torch.no_grad():
            outputs = self.model(input_ids=padded_ids, attention_mask=attn_arg)
            logits = outputs.logits

        option_logits_list: list[np.ndarray] = []
        for b, r in enumerate(rendered):
            ans_pos = int(r["answer_pos"])
            lids = label_ids_list[b]
            row_logits = logits[b, ans_pos, lids].detach().float().cpu().numpy()
            option_logits_list.append(np.asarray(row_logits, dtype=np.float64))

        return option_logits_list, input_counts

    def _build_prediction(
        self,
        item: DecisionItem,
        option_logits: np.ndarray,
        latency_ms: float,
        input_tokens: int,
    ) -> DecisionPrediction:
        temp = self._item_temperature(item)
        probs = _softmax_1d(option_logits / temp)
        best_idx = int(np.argmax(probs))
        choice = item.options[best_idx].key
        confidence = float(probs[best_idx])

        prob_map = {opt.key: float(p) for opt, p in zip(item.options, probs)}
        logit_map = {opt.key: float(l) for opt, l in zip(item.options, option_logits)}

        expected_score: float | None = None
        median_score: float | None = None
        if item.primitive == "score":
            expected_score, median_score = _compute_score_stats(item.options, probs)

        return DecisionPrediction(
            choice=choice,
            confidence=confidence,
            probabilities=prob_map,
            raw_logits=logit_map,
            temperature=temp,
            primitive=item.primitive,
            latency_ms=float(latency_ms),
            input_tokens=int(input_tokens),
            expected_score=expected_score,
            median_score=median_score,
        )

    def predict_one(
        self,
        item: DecisionItem,
        pride: bool | None = None,
    ) -> DecisionPrediction:
        """Predict a single DecisionItem with optional PriDe cyclic permutation debiasing."""
        return self.predict_batch([item], batch_size=1, pride=pride)[0]

    def predict_batch(
        self,
        items: list[DecisionItem],
        batch_size: int = 8,
        pride: bool | None = None,
    ) -> list[DecisionPrediction]:
        """Run batched single-pass inference across items, preserving exact answer_pos per row."""
        if not items:
            return []
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        use_pride = self.config.pride_debiasing if pride is None else pride
        predictions: list[DecisionPrediction] = []

        if not use_pride:
            for start in range(0, len(items), batch_size):
                chunk = items[start : start + batch_size]
                t0 = time.perf_counter()
                logits_list, token_counts = self._forward_option_logits_batch(chunk)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                per_item_ms = elapsed_ms / len(chunk)
                for it, raw_logits, n_tok in zip(chunk, logits_list, token_counts):
                    predictions.append(
                        self._build_prediction(it, raw_logits, per_item_ms, n_tok)
                    )
            return predictions

        for item in items:
            t0 = time.perf_counter()
            k = len(item.options)
            if k <= 1:
                logits_list, token_counts = self._forward_option_logits_batch([item])
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                predictions.append(
                    self._build_prediction(item, logits_list[0], elapsed_ms, token_counts[0])
                )
                continue

            num_perms = min(k, max(2, int(self.config.pride_permutations)))
            shifts = sorted({(p * k) // num_perms for p in range(num_perms)})
            shifted_items = [
                replace(item, options=[item.options[(j + s) % k] for j in range(k)])
                for s in shifts
            ]

            perm_logits: list[np.ndarray] = []
            perm_tokens: list[int] = []
            for p_start in range(0, len(shifted_items), batch_size):
                p_chunk = shifted_items[p_start : p_start + batch_size]
                l_chunk, t_chunk = self._forward_option_logits_batch(p_chunk)
                perm_logits.extend(l_chunk)
                perm_tokens.extend(t_chunk)

            unpermuted_log_probs = np.zeros(k, dtype=np.float64)
            for s, raw_l in zip(shifts, perm_logits):
                lp = _log_softmax_1d(raw_l)
                for orig_idx in range(k):
                    new_idx = (orig_idx - s) % k
                    unpermuted_log_probs[orig_idx] += lp[new_idx]
            debiased_logits = unpermuted_log_probs / len(shifts)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            predictions.append(
                self._build_prediction(item, debiased_logits, elapsed_ms, perm_tokens[0])
            )

        return predictions

    def predict_set(
        self,
        state: str,
        questions: list[DecisionItem],
    ) -> list[DecisionPrediction]:
        """Score multiple questions sharing one state in a single set-attention forward pass."""
        if not questions:
            return []

        can_set = (
            self.config.effective_attention_mode in ("bidirectional", "set")
            and self.config.architecture_family != "encoder"
        )
        if can_set:
            try:
                t0 = time.perf_counter()
                packed = render_set_inputs(
                    self.tokenizer,
                    state=state,
                    questions=questions,
                    disable_thinking=self.config.disable_thinking,
                    max_seq_len=max(self.config.max_seq_len, 1024),
                )
                input_ids = packed["input_ids"].unsqueeze(0).to(self.device)
                attn_2d = packed["attention_mask"].unsqueeze(0).to(self.device)
                mask_4d = build_4d_attention_mask(
                    attn_2d,
                    "set",
                    self._mask_dtype(),
                    segment_spans=[packed["segment_spans"]],
                )

                with torch.no_grad():
                    logits = self.model(input_ids=input_ids, attention_mask=mask_4d).logits

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                per_q_ms = elapsed_ms / len(questions)
                total_tokens = int(packed["input_ids"].numel())

                preds: list[DecisionPrediction] = []
                for q, ans_pos, codes in zip(
                    questions,
                    packed["answer_positions"],
                    packed["option_codes_per_q"],
                ):
                    label_ids = resolve_label_token_ids(self.tokenizer, codes)
                    raw_l = logits[0, int(ans_pos), label_ids].detach().float().cpu().numpy()
                    preds.append(
                        self._build_prediction(
                            q,
                            np.asarray(raw_l, dtype=np.float64),
                            per_q_ms,
                            total_tokens,
                        )
                    )
                return preds
            except Exception:
                pass

        batched_items = [
            replace(q, state=state) if not q.state else q for q in questions
        ]
        return self.predict_batch(batched_items)

    def system_one(
        self,
        state: str,
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Handle OpenJev/Jev /v1/systemone question specifications."""
        if not questions:
            return {
                "results": {},
                "answers": {},
                "latency_ms": 0.0,
                "attention_mode": self.config.effective_attention_mode,
            }

        t0 = time.perf_counter()
        q_ids: list[str] = []
        items: list[DecisionItem] = []

        for qid, spec in questions.items():
            q_ids.append(qid)
            raw_prim = str(spec.get("primitive") or spec.get("type") or spec.get("kind") or "choice").lower()
            primitive: PrimitiveKind
            if raw_prim in ("choice", "noul", "score"):
                primitive = raw_prim  # type: ignore[assignment]
            elif raw_prim in ("bool", "boolean", "yes_no", "binary"):
                primitive = "noul"
            elif raw_prim in ("rating", "scale", "likert"):
                primitive = "score"
            else:
                primitive = "choice"

            q_text = str(spec.get("question") or spec.get("prompt") or spec.get("text") or f"Evaluate: {qid}")
            options = _parse_system_one_options(spec, primitive)
            examples = spec.get("examples") if isinstance(spec.get("examples"), list) else []

            items.append(
                DecisionItem(
                    id=qid,
                    state=state,
                    question=q_text,
                    options=options,
                    primitive=primitive,
                    examples=examples,
                )
            )

        preds = self.predict_set(state, items)
        total_ms = (time.perf_counter() - t0) * 1000.0

        results: dict[str, dict[str, Any]] = {}
        answers: dict[str, str] = {}
        for qid, pred in zip(q_ids, preds):
            answers[qid] = pred.choice
            results[qid] = {
                "choice": pred.choice,
                "confidence": pred.confidence,
                "probabilities": pred.probabilities,
                "raw_logits": pred.raw_logits,
                "primitive": pred.primitive,
                "temperature": pred.temperature,
                "latency_ms": pred.latency_ms,
                "input_tokens": pred.input_tokens,
                "expected_score": pred.expected_score,
                "median_score": pred.median_score,
            }

        return {
            "results": results,
            "answers": answers,
            "latency_ms": float(total_ms),
            "attention_mode": self.config.effective_attention_mode,
        }


def _parse_system_one_options(
    spec: dict[str, Any],
    primitive: PrimitiveKind,
) -> list[DecisionOption]:
    raw_opts = spec.get("options")
    if isinstance(raw_opts, dict):
        return [DecisionOption(key=str(k), description=str(v)) for k, v in raw_opts.items()]

    if isinstance(raw_opts, list) and raw_opts:
        parsed: list[DecisionOption] = []
        for opt in raw_opts:
            if isinstance(opt, DecisionOption):
                parsed.append(opt)
            elif isinstance(opt, dict):
                key = str(opt.get("key") or opt.get("id") or opt.get("name") or opt.get("label") or opt.get("value") or "")
                desc = str(opt.get("description") or opt.get("text") or opt.get("label") or key)
                if not key:
                    key = desc
                parsed.append(DecisionOption(key=key, description=desc))
            else:
                text = str(opt)
                parsed.append(DecisionOption(key=text, description=text))
        return parsed

    if primitive == "noul":
        return [
            DecisionOption(key="yes", description="Yes"),
            DecisionOption(key="no", description="No"),
        ]

    if primitive == "score":
        min_s = int(spec.get("min_score", spec.get("min", 1)))
        max_s = int(spec.get("max_score", spec.get("max", spec.get("scale", 5))))
        if max_s < min_s:
            min_s, max_s = 1, 5
        return [DecisionOption(key=str(v), description=str(v)) for v in range(min_s, max_s + 1)]

    raise ValueError("choice question spec must define non-empty 'options'.")
