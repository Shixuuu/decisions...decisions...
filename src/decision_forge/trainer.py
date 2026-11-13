"""Model loading, LoRA target discovery, zero-shot / LoRA / full training, and saving."""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from transformers.modeling_outputs import CausalLMOutput

from decision_forge.attention import build_4d_attention_mask, probe_bidirectional_support
from decision_forge.calibration import (
    bucket_for_item,
    fit_temperature_scaling,
    measure_order_flip_rate,
)
from decision_forge.constants import OPTION_LETTERS, SCORE_DIGITS
from decision_forge.engine import DecisionEngine, resolve_label_token_ids
from decision_forge.prompts import render_model_inputs, shuffle_item_options
from decision_forge.types import (
    CalibrationConfig,
    CalibrationReport,
    DecisionConfig,
    DecisionItem,
    KernelStackConfig,
    ModelInfo,
    TrainConfig,
    save_json,
)

logger = logging.getLogger(__name__)

_ATTN_MODULE_NAMES = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "qkv_proj",
        "out_proj",
        "in_proj",
        "query",
        "key",
        "value",
        "query_proj",
        "key_proj",
        "value_proj",
        "query_key_value",
        "c_attn",
        "Wqkv",
        "Wo",
    }
)
_MLP_MODULE_NAMES = frozenset(
    {
        "gate_proj",
        "up_proj",
        "down_proj",
        "gate_up_proj",
        "w1",
        "w2",
        "w3",
        "Wi",
        "c_fc",
        "c_proj",
        "fc1",
        "fc2",
        "dense",
    }
)
_EXCLUDED_HEAD_NAMES = frozenset(
    {
        "lm_head",
        "classifier",
        "score",
        "pooler",
        "head",
        "embed_tokens",
        "output_projection",
    }
)


class EncoderDecisionWrapper(torch.nn.Module):
    """Wrap an encoder AutoModel with a vocabulary-aligned option scoring head."""

    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config

        hidden_size = int(getattr(backbone.config, "hidden_size", 0) or getattr(backbone.config, "d_model", 0))
        vocab_size = int(getattr(backbone.config, "vocab_size", 0))
        if hidden_size <= 0 or vocab_size <= 0:
            raise ValueError(
                f"Cannot infer hidden_size ({hidden_size}) and vocab_size ({vocab_size}) for encoder wrapper."
            )

        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        emb = backbone.get_input_embeddings() if hasattr(backbone, "get_input_embeddings") else None
        if emb is not None and hasattr(emb, "weight") and tuple(emb.weight.shape) == (vocab_size, hidden_size):
            with torch.no_grad():
                self.lm_head.weight.copy_(emb.weight.detach())
            self.lm_head.to(dtype=emb.weight.dtype, device=emb.weight.device)

    def get_input_embeddings(self) -> torch.nn.Module | None:
        return self.backbone.get_input_embeddings() if hasattr(self.backbone, "get_input_embeddings") else None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> CausalLMOutput:
        mask_to_pass = attention_mask
        try:
            out = self.backbone(
                input_ids=input_ids,
                attention_mask=mask_to_pass,
                output_hidden_states=output_hidden_states,
            )
        except (ValueError, RuntimeError, IndexError, TypeError):
            if attention_mask is not None and attention_mask.dim() == 4:
                mask_2d = (attention_mask[:, 0].amax(dim=-1) > -1e4).to(dtype=torch.long)
                out = self.backbone(
                    input_ids=input_ids,
                    attention_mask=mask_2d,
                    output_hidden_states=output_hidden_states,
                )
            else:
                raise

        hidden = out.last_hidden_state
        logits = self.lm_head(hidden)
        hidden_states = getattr(out, "hidden_states", None)
        if output_hidden_states and hidden_states is None:
            hidden_states = (hidden,)
        return CausalLMOutput(logits=logits, hidden_states=hidden_states)


def _kernel_torch_dtype(kernel: KernelStackConfig) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if kernel.dtype not in mapping:
        raise ValueError(f"Unsupported kernel dtype: {kernel.dtype!r}")
    return mapping[kernel.dtype]


def _load_pretrained_with_attn(
    model_cls: Any,
    source: str,
    torch_dtype: torch.dtype,
    preferred_attn: str | None,
    extra_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    base_kwargs: dict[str, Any] = dict(extra_kwargs or {})
    candidates: list[str | None]
    if preferred_attn == "sdpa":
        candidates = ["sdpa", "eager", None]
    elif preferred_attn == "eager":
        candidates = ["eager", None]
    else:
        candidates = [None]

    last_exc: Exception | None = None
    for attn_impl in candidates:
        for dtype_key in ("dtype", "torch_dtype"):
            call_kwargs = dict(base_kwargs)
            call_kwargs[dtype_key] = torch_dtype
            if attn_impl is not None:
                call_kwargs["attn_implementation"] = attn_impl
            try:
                model = model_cls.from_pretrained(source, **call_kwargs)
                return model, (attn_impl or "eager")
            except TypeError as exc:
                last_exc = exc
                continue
            except (ValueError, NotImplementedError) as exc:
                last_exc = exc
                break

    assert last_exc is not None
    raise last_exc


def _resolve_trust_remote_and_config(source: str, requires_remote: bool) -> tuple[Any, bool]:
    trust_remote = bool(requires_remote)
    try:
        hf_cfg = (
            AutoConfig.from_pretrained(source, trust_remote_code=True)
            if trust_remote
            else AutoConfig.from_pretrained(source)
        )
    except ValueError as exc:
        if not trust_remote and ("trust_remote_code" in str(exc) or "custom code" in str(exc).lower()):
            trust_remote = True
            hf_cfg = AutoConfig.from_pretrained(source, trust_remote_code=True)
        else:
            raise
    if getattr(hf_cfg, "auto_map", None):
        trust_remote = True
    return hf_cfg, trust_remote


def load_base_model_and_tokenizer(
    model_info: ModelInfo,
    kernel: KernelStackConfig,
    attention_mode: str = "bidirectional",
    for_training: bool = True,
) -> tuple[Any, Any, str]:
    """Load model and tokenizer for decoder, encoder, or custom_decision architectures."""
    source = model_info.local_path or model_info.repo_id
    torch_dtype = _kernel_torch_dtype(kernel)

    hf_cfg, trust_remote = _resolve_trust_remote_and_config(source, model_info.requires_trust_remote_code)
    if trust_remote and not model_info.requires_trust_remote_code:
        model_info.requires_trust_remote_code = True

    try:
        tokenizer = (
            AutoTokenizer.from_pretrained(source, trust_remote_code=True)
            if trust_remote
            else AutoTokenizer.from_pretrained(source)
        )
    except ValueError as exc:
        if not trust_remote and ("trust_remote_code" in str(exc) or "custom code" in str(exc).lower()):
            trust_remote = True
            model_info.requires_trust_remote_code = True
            tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        else:
            raise
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    preferred_attn: str | None = None
    if attention_mode in ("bidirectional", "set"):
        preferred_attn = kernel.attn_implementation if kernel.attn_implementation in ("sdpa", "eager") else "sdpa"

    extra_kwargs: dict[str, Any] = {}
    if trust_remote:
        extra_kwargs["trust_remote_code"] = True
    if kernel.use_4bit_qlora:
        try:
            from transformers import BitsAndBytesConfig

            extra_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            extra_kwargs["device_map"] = "auto"
        except ImportError as exc:
            raise RuntimeError("use_4bit_qlora=True requires bitsandbytes to be installed.") from exc

    arch = model_info.architecture_family
    if arch == "encoder":
        backbone, attn_used = _load_pretrained_with_attn(
            AutoModel,
            source,
            torch_dtype,
            preferred_attn=preferred_attn,
            extra_kwargs=extra_kwargs,
        )
        model: Any = EncoderDecisionWrapper(backbone)
    elif arch == "custom_decision":
        is_causal_cls = any(
            str(a).endswith("ForCausalLM") for a in (getattr(hf_cfg, "architectures", None) or [])
        )
        if is_causal_cls:
            model, attn_used = _load_pretrained_with_attn(
                AutoModelForCausalLM,
                source,
                torch_dtype,
                preferred_attn=preferred_attn,
                extra_kwargs=extra_kwargs,
            )
        else:
            backbone, attn_used = _load_pretrained_with_attn(
                AutoModel,
                source,
                torch_dtype,
                preferred_attn=preferred_attn,
                extra_kwargs=extra_kwargs,
            )
            model = EncoderDecisionWrapper(backbone)
    else:
        model, attn_used = _load_pretrained_with_attn(
            AutoModelForCausalLM,
            source,
            torch_dtype,
            preferred_attn=preferred_attn,
            extra_kwargs=extra_kwargs,
        )

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if kernel.use_4bit_qlora and for_training:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

    if for_training and kernel.use_gradient_checkpointing:
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if not kernel.use_4bit_qlora:
        model.to(torch.device(kernel.device_str))

    return model, tokenizer, attn_used


def discover_lora_targets(
    model: torch.nn.Module,
    scope: Literal["attn", "attn_mlp"] = "attn_mlp",
) -> list[str]:
    """Dynamically inspect nn.Linear (and Conv1D) submodules to find LoRA target names."""
    attn_targets: list[str] = []
    mlp_targets: list[str] = []
    fallback_targets: list[str] = []
    seen_attn: set[str] = set()
    seen_mlp: set[str] = set()
    seen_fallback: set[str] = set()

    def _collect(include_conv1d: bool) -> None:
        for name, mod in model.named_modules():
            if not name:
                continue
            is_linear = isinstance(mod, torch.nn.Linear)
            is_conv1d = include_conv1d and mod.__class__.__name__ == "Conv1D"
            if not (is_linear or is_conv1d):
                continue

            parts = name.split(".")
            leaf = parts[-1]
            if leaf in _EXCLUDED_HEAD_NAMES or any(p in _EXCLUDED_HEAD_NAMES for p in parts):
                continue

            if leaf not in seen_fallback:
                seen_fallback.add(leaf)
                fallback_targets.append(leaf)

            lower_path = name.lower()
            is_attn_path = "attn" in lower_path or "attention" in lower_path
            is_mlp_path = any(k in lower_path for k in ("mlp", "feed_forward", "ffn", "intermediate"))

            if leaf in _ATTN_MODULE_NAMES or (is_attn_path and leaf not in _MLP_MODULE_NAMES):
                if leaf not in seen_attn:
                    seen_attn.add(leaf)
                    attn_targets.append(leaf)
            elif leaf in _MLP_MODULE_NAMES or is_mlp_path:
                if leaf not in seen_mlp:
                    seen_mlp.add(leaf)
                    mlp_targets.append(leaf)

    _collect(include_conv1d=False)
    if not fallback_targets:
        _collect(include_conv1d=True)

    if scope == "attn":
        return attn_targets if attn_targets else fallback_targets

    combined: list[str] = []
    seen: set[str] = set()
    for name in attn_targets + mlp_targets:
        if name not in seen:
            seen.add(name)
            combined.append(name)
    return combined if combined else fallback_targets


def _save_merged_artifacts(
    model: Any,
    tokenizer: Any,
    merged_dir: Path,
) -> None:
    merged_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(model, EncoderDecisionWrapper):
        model.backbone.save_pretrained(merged_dir)
        torch.save(model.lm_head.state_dict(), merged_dir / "encoder_lm_head.pt")
    else:
        model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)


def _evaluate_and_calibrate(
    model: Any,
    tokenizer: Any,
    decision_cfg: DecisionConfig,
    val_items: list[DecisionItem],
    calib_cfg: CalibrationConfig,
    device: torch.device,
    model_dtype: torch.dtype,
) -> CalibrationReport | None:
    if not calib_cfg.enabled or not val_items:
        return None

    eval_items = [it for it in val_items if 0 <= it.gold_idx < len(it.options)]
    if not eval_items:
        return None

    engine = DecisionEngine(
        model=model,
        tokenizer=tokenizer,
        config=decision_cfg,
        calibration=None,
        device=str(device),
        dtype=model_dtype,
    )
    preds = engine.predict_batch(eval_items, batch_size=8, pride=False)

    logits_list = [
        np.array([pred.raw_logits[opt.key] for opt in item.options], dtype=np.float64)
        for item, pred in zip(eval_items, preds)
    ]
    gold_indices = [it.gold_idx for it in eval_items]
    buckets = [bucket_for_item(it) for it in eval_items]

    report = fit_temperature_scaling(
        logits_list=logits_list,
        gold_indices=gold_indices,
        buckets=buckets,
        per_bucket=calib_cfg.per_bucket,
        min_bucket_samples=calib_cfg.min_bucket_samples,
        num_bins=calib_cfg.num_bins,
    )

    if calib_cfg.order_flip_eval_samples > 0 and calib_cfg.order_flip_shuffles > 0:
        flip_subset = eval_items[: calib_cfg.order_flip_eval_samples]
        cal_engine = DecisionEngine(
            model=model,
            tokenizer=tokenizer,
            config=decision_cfg,
            calibration=report,
            device=str(device),
            dtype=model_dtype,
        )
        report.order_flip_rate = measure_order_flip_rate(
            cal_engine,
            flip_subset,
            num_shuffles=calib_cfg.order_flip_shuffles,
            seed=42,
        )

    return report


def _compute_masked_option_loss(
    logits: torch.Tensor,
    answer_positions: list[int],
    label_ids_per_item: list[list[int]],
    gold_indices: list[int],
) -> torch.Tensor:
    """Compute exact per-item CrossEntropyLoss restricted to each item's valid option token IDs."""
    losses: list[torch.Tensor] = []
    device = logits.device
    for b, (ans_pos, label_ids, gold_idx) in enumerate(
        zip(answer_positions, label_ids_per_item, gold_indices)
    ):
        if not (0 <= gold_idx < len(label_ids)):
            raise ValueError(
                f"gold_idx {gold_idx} out of range for {len(label_ids)} options at batch index {b}."
            )
        opt_logits = logits[b, ans_pos, label_ids].float().unsqueeze(0)
        target = torch.tensor([gold_idx], dtype=torch.long, device=device)
        losses.append(F.cross_entropy(opt_logits, target))
    return torch.stack(losses).mean()


def train_decision_model(
    model_info: ModelInfo,
    train_items: list[DecisionItem],
    val_items: list[DecisionItem],
    train_cfg: TrainConfig,
    calib_cfg: CalibrationConfig,
    kernel: KernelStackConfig,
    output_dir: Path,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train (LoRA/full) or prepare (zero-shot) a Decision Model, calibrate it, and save artifacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = out_dir / "merged"

    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)

    device = torch.device(kernel.device_str)
    if device.type == "cpu" and kernel.num_threads > 0:
        torch.set_num_threads(kernel.num_threads)

    model_dtype = _kernel_torch_dtype(kernel)
    is_training = train_cfg.method != "zero_shot"

    model, tokenizer, attn_impl_used = load_base_model_and_tokenizer(
        model_info=model_info,
        kernel=kernel,
        attention_mode=train_cfg.attention_mode,
        for_training=is_training,
    )

    effective_attention_mode: str = (
        "encoder" if model_info.architecture_family == "encoder" else str(train_cfg.attention_mode)
    )
    primitives = sorted({it.primitive for it in (train_items + val_items)} or {"choice"})
    base_model_str = model_info.local_path or model_info.repo_id

    if not is_training:
        probe_reason = "not_probed"
        if effective_attention_mode in ("bidirectional", "set"):
            supported, probe_reason = probe_bidirectional_support(model, device, model_dtype)
            if not supported:
                logger.warning(
                    "Bidirectional 4D mask unsupported (%s); falling back to causal.",
                    probe_reason,
                )
                effective_attention_mode = "causal"

        decision_cfg = DecisionConfig(
            model_name=out_dir.name,
            base_model=base_model_str,
            architecture_family=model_info.architecture_family,
            attention_mode=train_cfg.attention_mode,
            effective_attention_mode=effective_attention_mode,
            train_method="zero_shot",
            primitives=primitives,
            option_tokens=list(OPTION_LETTERS[:10]) + list(SCORE_DIGITS),
            disable_thinking=train_cfg.disable_thinking,
            pride_debiasing=train_cfg.pride_debiasing,
            pride_permutations=train_cfg.pride_permutations,
            max_seq_len=train_cfg.max_seq_len,
            created_at=datetime.now(timezone.utc).isoformat(),
            hardware_backend=kernel.backend,
            requires_trust_remote_code=bool(model_info.requires_trust_remote_code),
        )
        _save_merged_artifacts(model, tokenizer, merged_dir)
        save_json(merged_dir / "decision_config.json", decision_cfg)
        save_json(out_dir / "decision_config.json", decision_cfg)

        if progress_cb is not None and calib_cfg.enabled and val_items:
            progress_cb({"event": "phase", "name": "calibrate"})
        calib_report = _evaluate_and_calibrate(
            model=model,
            tokenizer=tokenizer,
            decision_cfg=decision_cfg,
            val_items=val_items,
            calib_cfg=calib_cfg,
            device=device,
            model_dtype=model_dtype,
        )
        if calib_report is not None:
            save_json(merged_dir / "calibration.json", calib_report)
            save_json(out_dir / "calibration.json", calib_report)

        metrics = {
            "method": "zero_shot",
            "attn_implementation": attn_impl_used,
            "effective_attention_mode": effective_attention_mode,
            "probe_reason": probe_reason,
            "steps": 0,
            "step_1_loss": None,
            "final_loss": None,
            "val_accuracy": calib_report.accuracy if calib_report is not None else None,
            "merged_dir": str(merged_dir),
        }
        save_json(out_dir / "train_metrics.json", metrics)
        if progress_cb is not None:
            progress_cb({"event": "complete", **metrics})
        return metrics

    if not train_items:
        raise ValueError("train_items must be non-empty when method is 'lora' or 'full'.")

    if train_cfg.method == "lora":
        target_modules = discover_lora_targets(model, scope=train_cfg.lora_target_scope)
        is_enc_wrapper = isinstance(model, EncoderDecisionWrapper)
        lora_kwargs: dict[str, Any] = {
            "r": train_cfg.lora_r,
            "lora_alpha": train_cfg.lora_alpha,
            "lora_dropout": train_cfg.lora_dropout,
            "target_modules": target_modules,
            "bias": "none",
        }
        if is_enc_wrapper:
            lora_kwargs["modules_to_save"] = ["lm_head"]
        else:
            lora_kwargs["task_type"] = TaskType.CAUSAL_LM
        model = get_peft_model(model, LoraConfig(**lora_kwargs))

    probe_reason = "not_probed"
    if effective_attention_mode in ("bidirectional", "set"):
        supported, probe_reason = probe_bidirectional_support(model, device, model_dtype)
        if not supported:
            logger.warning(
                "Bidirectional 4D mask unsupported (%s); falling back to causal.",
                probe_reason,
            )
            effective_attention_mode = "causal"

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found for training.")

    use_autocast = (
        device.type in ("cuda", "xpu") and model_dtype in (torch.float16, torch.bfloat16)
    ) or (device.type == "cpu" and model_dtype == torch.bfloat16)
    # Mixed precision keeps float32 master weights: GradScaler refuses float16 parameters, and
    # half-precision parameters round away updates at fine-tuning learning rates.
    half_trainables = [p for p in trainable_params if p.dtype != torch.float32] if use_autocast else []
    for param in half_trainables:
        param.data = param.data.float()

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    batch_size = max(1, int(train_cfg.batch_size))
    grad_accum = max(1, int(train_cfg.grad_accum_steps))
    micro_batches_per_epoch = max(1, math.ceil(len(train_items) / batch_size))
    opt_steps_per_epoch = max(1, math.ceil(micro_batches_per_epoch / grad_accum))
    total_opt_steps = max(1, opt_steps_per_epoch * max(1, int(train_cfg.epochs)))
    warmup_steps = int(total_opt_steps * train_cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_opt_steps,
    )

    use_scaler = device.type == "cuda" and model_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_scaler else None

    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    rng = random.Random(train_cfg.seed)
    global_step = 0
    step_1_loss: float | None = None
    loss_history: list[float] = []

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(train_cfg.epochs):
        epoch_items = list(train_items)
        rng.shuffle(epoch_items)
        micro_batches = [
            epoch_items[i : i + batch_size] for i in range(0, len(epoch_items), batch_size)
        ]

        window_loss_sum = 0.0
        window_count = 0

        for mb_idx, raw_batch in enumerate(micro_batches):
            if train_cfg.shuffle_options:
                batch_items = [shuffle_item_options(it, rng) for it in raw_batch]
            else:
                batch_items = raw_batch

            rendered = [
                render_model_inputs(
                    tokenizer,
                    it,
                    disable_thinking=train_cfg.disable_thinking,
                    max_seq_len=train_cfg.max_seq_len,
                )
                for it in batch_items
            ]
            label_ids_per_item = [
                resolve_label_token_ids(tokenizer, r["option_codes"]) for r in rendered
            ]
            gold_indices = [it.gold_idx for it in batch_items]
            answer_positions = [int(r["answer_pos"]) for r in rendered]

            max_len = max(int(r["input_ids"].numel()) for r in rendered)
            cur_bsz = len(rendered)
            padded_ids = torch.full((cur_bsz, max_len), pad_id, dtype=torch.long, device=device)
            mask_2d = torch.zeros((cur_bsz, max_len), dtype=torch.long, device=device)
            for b, r in enumerate(rendered):
                seq_len = int(r["input_ids"].numel())
                padded_ids[b, :seq_len] = r["input_ids"].to(device)
                mask_2d[b, :seq_len] = r["attention_mask"].to(device)

            if effective_attention_mode in ("bidirectional", "set") and (
                model_info.architecture_family != "encoder"
            ):
                if use_autocast:
                    mask_dtype = model_dtype
                else:
                    mask_dtype = next(
                        (p.dtype for p in model.parameters() if p.is_floating_point()),
                        torch.float32,
                    )
                attn_arg: torch.Tensor = build_4d_attention_mask(
                    mask_2d, "bidirectional", mask_dtype
                )
            else:
                attn_arg = mask_2d

            if use_autocast:
                with torch.autocast(device_type=device.type, dtype=model_dtype):
                    logits = model(input_ids=padded_ids, attention_mask=attn_arg).logits
                    loss = _compute_masked_option_loss(
                        logits, answer_positions, label_ids_per_item, gold_indices
                    )
            else:
                logits = model(input_ids=padded_ids, attention_mask=attn_arg).logits
                loss = _compute_masked_option_loss(
                    logits, answer_positions, label_ids_per_item, gold_indices
                )

            raw_loss_val = float(loss.detach().item())
            if step_1_loss is None:
                step_1_loss = raw_loss_val

            remaining_in_epoch = len(micro_batches) - (mb_idx - (mb_idx % grad_accum))
            effective_accum = min(grad_accum, remaining_in_epoch)
            scaled_loss = loss / float(effective_accum)

            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            window_loss_sum += raw_loss_val
            window_count += 1

            is_step_boundary = ((mb_idx + 1) % grad_accum == 0) or (
                (mb_idx + 1) == len(micro_batches)
            )
            if is_step_boundary:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, train_cfg.max_grad_norm)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                avg_step_loss = window_loss_sum / max(1, window_count)
                loss_history.append(avg_step_loss)
                window_loss_sum = 0.0
                window_count = 0

                if progress_cb is not None:
                    progress_cb(
                        {
                            "event": "step",
                            "step": global_step,
                            "total_steps": total_opt_steps,
                            "epoch": epoch + 1,
                            "loss": avg_step_loss,
                            "lr": float(scheduler.get_last_lr()[0]),
                        }
                    )

    if train_cfg.method == "lora":
        adapter_dir = out_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        merged_model = model.merge_and_unload()
    else:
        merged_model = model

    for param in half_trainables:
        param.data = param.data.to(model_dtype)
    merged_model.eval()
    _save_merged_artifacts(merged_model, tokenizer, merged_dir)

    decision_cfg = DecisionConfig(
        model_name=out_dir.name,
        base_model=base_model_str,
        architecture_family=model_info.architecture_family,
        attention_mode=train_cfg.attention_mode,
        effective_attention_mode=effective_attention_mode,
        train_method=train_cfg.method,
        primitives=primitives,
        option_tokens=list(OPTION_LETTERS[:10]) + list(SCORE_DIGITS),
        disable_thinking=train_cfg.disable_thinking,
        pride_debiasing=train_cfg.pride_debiasing,
        pride_permutations=train_cfg.pride_permutations,
        max_seq_len=train_cfg.max_seq_len,
        created_at=datetime.now(timezone.utc).isoformat(),
        hardware_backend=kernel.backend,
        requires_trust_remote_code=bool(model_info.requires_trust_remote_code),
    )
    save_json(merged_dir / "decision_config.json", decision_cfg)
    save_json(out_dir / "decision_config.json", decision_cfg)

    if progress_cb is not None and calib_cfg.enabled and val_items:
        progress_cb({"event": "phase", "name": "calibrate"})
    calib_report = _evaluate_and_calibrate(
        model=merged_model,
        tokenizer=tokenizer,
        decision_cfg=decision_cfg,
        val_items=val_items,
        calib_cfg=calib_cfg,
        device=device,
        model_dtype=model_dtype,
    )
    if calib_report is not None:
        save_json(merged_dir / "calibration.json", calib_report)
        save_json(out_dir / "calibration.json", calib_report)

    metrics = {
        "method": train_cfg.method,
        "attn_implementation": attn_impl_used,
        "effective_attention_mode": effective_attention_mode,
        "probe_reason": probe_reason,
        "steps": global_step,
        "step_1_loss": step_1_loss,
        "final_loss": loss_history[-1] if loss_history else None,
        "loss_history": loss_history,
        "val_accuracy": calib_report.accuracy if calib_report is not None else None,
        "nll_before": calib_report.nll_before if calib_report is not None else None,
        "nll_after": calib_report.nll_after if calib_report is not None else None,
        "ece_10_after": calib_report.ece_10_after if calib_report is not None else None,
        "order_flip_rate": calib_report.order_flip_rate if calib_report is not None else None,
        "merged_dir": str(merged_dir),
    }
    save_json(out_dir / "train_metrics.json", metrics)
    if progress_cb is not None:
        progress_cb({"event": "complete", **metrics})
    return metrics
