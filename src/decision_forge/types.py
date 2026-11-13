"""Shared dataclasses and JSON serialization helpers across all Decision Forge modules."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

ExecutionTarget = Literal["gpu", "igpu", "cpu", "colab"]
DeviceKind = Literal["discrete_gpu", "igpu", "cpu"]
BackendKind = Literal["cuda", "rocm", "xpu", "mps", "cpu"]
ArchitectureFamily = Literal["decoder", "encoder", "custom_decision"]
PrimitiveKind = Literal["choice", "noul", "score"]
AttentionMode = Literal["bidirectional", "set", "causal", "encoder"]
TrainMethod = Literal["lora", "full", "zero_shot"]


def sanitize_name(raw: str) -> str:
    """Create a cross-platform (Windows/macOS/Linux) safe directory or filename component."""
    cleaned = raw.replace("/", "--").replace("\\", "--")
    cleaned = re.sub(r'[<>:"|?*\x00-\x1f]', "_", cleaned)
    cleaned = re.sub(r"\s+", "_", cleaned).strip(" ._")
    return cleaned or "unnamed"


def make_run_id(model_repo: str) -> str:
    """Generate a Windows-safe timestamped run identifier."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_model = sanitize_name(model_repo.split("/")[-1])[:40]
    return f"{stamp}_{short_model}"


def save_json(path: Path | str, data: Any) -> Path:
    """Write dataclass or dict payload cleanly as UTF-8 JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(data) if hasattr(data, "__dataclass_fields__") else data
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def load_json(path: Path | str) -> Any:
    """Read UTF-8 JSON from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class HardwareDevice:
    vendor: str
    name: str
    kind: DeviceKind
    backend: BackendKind
    memory_gb: float
    unified_memory: bool = False
    driver: str = ""
    available_in_torch: bool = False
    missing_runtime: list[str] = field(default_factory=list)
    install_hint: str = ""


@dataclass
class HardwareProfile:
    os_name: str
    os_distro: str
    arch: str
    cpu_model: str
    physical_cores: int
    logical_cores: int
    cpu_flags: list[str]
    ram_total_gb: float
    ram_available_gb: float
    devices: list[HardwareDevice] = field(default_factory=list)
    has_discrete_gpu: bool = False
    has_igpu: bool = False
    recommended_backend: BackendKind = "cpu"
    active_torch_version: str = "not_installed"
    active_torch_backend: str = "none"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HardwareProfile:
        devices = [HardwareDevice(**d) for d in data.get("devices", [])]
        payload = {**data, "devices": devices}
        return cls(**payload)


@dataclass
class KernelStackConfig:
    target: ExecutionTarget
    backend: BackendKind
    device_str: str
    dtype: Literal["bfloat16", "float16", "float32"]
    attn_implementation: Literal["sdpa", "eager"]
    use_compile: bool = False
    compile_backend: str | None = None
    use_gradient_checkpointing: bool = False
    use_4bit_qlora: bool = False
    recommended_micro_batch_size: int = 2
    recommended_grad_accum: int = 8
    num_threads: int = 4
    notes: list[str] = field(default_factory=list)


@dataclass
class ModelInfo:
    repo_id: str
    revision: str = "main"
    local_path: str | None = None
    architecture_family: ArchitectureFamily = "decoder"
    model_type: str = ""
    param_count_b: float | None = None
    estimated_size_gb: float | None = None
    downloads: int = 0
    likes: int = 0
    license: str | None = None
    pipeline_tag: str | None = None
    has_chat_template: bool = True
    supports_thinking_toggle: bool = False
    requires_trust_remote_code: bool = False
    transformers_version: str = ""
    fit_by_target: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelInfo:
        return cls(**data)


@dataclass(frozen=True)
class DecisionOption:
    key: str
    description: str


@dataclass
class DecisionItem:
    id: str
    state: str
    question: str
    options: list[DecisionOption]
    gold_idx: int = -1
    primitive: PrimitiveKind = "choice"
    source: str = ""
    category: str = ""
    pair_id: str = ""
    difficulty: str = ""
    examples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def keys(self) -> list[str]:
        return [o.key for o in self.options]


@dataclass
class DatasetSpec:
    mode: Literal["hf_dataset", "local_file", "auto_general_mix", "zero_shot"] = "zero_shot"
    repo_id: str | None = None
    config_name: str | None = None
    split: str = "train"
    local_path: str | None = None
    schema_type: Literal["auto", "classification", "nli", "multiple_choice"] = "auto"
    state_col: str | None = None
    second_text_col: str | None = None
    question_col: str | None = None
    options_col: str | None = None
    label_col: str | None = None
    default_question: str = "Which option best applies to the input?"
    primitive: PrimitiveKind = "choice"
    max_train_samples: int = 2000
    max_val_samples: int = 400
    val_ratio: float = 0.15
    max_options_per_item: int = 10
    mix_datasets: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetSpec:
        return cls(**data)


@dataclass
class TrainConfig:
    method: TrainMethod = "lora"
    attention_mode: AttentionMode = "bidirectional"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_scope: Literal["attn", "attn_mlp"] = "attn_mlp"
    epochs: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    batch_size: int = 4
    grad_accum_steps: int = 4
    max_seq_len: int = 512
    max_grad_norm: float = 1.0
    shuffle_options: bool = True
    disable_thinking: bool = True
    pride_debiasing: bool = True
    pride_permutations: int = 3
    seed: int = 42

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        return cls(**data)


@dataclass
class CalibrationConfig:
    enabled: bool = True
    per_bucket: bool = True
    min_bucket_samples: int = 30
    num_bins: int = 10
    order_flip_eval_samples: int = 64
    order_flip_shuffles: int = 3

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationConfig:
        return cls(**data)


@dataclass
class DecideBenchConfig:
    enabled: bool = True
    version: Literal["v2", "v1", "both"] = "v2"
    protocols: list[Literal["few_shot", "zero_shot"]] = field(
        default_factory=lambda: ["few_shot", "zero_shot"]
    )
    limit: int | None = None
    via_server: bool = False
    concurrency: int = 4

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecideBenchConfig:
        return cls(**data)


@dataclass
class DecisionConfig:
    model_name: str
    base_model: str
    architecture_family: ArchitectureFamily
    attention_mode: AttentionMode
    effective_attention_mode: str
    train_method: str
    primitives: list[str]
    option_tokens: list[str]
    disable_thinking: bool = True
    pride_debiasing: bool = False
    pride_permutations: int = 3
    max_seq_len: int = 512
    created_at: str = ""
    hardware_backend: str = "cpu"
    requires_trust_remote_code: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionConfig:
        return cls(**data)


@dataclass
class CalibrationReport:
    global_temperature: float
    bucket_temperatures: dict[str, float]
    val_samples: int
    accuracy: float
    nll_before: float
    nll_after: float
    ece_10_before: float
    ece_10_after: float
    ece_15_before: float
    ece_15_after: float
    brier_before: float
    brier_after: float
    order_flip_rate: float | None = None
    reliability_bins: list[dict[str, Any]] = field(default_factory=list)
    risk_coverage: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationReport:
        return cls(**data)


@dataclass
class DecisionPrediction:
    choice: str
    confidence: float
    probabilities: dict[str, float]
    raw_logits: dict[str, float]
    temperature: float
    primitive: PrimitiveKind
    latency_ms: float
    input_tokens: int = 0
    expected_score: float | None = None
    median_score: float | None = None


@dataclass
class BenchSummary:
    version: str
    protocol: str
    total_items: int
    accuracy: float
    ci_low: float
    ci_high: float
    pair_accuracy: float
    ece_10: float | None
    brier: float | None
    latency_p50_ms: float
    latency_p95_ms: float
    estimated_cost_per_1m: float
    by_family: dict[str, float] = field(default_factory=dict)
    by_option_count: dict[str, float] = field(default_factory=dict)
    leaderboard_rows: list[dict[str, Any]] = field(default_factory=list)
    report_path: str | None = None


@dataclass
class WizardSessionConfig:
    run_id: str
    output_dir: str
    model: ModelInfo
    dataset: DatasetSpec = field(default_factory=DatasetSpec)
    train: TrainConfig = field(default_factory=TrainConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    decidebench: DecideBenchConfig = field(default_factory=DecideBenchConfig)
    export_gguf: bool = False
    start_server_after: bool = False
    server_port: int = 8765
    execution_target: ExecutionTarget | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WizardSessionConfig:
        return cls(
            run_id=data["run_id"],
            output_dir=data["output_dir"],
            model=ModelInfo.from_dict(data["model"]),
            dataset=DatasetSpec.from_dict(data.get("dataset", {})),
            train=TrainConfig.from_dict(data.get("train", {})),
            calibration=CalibrationConfig.from_dict(data.get("calibration", {})),
            decidebench=DecideBenchConfig.from_dict(data.get("decidebench", {})),
            export_gguf=bool(data.get("export_gguf", False)),
            start_server_after=bool(data.get("start_server_after", False)),
            server_port=int(data.get("server_port", 8765)),
            execution_target=data.get("execution_target"),
        )
