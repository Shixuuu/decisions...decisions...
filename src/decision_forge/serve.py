"""Micro-batched HTTP server implementing OpenJev/Jev `/v1/systemone`, `/v1/models`, and `/health`."""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import queue
import sys
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from decision_forge.constants import SCORE_DIGITS
from decision_forge.types import DecisionItem, DecisionOption, DecisionPrediction, PrimitiveKind

logger = logging.getLogger(__name__)

_STOP = object()


@dataclass
class _QueuedTask:
    item: DecisionItem
    future: Future[DecisionPrediction]


def _parse_question_to_item(
    req_id: str,
    qid: str,
    state: str,
    qspec: dict[str, Any],
) -> tuple[DecisionItem, PrimitiveKind]:
    """Translate one `/v1/systemone` question spec into a `DecisionItem`."""
    if not isinstance(qspec, dict):
        raise ValueError(f"Question {qid!r} must be an object")

    qtype_raw = str(qspec.get("type", "choice")).strip().lower()
    if qtype_raw not in ("choice", "noul", "score"):
        raise ValueError(f"Unsupported question type {qtype_raw!r} for {qid!r}")
    primitive: PrimitiveKind = qtype_raw  # type: ignore[assignment]

    instructions = str(qspec.get("instructions") or qspec.get("question") or "")
    raw_criteria = qspec.get("criteria")

    options: list[DecisionOption] = []
    examples: list[dict[str, Any]] = []

    if isinstance(raw_criteria, dict):
        for key, val in raw_criteria.items():
            key_str = str(key)
            if isinstance(val, dict):
                desc = str(val.get("what") if "what" in val else val.get("description", ""))
                for ex in val.get("examples") or []:
                    if isinstance(ex, str):
                        examples.append({"state": ex, "gold": key_str, "answer": key_str})
                    elif isinstance(ex, dict):
                        ex_state = str(ex.get("state") or ex.get("input") or "")
                        ex_gold = str(ex.get("gold") or ex.get("answer") or key_str)
                        if ex_state:
                            examples.append({"state": ex_state, "gold": ex_gold, "answer": ex_gold})
            elif val is None:
                desc = ""
            else:
                desc = str(val)
            options.append(DecisionOption(key=key_str, description=desc))
    elif isinstance(raw_criteria, list):
        for entry in raw_criteria:
            if isinstance(entry, dict) and "key" in entry:
                k = str(entry["key"])
                desc = str(entry.get("description") or entry.get("what") or "")
                options.append(DecisionOption(key=k, description=desc))
                for ex in entry.get("examples") or []:
                    if isinstance(ex, str):
                        examples.append({"state": ex, "gold": k, "answer": k})
            elif isinstance(entry, str):
                options.append(DecisionOption(key=entry, description=entry))
    elif raw_criteria is not None:
        raise ValueError(f"Question {qid!r} criteria must be an object or list")

    for ex in qspec.get("examples") or []:
        if isinstance(ex, dict):
            ex_state = str(ex.get("state") or ex.get("input") or "")
            ex_gold = str(ex.get("gold") or ex.get("answer") or "")
            if ex_state and ex_gold:
                examples.append({"state": ex_state, "gold": ex_gold, "answer": ex_gold})

    if primitive == "choice":
        if not options:
            raise ValueError(f"Choice question {qid!r} requires non-empty criteria")
    elif primitive == "noul":
        if not options:
            options = [DecisionOption("yes", "Yes"), DecisionOption("no", "No")]
        elif len(options) == 1:
            fallback_key = "null" if options[0].key != "null" else "no"
            options.append(DecisionOption(fallback_key, "None / does not apply"))
    elif primitive == "score":
        if not options:
            lo = int(qspec.get("min_score", 1))
            hi = int(qspec.get("max_score", qspec.get("scale", 9)))
            if lo < hi and (hi - lo) <= 20:
                options = [DecisionOption(str(n), f"Score {n}") for n in range(lo, hi + 1)]
            else:
                options = [DecisionOption(d, f"Score {d}") for d in SCORE_DIGITS]

    item = DecisionItem(
        id=f"{req_id}:{qid}",
        state=state,
        question=instructions,
        options=options,
        gold_idx=-1,
        primitive=primitive,
        source="systemone_api",
        examples=examples,
    )
    return item, primitive


def _compute_expected_score(probabilities: dict[str, float]) -> float | None:
    numeric: list[tuple[float, float]] = []
    for k, p in probabilities.items():
        try:
            numeric.append((float(k), float(p)))
        except (TypeError, ValueError):
            continue
    if not numeric:
        return None
    total_p = sum(p for _, p in numeric)
    if total_p <= 0:
        return None
    return sum(val * p for val, p in numeric) / total_p


class _DecisionRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler forwarding requests to the enclosing `DecisionHTTPServer`."""

    server: DecisionHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("DecisionHTTPServer: " + format, *args)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _read_json_body(self) -> dict[str, Any]:
        length_hdr = self.headers.get("Content-Length")
        if not length_hdr:
            raise ValueError("Missing Content-Length header")
        length = int(length_hdr)
        if length <= 0 or length > 16 * 1024 * 1024:
            raise ValueError(f"Invalid Content-Length: {length}")
        raw = self.rfile.read(length)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {"status": "ok", "model": self.server.model_name},
            )
            return
        if path == "/v1/models":
            self._send_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.model_name,
                            "object": "model",
                            "created": self.server.created_ts,
                            "owned_by": "decision-forge",
                        }
                    ],
                },
            )
            return
        self._send_json(
            HTTPStatus.NOT_FOUND,
            {"error": {"message": f"Unknown endpoint: {path}", "type": "not_found"}},
        )

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/v1/systemone":
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": {"message": f"Unknown endpoint: {path}", "type": "not_found"}},
            )
            return

        try:
            payload = self._read_json_body()
            state = payload.get("state")
            questions = payload.get("questions")
            if not isinstance(state, str):
                raise ValueError("Field 'state' must be a string")
            if not isinstance(questions, dict) or not questions:
                raise ValueError("Field 'questions' must be a non-empty object")

            req_id = f"sys1-{uuid.uuid4().hex[:16]}"
            model_requested = str(payload.get("model") or self.server.model_name)

            parsed_questions: list[tuple[str, DecisionItem, PrimitiveKind]] = []
            for qid, qspec in questions.items():
                item, primitive = _parse_question_to_item(req_id, str(qid), state, qspec)
                parsed_questions.append((str(qid), item, primitive))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"message": str(exc), "type": "invalid_request_error"}},
            )
            return

        try:
            futures = [self.server.enqueue_item(item) for _, item, _ in parsed_questions]
            preds = [fut.result(timeout=120.0) for fut in futures]
        except Exception as exc:
            logger.exception("Inference failure on /v1/systemone")
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": {"message": str(exc), "type": "server_error"}},
            )
            return

        answers: dict[str, Any] = {}
        examples_shown: dict[str, int] = {}
        total_input_tokens = 0
        for (qid, item, primitive), pred in zip(parsed_questions, preds, strict=True):
            probs = {str(k): round(float(v), 6) for k, v in (pred.probabilities or {}).items()}
            ans_obj: dict[str, Any] = {
                "choice": str(pred.choice),
                "confidence": round(float(pred.confidence), 6),
                "probabilities": probs,
            }
            exp_score = pred.expected_score
            if exp_score is None and primitive == "score":
                exp_score = _compute_expected_score(probs)
            if exp_score is not None:
                ans_obj["expected_score"] = round(float(exp_score), 6)
            if pred.median_score is not None:
                ans_obj["median_score"] = float(pred.median_score)

            answers[qid] = ans_obj
            examples_shown[qid] = len(item.examples)
            total_input_tokens += int(pred.input_tokens or 0)

        if total_input_tokens == 0 and state:
            total_input_tokens = max(1, len(state) // 4)

        self._send_json(
            HTTPStatus.OK,
            {
                "id": req_id,
                "object": "systemone",
                "model": model_requested,
                "answers": answers,
                "examples_shown": examples_shown,
                "usage": {
                    "input_tokens": total_input_tokens,
                    "output_tokens": 0,
                },
            },
        )


class DecisionHTTPServer(ThreadingHTTPServer):
    """Micro-batched `ThreadingHTTPServer` serving `/v1/systemone`, `/v1/models`, and `/health`."""

    allow_reuse_address = sys.platform != "win32"
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        engine: Any,
        model_name: str | None = None,
        max_batch_size: int = 8,
        batch_wait_ms: float = 10.0,
    ) -> None:
        self.engine = engine
        self.model_name = (
            model_name
            or getattr(engine, "model_name", None)
            or getattr(getattr(engine, "config", None), "model_name", None)
            or "decision-forge"
        )
        self.created_ts = int(time.time())
        self.max_batch_size = max(1, int(max_batch_size))
        self.batch_wait_ms = max(0.0, float(batch_wait_ms))

        self._queue: queue.Queue[_QueuedTask | object] = queue.Queue()
        self._closed = threading.Event()
        self._stats_lock = threading.Lock()
        self.batch_sizes: list[int] = []

        super().__init__(server_address, _DecisionRequestHandler)
        self._worker = threading.Thread(
            target=self._batch_worker_loop,
            name="decision-forge-batcher",
            daemon=True,
        )
        self._worker.start()

    def enqueue_item(self, item: DecisionItem) -> Future[DecisionPrediction]:
        if self._closed.is_set():
            raise RuntimeError("Server is shutting down")
        fut: Future[DecisionPrediction] = Future()
        self._queue.put(_QueuedTask(item=item, future=fut))
        return fut

    def _run_engine_batch(self, items: list[DecisionItem]) -> list[DecisionPrediction]:
        if hasattr(self.engine, "predict_batch"):
            sig = inspect.signature(self.engine.predict_batch)
            if "batch_size" in sig.parameters:
                return list(self.engine.predict_batch(items, batch_size=max(1, len(items))))
            return list(self.engine.predict_batch(items))
        if hasattr(self.engine, "system_one"):
            sig = inspect.signature(self.engine.system_one)
            out: list[DecisionPrediction] = []
            for it in items:
                if "questions" in sig.parameters:
                    res_wrap = self.engine.system_one(
                        state=it.state,
                        questions={
                            "q": {
                                "type": it.primitive,
                                "question": it.question,
                                "options": {o.key: o.description for o in it.options},
                                "examples": it.examples,
                            }
                        },
                    )
                    res = (res_wrap.get("results") or {}).get("q") or {}
                else:
                    res = self.engine.system_one(
                        state=it.state,
                        instructions=it.question,
                        criteria={o.key: o.description for o in it.options},
                        primitive=it.primitive,
                    )
                if isinstance(res, DecisionPrediction):
                    out.append(res)
                else:
                    probs = {str(k): float(v) for k, v in (res.get("probabilities") or {}).items()}
                    choice = str(res.get("choice", ""))
                    out.append(
                        DecisionPrediction(
                            choice=choice,
                            confidence=float(res.get("confidence", probs.get(choice, 0.0))),
                            probabilities=probs,
                            raw_logits={},
                            temperature=1.0,
                            primitive=it.primitive,
                            latency_ms=float(res.get("latency_ms", 0.0)),
                            input_tokens=int(res.get("input_tokens", 0)),
                            expected_score=res.get("expected_score"),
                            median_score=res.get("median_score"),
                        )
                    )
            return out
        raise AttributeError("Engine must implement `predict_batch(items)` or `system_one(...)`")

    def _batch_worker_loop(self) -> None:
        while True:
            first = self._queue.get()
            if first is _STOP:
                break

            assert isinstance(first, _QueuedTask)
            batch: list[_QueuedTask] = [first]
            deadline = time.monotonic() + (self.batch_wait_ms / 1000.0)

            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    nxt = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if nxt is _STOP:
                    self._queue.put(_STOP)
                    break
                assert isinstance(nxt, _QueuedTask)
                batch.append(nxt)

            with self._stats_lock:
                self.batch_sizes.append(len(batch))

            try:
                preds = self._run_engine_batch([t.item for t in batch])
                if len(preds) != len(batch):
                    raise RuntimeError(
                        f"Engine returned {len(preds)} predictions for batch of {len(batch)}"
                    )
                for task, pred in zip(batch, preds, strict=True):
                    if not task.future.done():
                        task.future.set_result(pred)
            except Exception as exc:
                for task in batch:
                    if not task.future.done():
                        task.future.set_exception(exc)

    def server_close(self) -> None:
        if not self._closed.is_set():
            self._closed.set()
            self._queue.put(_STOP)
            if self._worker.is_alive():
                self._worker.join(timeout=5.0)
            while True:
                try:
                    pending = self._queue.get_nowait()
                except queue.Empty:
                    break
                if isinstance(pending, _QueuedTask) and not pending.future.done():
                    pending.future.set_exception(RuntimeError("Server closed"))
        super().server_close()


def _load_decision_engine(model_dir: Path | str, device: str = "auto") -> Any:
    """Lazily locate and instantiate `DecisionEngine` from the inference module."""
    candidates = (
        "decision_forge.infer",
        "decision_forge.inference",
        "decision_forge.engine",
        "decision_forge.predict",
        "decision_forge.model",
        "decision_forge.model.engine",
        "decision_forge.modeling.engine",
        "decision_forge.runtime",
    )
    engine_cls = None
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        engine_cls = getattr(mod, "DecisionEngine", None)
        if engine_cls is not None:
            break

    if engine_cls is None:
        raise ImportError(
            "Could not locate `DecisionEngine` in decision_forge.{infer,inference,engine,model.engine}"
        )

    if hasattr(engine_cls, "from_pretrained"):
        sig = inspect.signature(engine_cls.from_pretrained)
        if "device" in sig.parameters:
            return engine_cls.from_pretrained(model_dir, device=device)
        return engine_cls.from_pretrained(model_dir)

    sig = inspect.signature(engine_cls)
    if "device" in sig.parameters:
        return engine_cls(model_dir, device=device)
    return engine_cls(model_dir)


def serve_model(
    model_dir: Path | str,
    host: str = "127.0.0.1",
    port: int = 8765,
    max_batch: int = 8,
    device: str = "auto",
) -> None:
    """Load a trained Decision Model directory and serve `/v1/systemone` until interrupted."""
    engine = _load_decision_engine(model_dir, device=device)
    server = DecisionHTTPServer(
        (host, port),
        engine=engine,
        max_batch_size=max_batch,
    )
    bound_host, bound_port = server.server_address[:2]
    print(
        f"Decision Forge serving '{server.model_name}' at http://{bound_host}:{bound_port} "
        f"(POST /v1/systemone, GET /v1/models, GET /health, max_batch={max_batch})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
