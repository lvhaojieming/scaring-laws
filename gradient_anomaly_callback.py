#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradient anomaly callback for Megatron-style pretraining.

Call `callback.inspect(...)` after backward and before optimizer.step()/gradient clipping.

It records:
- global pre-clip grad norm
- max absolute gradient
- non-finite gradient count
- zero-gradient count
- rolling EMA and spike ratio
- top-k parameter gradient norms
- optional W&B metrics
- JSON anomaly report
- optional emergency checkpoint callback
- optional skip-step / abort behavior
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist


@dataclass
class GradientAnomalyConfig:
    enabled: bool = True
    log_interval: int = 10
    topk: int = 20

    # Absolute thresholds
    max_global_grad_norm: float = 1.0e4
    max_abs_grad: float = 1.0e3

    # Relative spike detection
    ema_beta: float = 0.98
    spike_factor: float = 8.0
    min_steps_before_spike_check: int = 50

    # Trigger policies
    trigger_on_nonfinite: bool = True
    trigger_on_spike: bool = True
    trigger_on_large_abs_grad: bool = True
    trigger_on_large_global_norm: bool = True
    trigger_on_nonfinite_loss: bool = True

    # Actions when anomaly is detected
    save_report: bool = True
    save_grad_snapshot: bool = True
    call_emergency_checkpoint: bool = True
    skip_optimizer_step: bool = True
    abort_training: bool = False

    output_dir: str = "./gradient_anomalies"


class GradientAnomalyCallback:
    def __init__(self, config: GradientAnomalyConfig):
        self.cfg = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ema_grad_norm: Optional[float] = None
        self.num_updates = 0

    @staticmethod
    def _dist_ready() -> bool:
        return dist.is_available() and dist.is_initialized()

    @staticmethod
    def _rank() -> int:
        return dist.get_rank() if GradientAnomalyCallback._dist_ready() else 0

    @staticmethod
    def _world_size() -> int:
        return dist.get_world_size() if GradientAnomalyCallback._dist_ready() else 1

    @staticmethod
    def _unwrap_models(model: Any) -> List[torch.nn.Module]:
        if isinstance(model, (list, tuple)):
            models = list(model)
        else:
            models = [model]

        unwrapped: List[torch.nn.Module] = []
        for m in models:
            while hasattr(m, "module"):
                m = m.module
            unwrapped.append(m)
        return unwrapped

    @staticmethod
    def _iter_named_parameters(
        models: Sequence[torch.nn.Module],
    ) -> Iterable[Tuple[str, torch.nn.Parameter]]:
        for model_index, model in enumerate(models):
            prefix = f"model{model_index}."
            for name, param in model.named_parameters():
                yield prefix + name, param

    @staticmethod
    def _get_gradient_tensor(param: torch.nn.Parameter) -> Optional[torch.Tensor]:
        main_grad = getattr(param, "main_grad", None)
        if main_grad is not None:
            return main_grad
        return param.grad

    @staticmethod
    def _all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
        if GradientAnomalyCallback._dist_ready():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    @staticmethod
    def _all_reduce_max(value: torch.Tensor) -> torch.Tensor:
        if GradientAnomalyCallback._dist_ready():
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return value

    def _collect_stats(
        self,
        models: Sequence[torch.nn.Module],
    ) -> Dict[str, Any]:
        device: Optional[torch.device] = None
        for _, p in self._iter_named_parameters(models):
            if p.is_cuda:
                device = p.device
                break
        if device is None:
            device = torch.device("cpu")

        local_sq_sum = torch.zeros((), device=device, dtype=torch.float64)
        local_nonfinite = torch.zeros((), device=device, dtype=torch.int64)
        local_zero = torch.zeros((), device=device, dtype=torch.int64)
        local_numel = torch.zeros((), device=device, dtype=torch.int64)
        local_max_abs = torch.zeros((), device=device, dtype=torch.float64)

        local_top: List[Tuple[float, str, int, float, int]] = []

        for name, param in self._iter_named_parameters(models):
            grad = self._get_gradient_tensor(param)
            if grad is None:
                continue

            if grad.is_sparse:
                grad = grad.coalesce().values()

            grad_detached = grad.detach()
            finite_mask = torch.isfinite(grad_detached)
            nonfinite_count = int((~finite_mask).sum().item())

            safe_grad = torch.where(
                finite_mask,
                grad_detached,
                torch.zeros_like(grad_detached),
            )

            grad_float = safe_grad.float()
            sq_sum = torch.sum(grad_float * grad_float, dtype=torch.float64)
            grad_norm = float(torch.sqrt(sq_sum).item())
            max_abs = float(torch.max(torch.abs(grad_float)).item()) if grad_float.numel() else 0.0
            zero_count = int((grad_float == 0).sum().item())
            numel = int(grad_float.numel())

            local_sq_sum += sq_sum
            local_nonfinite += nonfinite_count
            local_zero += zero_count
            local_numel += numel
            local_max_abs = torch.maximum(
                local_max_abs,
                torch.tensor(max_abs, device=device, dtype=torch.float64),
            )

            local_top.append((grad_norm, name, nonfinite_count, max_abs, numel))

        global_sq_sum = self._all_reduce_sum(local_sq_sum)
        global_nonfinite = self._all_reduce_sum(local_nonfinite)
        global_zero = self._all_reduce_sum(local_zero)
        global_numel = self._all_reduce_sum(local_numel)
        global_max_abs = self._all_reduce_max(local_max_abs)

        global_grad_norm = float(torch.sqrt(global_sq_sum).item())
        nonfinite_count = int(global_nonfinite.item())
        zero_count = int(global_zero.item())
        numel = int(global_numel.item())
        max_abs_grad = float(global_max_abs.item())

        local_top.sort(key=lambda x: x[0], reverse=True)
        top_entries = [
            {
                "rank": self._rank(),
                "name": name,
                "grad_norm": norm,
                "nonfinite_count": nonfinite,
                "max_abs_grad": max_abs,
                "numel": n,
            }
            for norm, name, nonfinite, max_abs, n in local_top[: self.cfg.topk]
        ]

        if self._dist_ready():
            gathered: List[Any] = [None for _ in range(self._world_size())]
            dist.all_gather_object(gathered, top_entries)
            merged = [item for rank_items in gathered for item in (rank_items or [])]
            merged.sort(key=lambda x: x["grad_norm"], reverse=True)
            top_entries = merged[: self.cfg.topk]

        return {
            "global_grad_norm_pre_clip": global_grad_norm,
            "max_abs_grad": max_abs_grad,
            "nonfinite_count": nonfinite_count,
            "zero_grad_count": zero_count,
            "grad_numel": numel,
            "zero_grad_fraction": (zero_count / numel) if numel else 0.0,
            "top_parameters": top_entries,
        }

    def _update_ema(self, grad_norm: float) -> Tuple[float, float]:
        if self.ema_grad_norm is None:
            self.ema_grad_norm = grad_norm
        else:
            beta = self.cfg.ema_beta
            self.ema_grad_norm = beta * self.ema_grad_norm + (1.0 - beta) * grad_norm

        denominator = max(self.ema_grad_norm, 1.0e-12)
        spike_ratio = grad_norm / denominator
        return self.ema_grad_norm, spike_ratio

    def _detect_anomaly(self, stats: Dict[str, Any]) -> Tuple[bool, List[str]]:
        reasons: List[str] = []

        if self.cfg.trigger_on_nonfinite and stats["nonfinite_count"] > 0:
            reasons.append(f"nonfinite_gradients={stats['nonfinite_count']}")

        loss = stats.get("loss")
        if (
            self.cfg.trigger_on_nonfinite_loss
            and isinstance(loss, (int, float))
            and not math.isfinite(float(loss))
        ):
            reasons.append(f"nonfinite_loss={loss}")

        if (
            self.cfg.trigger_on_large_global_norm
            and stats["global_grad_norm_pre_clip"] > self.cfg.max_global_grad_norm
        ):
            reasons.append(
                "global_grad_norm_pre_clip="
                f"{stats['global_grad_norm_pre_clip']:.6g}>"
                f"{self.cfg.max_global_grad_norm:.6g}"
            )

        if (
            self.cfg.trigger_on_large_abs_grad
            and stats["max_abs_grad"] > self.cfg.max_abs_grad
        ):
            reasons.append(
                f"max_abs_grad={stats['max_abs_grad']:.6g}>"
                f"{self.cfg.max_abs_grad:.6g}"
            )

        if (
            self.cfg.trigger_on_spike
            and self.num_updates >= self.cfg.min_steps_before_spike_check
            and stats["spike_ratio"] > self.cfg.spike_factor
        ):
            reasons.append(
                f"grad_norm_spike_ratio={stats['spike_ratio']:.4f}>"
                f"{self.cfg.spike_factor:.4f}"
            )

        return bool(reasons), reasons

    @staticmethod
    def _safe_scalar(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().float().item())
            return None
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        return str(value)

    def _save_report(
        self,
        report: Dict[str, Any],
        models: Sequence[torch.nn.Module],
    ) -> Tuple[Optional[str], Optional[str]]:
        if self._rank() != 0:
            return None, None

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        iteration = int(report["iteration"])
        stem = f"grad_anomaly_iter_{iteration:07d}_{timestamp}"

        json_path: Optional[Path] = None
        snapshot_path: Optional[Path] = None

        if self.cfg.save_report:
            json_path = self.output_dir / f"{stem}.json"
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

        if self.cfg.save_grad_snapshot:
            snapshot: Dict[str, torch.Tensor] = {}
            top_names = {x["name"] for x in report["top_parameters"]}

            for name, param in self._iter_named_parameters(models):
                grad = self._get_gradient_tensor(param)
                if name not in top_names or grad is None:
                    continue
                if grad.is_sparse:
                    grad = grad.coalesce().to_dense()
                snapshot[name] = grad.detach().float().cpu()

            snapshot_path = self.output_dir / f"{stem}.pt"
            torch.save(
                {
                    "metadata": report,
                    "top_gradient_tensors": snapshot,
                },
                snapshot_path,
            )

        return (
            str(json_path) if json_path else None,
            str(snapshot_path) if snapshot_path else None,
        )

    def _log_wandb(self, metrics: Dict[str, Any], iteration: int) -> None:
        if self._rank() != 0:
            return

        try:
            import wandb

            if wandb.run is not None:
                wandb.log(metrics, step=iteration, commit=False)
        except Exception:
            pass

    def inspect(
        self,
        *,
        model: Any,
        iteration: int,
        loss: Any,
        learning_rate: Optional[float] = None,
        batch_metadata: Optional[Dict[str, Any]] = None,
        emergency_checkpoint_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Inspect gradients after backward and before clipping/optimizer.step.

        Returns:
            {
              "anomaly": bool,
              "reasons": list[str],
              "skip_optimizer_step": bool,
              "abort_training": bool,
              ...
            }
        """
        if not self.cfg.enabled:
            return {
                "anomaly": False,
                "reasons": [],
                "skip_optimizer_step": False,
                "abort_training": False,
            }

        models = self._unwrap_models(model)
        stats = self._collect_stats(models)

        ema, spike_ratio = self._update_ema(stats["global_grad_norm_pre_clip"])
        stats["ema_grad_norm"] = ema
        stats["spike_ratio"] = spike_ratio

        self.num_updates += 1

        report: Dict[str, Any] = {
            "iteration": int(iteration),
            "rank": self._rank(),
            "world_size": self._world_size(),
            "loss": self._safe_scalar(loss),
            "learning_rate": self._safe_scalar(learning_rate),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": asdict(self.cfg),
            **stats,
            "batch_metadata": batch_metadata or {},
        }

        anomaly, reasons = self._detect_anomaly(report)
        report["anomaly"] = anomaly
        report["reasons"] = reasons

        wandb_metrics = {
            "grad/global_norm_pre_clip": report["global_grad_norm_pre_clip"],
            "grad/ema_global_norm": report["ema_grad_norm"],
            "grad/spike_ratio": report["spike_ratio"],
            "grad/max_abs": report["max_abs_grad"],
            "grad/nonfinite_count": report["nonfinite_count"],
            "grad/zero_fraction": report["zero_grad_fraction"],
            "grad/anomaly": int(anomaly),
        }
        if iteration % self.cfg.log_interval == 0 or anomaly:
            self._log_wandb(wandb_metrics, iteration)

        report_path = None
        snapshot_path = None
        if anomaly:
            report_path, snapshot_path = self._save_report(report, models)
            report["report_path"] = report_path
            report["snapshot_path"] = snapshot_path

            if (
                self.cfg.call_emergency_checkpoint
                and emergency_checkpoint_fn is not None
            ):
                emergency_checkpoint_fn(report)

        result = {
            **report,
            "skip_optimizer_step": bool(
                anomaly and self.cfg.skip_optimizer_step
            ),
            "abort_training": bool(
                anomaly and self.cfg.abort_training
            ),
        }

        return result


def config_from_env(
    output_dir: Optional[str] = None,
) -> GradientAnomalyConfig:
    def env_bool(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def env_int(name: str, default: int) -> int:
        return int(os.environ.get(name, default))

    def env_float(name: str, default: float) -> float:
        return float(os.environ.get(name, default))

    return GradientAnomalyConfig(
        enabled=env_bool("GRAD_CB_ENABLED", True),
        log_interval=env_int("GRAD_CB_LOG_INTERVAL", 10),
        topk=env_int("GRAD_CB_TOPK", 20),
        max_global_grad_norm=env_float("GRAD_CB_MAX_GLOBAL_NORM", 1.0e4),
        max_abs_grad=env_float("GRAD_CB_MAX_ABS_GRAD", 1.0e3),
        ema_beta=env_float("GRAD_CB_EMA_BETA", 0.98),
        spike_factor=env_float("GRAD_CB_SPIKE_FACTOR", 8.0),
        min_steps_before_spike_check=env_int("GRAD_CB_MIN_STEPS", 50),
        trigger_on_nonfinite=env_bool("GRAD_CB_TRIGGER_NONFINITE", True),
        trigger_on_spike=env_bool("GRAD_CB_TRIGGER_SPIKE", True),
        trigger_on_large_abs_grad=env_bool("GRAD_CB_TRIGGER_ABS", True),
        trigger_on_large_global_norm=env_bool("GRAD_CB_TRIGGER_GLOBAL_NORM", True),
        trigger_on_nonfinite_loss=env_bool("GRAD_CB_TRIGGER_NONFINITE_LOSS", True),
        save_report=env_bool("GRAD_CB_SAVE_REPORT", True),
        save_grad_snapshot=env_bool("GRAD_CB_SAVE_SNAPSHOT", True),
        call_emergency_checkpoint=env_bool("GRAD_CB_EMERGENCY_CKPT", True),
        skip_optimizer_step=env_bool("GRAD_CB_SKIP_STEP", True),
        abort_training=env_bool("GRAD_CB_ABORT", False),
        output_dir=(
            output_dir
            or os.environ.get("GRAD_CB_OUTPUT_DIR", "./gradient_anomalies")
        ),
    )
