from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from smlr.backends.base import BackendUnavailableError


@dataclass
class OptimizationResult:
    params: object
    loss_history: list[float]
    best_loss: float
    best_params: object


class OptimizerBackend:
    name = "base"

    def minimize(self, loss_fn: Callable, init_params, *, learning_rate: float, num_iter: int) -> OptimizationResult:
        raise NotImplementedError


class TensorFlowOptimizerBackend(OptimizerBackend):
    name = "tensorflow"

    def minimize(self, loss_fn: Callable, init_params, *, learning_rate: float, num_iter: int) -> OptimizationResult:
        try:
            import tensorflow as tf
        except ImportError as exc:  # pragma: no cover
            raise BackendUnavailableError("TensorFlow backend requires `tensorflow`.") from exc

        params = tf.Variable(init_params, dtype=getattr(init_params, "dtype", None))
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        history: list[float] = []
        best_loss = float("inf")
        best_params = None
        for _ in range(int(num_iter)):
            with tf.GradientTape() as tape:
                loss = loss_fn(params)
            grads = tape.gradient(loss, [params])
            optimizer.apply_gradients(zip(grads, [params]))
            loss_value = float(loss.numpy())
            history.append(loss_value)
            if loss_value < best_loss:
                best_loss = loss_value
                best_params = params.numpy().copy()
        return OptimizationResult(params=params.numpy(), loss_history=history, best_loss=best_loss, best_params=best_params)


class TorchOptimizerBackend(OptimizerBackend):
    name = "torch"

    def minimize(self, loss_fn: Callable, init_params, *, learning_rate: float, num_iter: int) -> OptimizationResult:
        raise BackendUnavailableError("PyTorch optimization is not supported in the v0.1.0 release.")


class JaxOptimizerBackend(OptimizerBackend):
    name = "jax"

    def minimize(self, loss_fn: Callable, init_params, *, learning_rate: float, num_iter: int) -> OptimizationResult:
        raise BackendUnavailableError("JAX optimization is not supported in the v0.1.0 release.")


def get_optimizer_backend(name: str) -> OptimizerBackend:
    normalized = name.lower()
    if normalized in {"tf", "tensorflow"}:
        return TensorFlowOptimizerBackend()
    if normalized in {"torch", "pytorch", "jax"}:
        raise BackendUnavailableError(f"{name!r} is not supported in the v0.1.0 release; use 'tensorflow'.")
    raise ValueError(f"Unknown optimizer backend {name!r}; supported backend: 'tensorflow'.")
