from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class OptimizationResult:
    params: object
    loss_history: list[float]
    best_loss: float
    best_params: object


class TensorFlowAdamOptimizer:
    name = "tensorflow-adam"

    def minimize(self, loss_fn: Callable, init_params, *, learning_rate: float, num_iter: int) -> OptimizationResult:
        import tensorflow as tf

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


def get_optimizer_backend(name: str) -> TensorFlowAdamOptimizer:
    normalized = name.lower()
    if normalized in {"adam", "tf", "tensorflow", "tensorflow-adam"}:
        return TensorFlowAdamOptimizer()
    raise ValueError(f"Unknown optimizer backend {name!r}; supported backend: 'tensorflow-adam'.")
