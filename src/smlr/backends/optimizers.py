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
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise BackendUnavailableError("PyTorch backend requires `torch`.") from exc

        params = torch.as_tensor(init_params, dtype=torch.float64).clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([params], lr=learning_rate)
        history: list[float] = []
        best_loss = float("inf")
        best_params = None
        for _ in range(int(num_iter)):
            optimizer.zero_grad()
            loss = loss_fn(params)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())
            history.append(loss_value)
            if loss_value < best_loss:
                best_loss = loss_value
                best_params = params.detach().cpu().numpy().copy()
        return OptimizationResult(
            params=params.detach().cpu().numpy(),
            loss_history=history,
            best_loss=best_loss,
            best_params=best_params,
        )


class JaxOptimizerBackend(OptimizerBackend):
    name = "jax"

    def minimize(self, loss_fn: Callable, init_params, *, learning_rate: float, num_iter: int) -> OptimizationResult:
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as exc:  # pragma: no cover
            raise BackendUnavailableError("JAX backend requires `jax`.") from exc

        params = jnp.asarray(init_params)
        m = jnp.zeros_like(params)
        v = jnp.zeros_like(params)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        grad_fn = jax.value_and_grad(loss_fn)
        history: list[float] = []
        best_loss = float("inf")
        best_params = None
        for step in range(1, int(num_iter) + 1):
            loss, grads = grad_fn(params)
            m = beta1 * m + (1.0 - beta1) * grads
            v = beta2 * v + (1.0 - beta2) * jnp.square(grads)
            m_hat = m / (1.0 - beta1**step)
            v_hat = v / (1.0 - beta2**step)
            params = params - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
            loss_value = float(loss)
            history.append(loss_value)
            if loss_value < best_loss:
                best_loss = loss_value
                best_params = jax.device_get(params).copy()
        return OptimizationResult(params=jax.device_get(params), loss_history=history, best_loss=best_loss, best_params=best_params)


def get_optimizer_backend(name: str) -> OptimizerBackend:
    normalized = name.lower()
    if normalized in {"tf", "tensorflow"}:
        return TensorFlowOptimizerBackend()
    if normalized in {"torch", "pytorch"}:
        return TorchOptimizerBackend()
    if normalized == "jax":
        return JaxOptimizerBackend()
    raise ValueError(f"Unknown optimizer backend {name!r}.")
