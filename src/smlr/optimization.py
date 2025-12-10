"""Optimization backends for SMLR emulator training.

This module provides a unified interface for multiple optimization backends:
- scipy (default, no extra dependencies)
- tensorflow (optional, GPU-accelerated)
- jax/optax (optional, modern autodiff with JIT)

The optimization API is designed to be backend-agnostic, allowing users to switch
backends at runtime without changing their code.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize, least_squares

Array = np.ndarray


class OptimizerBackend(Enum):
    """Available optimization backends."""
    SCIPY = auto()
    TENSORFLOW = auto()
    JAX = auto()


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    params: Array
    cost: float
    n_iterations: int
    converged: bool
    history: List[float] = field(default_factory=list)
    message: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    """Configuration for optimizers.
    
    Parameters
    ----------
    backend : OptimizerBackend
        Which backend to use. Defaults to SCIPY.
    method : str
        Optimization method (backend-specific). For scipy: 'L-BFGS-B', 'TNC', etc.
        For tensorflow/jax: 'adam', 'sgd', 'rmsprop'.
    max_iterations : int
        Maximum number of iterations.
    learning_rate : float
        Learning rate for gradient-based methods (TF/JAX). Ignored by scipy.
    tolerance : float
        Convergence tolerance.
    patience : int
        Early stopping patience (number of iterations without improvement).
    min_delta : float
        Minimum relative improvement to reset patience counter.
    verbose : bool
        Whether to print progress.
    random_seed : int
        Random seed for reproducibility.
    """
    backend: OptimizerBackend = OptimizerBackend.SCIPY
    method: str = "L-BFGS-B"
    max_iterations: int = 1000
    learning_rate: float = 0.01
    tolerance: float = 1e-6
    patience: int = 100
    min_delta: float = 1e-4
    verbose: bool = False
    random_seed: int = 42
    # Additional backend-specific options
    extra_options: Dict[str, Any] = field(default_factory=dict)


class BaseOptimizer(abc.ABC):
    """Abstract base class for all optimizers."""
    
    def __init__(self, config: OptimizerConfig):
        self.config = config
        np.random.seed(config.random_seed)
    
    @abc.abstractmethod
    def minimize(
        self,
        cost_fn: Callable[[Array], float],
        x0: Array,
        grad_fn: Optional[Callable[[Array], Array]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> OptimizationResult:
        """Minimize a cost function.
        
        Parameters
        ----------
        cost_fn : Callable
            Function mapping parameter array to scalar cost.
        x0 : Array
            Initial parameter values.
        grad_fn : Callable, optional
            Gradient function. If None, numerical gradients will be used.
        bounds : list of tuples, optional
            Parameter bounds as [(low, high), ...].
            
        Returns
        -------
        OptimizationResult
            Container with optimized parameters and metadata.
        """
        pass


class ScipyOptimizer(BaseOptimizer):
    """Scipy-based optimizer (L-BFGS-B, TNC, SLSQP, etc.)."""
    
    def minimize(
        self,
        cost_fn: Callable[[Array], float],
        x0: Array,
        grad_fn: Optional[Callable[[Array], Array]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> OptimizationResult:
        history: List[float] = []
        
        def callback(xk: Array) -> None:
            if self.config.verbose:
                cost = cost_fn(xk)
                history.append(cost)
                if len(history) % 100 == 0:
                    print(f"Iter {len(history)}: cost = {cost:.6f}")
        
        result = minimize(
            cost_fn,
            x0,
            method=self.config.method,
            jac=grad_fn,
            bounds=bounds,
            options={
                "maxiter": self.config.max_iterations,
                "gtol": self.config.tolerance,
                **self.config.extra_options,
            },
            callback=callback,
        )
        
        return OptimizationResult(
            params=result.x,
            cost=float(result.fun),
            n_iterations=result.nit if hasattr(result, 'nit') else len(history),
            converged=result.success,
            history=history,
            message=result.message if hasattr(result, 'message') else "",
        )


class ScipyLeastSquaresOptimizer(BaseOptimizer):
    """Scipy least_squares optimizer (TRF, dogbox, lm)."""
    
    def __init__(self, config: OptimizerConfig):
        super().__init__(config)
        # Default to TRF for bounded problems
        if self.config.method not in ("trf", "dogbox", "lm"):
            self.config.method = "trf"
    
    def minimize(
        self,
        cost_fn: Callable[[Array], Array],  # Note: returns residual vector
        x0: Array,
        grad_fn: Optional[Callable[[Array], Array]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> OptimizationResult:
        # Convert bounds to scipy format
        if bounds is not None:
            lb = np.array([b[0] for b in bounds])
            ub = np.array([b[1] for b in bounds])
            bounds_scipy = (lb, ub)
        else:
            bounds_scipy = (-np.inf, np.inf)
        
        result = least_squares(
            cost_fn,
            x0,
            method=self.config.method,
            jac=grad_fn if grad_fn else "2-point",
            bounds=bounds_scipy,
            max_nfev=self.config.max_iterations * len(x0),
            xtol=self.config.tolerance,
            ftol=self.config.tolerance,
            gtol=self.config.tolerance,
            verbose=2 if self.config.verbose else 0,
            **self.config.extra_options,
        )
        
        return OptimizationResult(
            params=result.x,
            cost=float(0.5 * np.sum(result.fun**2)),  # Convert residuals to cost
            n_iterations=result.nfev,
            converged=result.success,
            message=result.message,
        )


def _check_tensorflow_available() -> bool:
    """Check if TensorFlow is available."""
    try:
        import tensorflow as tf
        return True
    except ImportError:
        return False


def _check_jax_available() -> bool:
    """Check if JAX and Optax are available."""
    try:
        import jax
        import optax
        return True
    except ImportError:
        return False


class TensorFlowOptimizer(BaseOptimizer):
    """TensorFlow/Keras optimizer with Adam, SGD, etc."""
    
    def __init__(self, config: OptimizerConfig):
        super().__init__(config)
        if not _check_tensorflow_available():
            raise ImportError(
                "TensorFlow is required for this optimizer. "
                "Install with: pip install smlr[tensorflow]"
            )
        import tensorflow as tf
        self.tf = tf
        tf.random.set_seed(config.random_seed)
    
    def minimize(
        self,
        cost_fn: Callable[[Array], float],
        x0: Array,
        grad_fn: Optional[Callable[[Array], Array]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> OptimizationResult:
        tf = self.tf
        
        # Create variable
        params = tf.Variable(x0, dtype=tf.float64)
        
        # Create optimizer
        method = self.config.method.lower()
        lr = self.config.learning_rate
        if method == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif method == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        elif method == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        elif method == "adamw":
            optimizer = tf.keras.optimizers.AdamW(learning_rate=lr)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        # Bounds handling via projection
        if bounds is not None:
            lb = tf.constant([b[0] for b in bounds], dtype=tf.float64)
            ub = tf.constant([b[1] for b in bounds], dtype=tf.float64)
            
            def project(p: tf.Variable) -> None:
                p.assign(tf.clip_by_value(p, lb, ub))
        else:
            project = lambda p: None
        
        history: List[float] = []
        best_cost = float('inf')
        best_params = x0.copy()
        patience_counter = 0
        
        for i in range(self.config.max_iterations):
            with tf.GradientTape() as tape:
                cost = cost_fn(params.numpy())
                # If cost_fn is TF-compatible, use it directly
                # Otherwise wrap for numerical gradients
                if tf.is_tensor(cost):
                    cost_tensor = cost
                else:
                    # Numerical gradient fallback
                    cost_tensor = tf.constant(cost, dtype=tf.float64)
                    if grad_fn is not None:
                        grads_np = grad_fn(params.numpy())
                        grads = [tf.constant(grads_np, dtype=tf.float64)]
                    else:
                        # Finite difference
                        grads_np = self._numerical_gradient(cost_fn, params.numpy())
                        grads = [tf.constant(grads_np, dtype=tf.float64)]
            
            if tf.is_tensor(cost):
                grads = tape.gradient(cost_tensor, [params])
            
            if grads[0] is not None:
                optimizer.apply_gradients(zip(grads, [params]))
                project(params)
            
            cost_val = float(cost_tensor.numpy() if tf.is_tensor(cost_tensor) else cost)
            history.append(cost_val)
            
            # Early stopping
            if cost_val < best_cost - self.config.min_delta * abs(best_cost):
                best_cost = cost_val
                best_params = params.numpy().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.patience:
                if self.config.verbose:
                    print(f"Early stopping at iteration {i}")
                break
            
            if self.config.verbose and i % 100 == 0:
                print(f"Iter {i}: cost = {cost_val:.6f}")
        
        return OptimizationResult(
            params=best_params,
            cost=best_cost,
            n_iterations=len(history),
            converged=patience_counter >= self.config.patience or len(history) >= self.config.max_iterations,
            history=history,
        )
    
    def _numerical_gradient(self, fn: Callable, x: Array, eps: float = 1e-6) -> Array:
        """Compute numerical gradient via central differences."""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (fn(x_plus) - fn(x_minus)) / (2 * eps)
        return grad


class JAXOptimizer(BaseOptimizer):
    """JAX/Optax optimizer with Adam, SGD, etc."""
    
    def __init__(self, config: OptimizerConfig):
        super().__init__(config)
        if not _check_jax_available():
            raise ImportError(
                "JAX and Optax are required for this optimizer. "
                "Install with: pip install smlr[jax]"
            )
        import jax
        import jax.numpy as jnp
        import optax
        self.jax = jax
        self.jnp = jnp
        self.optax = optax
        jax.config.update("jax_enable_x64", True)
    
    def minimize(
        self,
        cost_fn: Callable[[Array], float],
        x0: Array,
        grad_fn: Optional[Callable[[Array], Array]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> OptimizationResult:
        jax = self.jax
        jnp = self.jnp
        optax = self.optax
        
        # Create optimizer
        method = self.config.method.lower()
        lr = self.config.learning_rate
        if method == "adam":
            optimizer = optax.adam(learning_rate=lr)
        elif method == "adamw":
            optimizer = optax.adamw(learning_rate=lr)
        elif method == "sgd":
            optimizer = optax.sgd(learning_rate=lr)
        elif method == "rmsprop":
            optimizer = optax.rmsprop(learning_rate=lr)
        elif method == "adagrad":
            optimizer = optax.adagrad(learning_rate=lr)
        else:
            optimizer = optax.adam(learning_rate=lr)
        
        # Convert cost function to JAX
        @jax.jit
        def jax_cost(p: jnp.ndarray) -> jnp.ndarray:
            # Wrap numpy cost function
            return jnp.asarray(cost_fn(np.asarray(p)))
        
        # Use JAX grad if no grad_fn provided
        if grad_fn is None:
            jax_grad = jax.grad(jax_cost)
        else:
            @jax.jit
            def jax_grad(p: jnp.ndarray) -> jnp.ndarray:
                return jnp.asarray(grad_fn(np.asarray(p)))
        
        # Initialize
        params = jnp.asarray(x0)
        opt_state = optimizer.init(params)
        
        # Bounds handling
        if bounds is not None:
            lb = jnp.asarray([b[0] for b in bounds])
            ub = jnp.asarray([b[1] for b in bounds])
            clip = lambda p: jnp.clip(p, lb, ub)
        else:
            clip = lambda p: p
        
        history: List[float] = []
        best_cost = float('inf')
        best_params = np.asarray(x0).copy()
        patience_counter = 0
        
        @jax.jit
        def step(params, opt_state):
            grads = jax_grad(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return clip(params), opt_state
        
        for i in range(self.config.max_iterations):
            params, opt_state = step(params, opt_state)
            cost_val = float(jax_cost(params))
            history.append(cost_val)
            
            # Early stopping
            if cost_val < best_cost - self.config.min_delta * abs(best_cost):
                best_cost = cost_val
                best_params = np.asarray(params).copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.patience:
                if self.config.verbose:
                    print(f"Early stopping at iteration {i}")
                break
            
            if self.config.verbose and i % 100 == 0:
                print(f"Iter {i}: cost = {cost_val:.6f}")
        
        return OptimizationResult(
            params=best_params,
            cost=best_cost,
            n_iterations=len(history),
            converged=patience_counter >= self.config.patience or len(history) >= self.config.max_iterations,
            history=history,
        )


def create_optimizer(
    backend: Union[str, OptimizerBackend] = "scipy",
    method: str = "L-BFGS-B",
    **kwargs
) -> BaseOptimizer:
    """Factory function to create an optimizer.
    
    Parameters
    ----------
    backend : str or OptimizerBackend
        Backend to use: "scipy", "tensorflow", or "jax".
    method : str
        Optimization method (backend-specific).
    **kwargs
        Additional arguments passed to OptimizerConfig.
        
    Returns
    -------
    BaseOptimizer
        Configured optimizer instance.
        
    Examples
    --------
    >>> opt = create_optimizer("scipy", method="L-BFGS-B", max_iterations=1000)
    >>> opt = create_optimizer("jax", method="adam", learning_rate=0.001)
    >>> opt = create_optimizer("tensorflow", method="adam", learning_rate=0.01)
    """
    if isinstance(backend, str):
        backend = OptimizerBackend[backend.upper()]
    
    config = OptimizerConfig(backend=backend, method=method, **kwargs)
    
    if backend == OptimizerBackend.SCIPY:
        return ScipyOptimizer(config)
    elif backend == OptimizerBackend.TENSORFLOW:
        return TensorFlowOptimizer(config)
    elif backend == OptimizerBackend.JAX:
        return JAXOptimizer(config)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def get_available_backends() -> List[str]:
    """Return list of available optimization backends.
    
    Returns
    -------
    list of str
        Names of available backends.
    """
    available = ["scipy"]  # Always available
    if _check_tensorflow_available():
        available.append("tensorflow")
    if _check_jax_available():
        available.append("jax")
    return available
