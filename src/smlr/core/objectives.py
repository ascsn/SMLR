from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import tensorflow as tf


@dataclass(frozen=True)
class ObjectiveTerm:
    name: str
    value: tf.Tensor
    weight: float = 1.0


def trapezoid_weights(energy, *, dtype=tf.float32):
    energy = tf.convert_to_tensor(energy, dtype=dtype)
    tf.debugging.assert_greater_equal(tf.size(energy), 2, message="energy grid must contain at least two points.")
    d = energy[1:] - energy[:-1]
    first = d[0] / 2.0
    last = d[-1] / 2.0
    middle = (energy[2:] - energy[:-2]) / 2.0 if energy.shape[0] is None or energy.shape[0] > 2 else tf.zeros([0], dtype)
    weights = tf.concat([tf.reshape(first, [1]), middle, tf.reshape(last, [1])], axis=0)
    return tf.maximum(weights, tf.cast(0.0, dtype))


def relative_integrated_strength_loss(predicted, true, energy, *, eps: float = 1e-12):
    predicted = tf.convert_to_tensor(predicted)
    dtype = predicted.dtype
    true = tf.cast(true, dtype)
    energy = tf.cast(energy, dtype)
    weights = trapezoid_weights(energy, dtype=dtype)
    numer = tf.reduce_sum(tf.square(predicted - true) * weights[None, :], axis=1)
    denom = tf.reduce_sum(tf.square(true) * weights[None, :], axis=1)
    return tf.reduce_mean(numer / tf.maximum(denom, tf.cast(eps, dtype)))


def scaled_mse(predicted, true, *, scale=None, log: bool = False, eps: float = 1e-12):
    predicted = tf.convert_to_tensor(predicted)
    dtype = predicted.dtype
    true = tf.cast(true, dtype)
    if log:
        predicted = tf.math.log(tf.maximum(predicted, tf.cast(eps, dtype)))
        true = tf.math.log(tf.maximum(true, tf.cast(eps, dtype)))
    if scale is None:
        scale_t = tf.cast(1.0, dtype)
    else:
        scale_t = tf.cast(scale, dtype)
    return tf.reduce_mean(tf.square((predicted - true) / tf.maximum(scale_t, tf.cast(eps, dtype))))


def combine_terms(terms: list[ObjectiveTerm]) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    if not terms:
        raise ValueError("At least one objective term is required.")
    dtype = terms[0].value.dtype
    total = tf.cast(0.0, dtype)
    components: dict[str, tf.Tensor] = {}
    for term in terms:
        weighted = tf.cast(term.weight, dtype) * tf.cast(term.value, dtype)
        components[term.name] = weighted
        total = total + weighted
    return total, components


def strength_plus_observable_objective(
    *,
    predicted_strength,
    true_strength,
    energy,
    predicted_observables: Mapping[str, tf.Tensor] | None = None,
    true_observables: Mapping[str, tf.Tensor] | None = None,
    observable_weights: Mapping[str, float] | None = None,
    observable_loss: Callable[[str, tf.Tensor, tf.Tensor], tf.Tensor] | None = None,
    strength_weight: float = 1.0,
):
    terms = [
        ObjectiveTerm(
            name="strength",
            value=relative_integrated_strength_loss(predicted_strength, true_strength, energy),
            weight=strength_weight,
        )
    ]

    predicted_observables = predicted_observables or {}
    true_observables = true_observables or {}
    observable_weights = observable_weights or {}
    observable_loss = observable_loss or (lambda _name, pred, true: scaled_mse(pred, true))

    for name, pred in predicted_observables.items():
        if name not in true_observables:
            raise KeyError(f"Missing true observable values for {name!r}.")
        terms.append(
            ObjectiveTerm(
                name=name,
                value=observable_loss(name, pred, true_observables[name]),
                weight=observable_weights.get(name, 1.0),
            )
        )
    return combine_terms(terms)
