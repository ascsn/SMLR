from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np

from .lorentz import LorentzianMixture

Array = np.ndarray


def plot_spectrum(energy: Array, strength: Array, *, label: str = "data") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(energy, strength, label=label)
    ax.set_xlabel("Energy")
    ax.set_ylabel("Strength")
    ax.legend()
    ax.set_xlim(float(np.min(energy)), float(np.max(energy)))
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    return fig


def plot_comparison(
    energy: Array,
    truth: Array,
    prediction: Array,
    *,
    title: str = "Strength comparison",
    labels: tuple[str, str] = ("reference", "predicted"),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(energy, truth, label=labels[0])
    ax.plot(energy, prediction, label=labels[1], linestyle="--")
    ax.set_xlabel("Energy")
    ax.set_ylabel("Strength")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(float(np.min(energy)), float(np.max(energy)))
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    return fig


def plot_mixture_components(mixture: LorentzianMixture, energy: Array) -> plt.Figure:
    energy = np.asarray(energy, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    for e, s, w in zip(mixture.energies, mixture.strengths, np.broadcast_to(mixture.widths, mixture.strengths.shape)):
        comp = s * (w / (2 * np.pi)) / ((energy - e) ** 2 + (w**2) / 4)
        ax.plot(energy, comp, alpha=0.7)
    ax.plot(energy, mixture.evaluate(energy), color="k", label="sum")
    ax.set_xlabel("Energy")
    ax.set_ylabel("Strength")
    ax.legend()
    ax.set_xlim(float(np.min(energy)), float(np.max(energy)))
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    return fig
