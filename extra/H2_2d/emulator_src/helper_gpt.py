from __future__ import annotations

import os
import re
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from scipy.optimize import least_squares, nnls

logging.getLogger('tensorflow').setLevel(logging.ERROR)

hqc = 197.33
ALPHAD_FAC = 8.0 * np.pi * (7.29735e-3) * hqc / 9.0


# -----------------------------------------------------------------------------
# Dataset utilities
# -----------------------------------------------------------------------------
@dataclass
class GenericDataset:
    param_names: List[str]
    param_values: np.ndarray           # (B, p)
    sample_ids: List[Tuple[str, ...]]  # stringified parameter tuples
    strengths: List[np.ndarray]        # each (G, 2)
    alphaD_values: np.ndarray          # (B,)
    alphaD_raw: List[np.ndarray]
    central_point: np.ndarray          # (p,)


@dataclass(frozen=True)
class AnsatzConfig:
    n: int
    n_params: int
    ansatz: str = "linear"            # linear | quadratic | linear_exp
    width_model: str = "affine"       # constant | affine
    use_vector_terms: bool = True


@dataclass
class PackedLayout:
    eta_size: int
    v0_slice: slice
    v_linear_slice: slice
    d_diag_slice: slice
    basis_slice: slice
    width_bias_slice: slice
    width_linear_slice: slice
    feature_param_slice: slice
    n_upper: int
    n_basis: int
    total_size: int


def _sorted_named_groups(match: re.Match) -> List[Tuple[str, str]]:
    groupdict = match.groupdict()
    if groupdict:
        return sorted(groupdict.items(), key=lambda kv: kv[0])
    return [(f"p{i+1}", match.group(i + 1)) for i in range(len(match.groups()))]



def _parse_filter_ranges(filter_ranges: Optional[str | Dict[str, Sequence[float]]]) -> Dict[str, Tuple[float, float]]:
    if filter_ranges is None:
        return {}
    if isinstance(filter_ranges, str):
        filter_ranges = json.loads(filter_ranges)
    parsed = {}
    for k, v in filter_ranges.items():
        if len(v) != 2:
            raise ValueError(f"Filter for {k!r} must have two entries [min, max].")
        parsed[k] = (float(v[0]), float(v[1]))
    return parsed



def _extract_alphaD_value(arr: np.ndarray) -> float:
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        return float(arr[-1])
    return float(arr.reshape(-1)[-1])


def _resolve_existing_dir(path: Optional[str], *, label: str, required: bool) -> Optional[str]:
    if path is None:
        return None

    candidates = []
    if os.path.isabs(path):
        candidates.append(path)
    else:
        module_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.extend([
            os.path.abspath(path),
            os.path.abspath(os.path.join(module_dir, path)),
            os.path.abspath(os.path.join(os.path.dirname(module_dir), path)),
        ])

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    if required:
        tried = "\n  ".join(candidates)
        raise FileNotFoundError(f"{label} directory {path!r} was not found. Tried:\n  {tried}")
    return path



def load_dataset(
    strength_dir: str,
    alphaD_dir: Optional[str] = None,
    strength_regex: str = r"strength_(?P<p2>[0-9.]+)_(?P<p1>[0-9.]+)\.out",
    alphaD_regex: Optional[str] = None,
    filter_ranges: Optional[str | Dict[str, Sequence[float]]] = None,
    central_point: Optional[Sequence[float]] = None,
) -> GenericDataset:
    """
    Generic loader for datasets whose parameter values are encoded in filenames.

    Notes
    -----
    - Uses named regex groups when available, e.g. (?P<alpha>[0-9.]+).
    - If unnamed groups are used, parameters are named p1, p2, ... in group order.
    - The default regex preserves the original project convention:
      strength_<beta>_<alpha>.out  -> names become p1=alpha, p2=beta.
    """
    strength_dir = _resolve_existing_dir(strength_dir, label="strength", required=True)
    alphaD_dir = _resolve_existing_dir(alphaD_dir, label="alphaD", required=False)

    strength_pat = re.compile(strength_regex)
    alphaD_pat = re.compile(alphaD_regex or strength_regex.replace("strength_", "alphaD_"))
    filter_ranges = _parse_filter_ranges(filter_ranges)

    alphaD_map: Dict[Tuple[str, ...], Tuple[np.ndarray, float]] = {}
    param_names: Optional[List[str]] = None

    if alphaD_dir is not None and os.path.isdir(alphaD_dir):
        for fname in sorted(os.listdir(alphaD_dir)):
            match = alphaD_pat.match(fname)
            if not match:
                continue
            pairs = _sorted_named_groups(match)
            if param_names is None:
                param_names = [name for name, _ in pairs]
            key = tuple(value for _, value in pairs)
            raw = np.loadtxt(os.path.join(alphaD_dir, fname))
            alphaD_map[key] = (np.asarray(raw), _extract_alphaD_value(raw))

    strengths: List[np.ndarray] = []
    alphaD_raw: List[np.ndarray] = []
    alphaD_values: List[float] = []
    sample_ids: List[Tuple[str, ...]] = []
    param_rows: List[List[float]] = []

    for fname in sorted(os.listdir(strength_dir)):
        match = strength_pat.match(fname)
        if not match:
            continue
        pairs = _sorted_named_groups(match)
        if param_names is None:
            param_names = [name for name, _ in pairs]

        row = {name: float(value) for name, value in pairs}
        if any((name in filter_ranges) and not (filter_ranges[name][0] <= row[name] <= filter_ranges[name][1])
               for name in row):
            continue

        key = tuple(str(value) for _, value in pairs)
        strength = np.asarray(np.loadtxt(os.path.join(strength_dir, fname)), dtype=np.float32)
        if strength.ndim != 2 or strength.shape[1] < 2:
            raise ValueError(f"Strength file {fname} must have at least two columns [omega, strength].")

        strengths.append(strength[:, :2])
        sample_ids.append(key)
        param_rows.append([row[name] for name in param_names])

        if key in alphaD_map:
            raw_alphaD, alphaD_val = alphaD_map[key]
        else:
            omega = strength[:, 0]
            sigma = strength[:, 1]
            alphaD_val = float(ALPHAD_FAC * np.trapz(sigma / np.maximum(omega, 1e-6), omega))
            raw_alphaD = np.array([np.nan, np.nan, alphaD_val], dtype=np.float32)
        alphaD_raw.append(np.asarray(raw_alphaD))
        alphaD_values.append(alphaD_val)

    if not strengths:
        raise RuntimeError(f"No strength files matched regex in {strength_dir!r}.")

    param_values = np.asarray(param_rows, dtype=np.float32)
    if central_point is None:
        mins = np.min(param_values, axis=0)
        maxs = np.max(param_values, axis=0)
        center = 0.5 * (mins + maxs)
        idx = int(np.argmin(np.sum((param_values - center[None, :]) ** 2, axis=1)))
        central_point_arr = param_values[idx].copy()
    else:
        central_point_arr = np.asarray(central_point, dtype=np.float32)

    return GenericDataset(
        param_names=list(param_names or []),
        param_values=param_values,
        sample_ids=sample_ids,
        strengths=strengths,
        alphaD_values=np.asarray(alphaD_values, dtype=np.float32),
        alphaD_raw=alphaD_raw,
        central_point=central_point_arr,
    )


# -----------------------------------------------------------------------------
# Lorentzian helpers and central fit
# -----------------------------------------------------------------------------
def _softplus(x):
    x = np.asarray(x, dtype=np.float32)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)



def _inv_softplus(y):
    y = np.maximum(np.asarray(y, dtype=np.float32), 1e-12)
    return np.log(np.expm1(y))



def _unpack_poles_and_strengths_tf(z, n, wmin, min_spacing):
    z = tf.convert_to_tensor(z, tf.float32)
    zE, zB = z[:n], z[n:]

    e0 = wmin + tf.nn.softplus(zE[0])
    gaps = tf.nn.softplus(zE[1:]) + min_spacing
    E = tf.concat([e0[None], e0 + tf.cumsum(gaps)], axis=0)
    B = tf.nn.softplus(zB) ** 2
    return E, B



def _pack_poles_and_strengths_np(E0, B0, wmin, min_spacing):
    zE = np.empty_like(E0, dtype=np.float32)
    zE[0] = _inv_softplus(E0[0] - wmin)
    gaps = np.diff(E0)
    zE[1:] = _inv_softplus(np.maximum(gaps - min_spacing, 1e-12))
    zB = _inv_softplus(np.sqrt(np.maximum(B0, 1e-12)))
    return np.concatenate([zE, zB])


@tf.function
def give_me_Lorentzian(energy, poles, strength, width):
    energy = tf.convert_to_tensor(energy, dtype=tf.float32)
    poles = tf.convert_to_tensor(poles, dtype=tf.float32)
    strength = tf.convert_to_tensor(strength, dtype=tf.float32)
    width = tf.convert_to_tensor(width, dtype=tf.float32)

    energy_expanded = tf.expand_dims(energy, axis=-1)
    numerator = strength * (width / (2.0 * np.pi))
    denominator = (energy_expanded - poles) ** 2 + (width ** 2 / 4.0)
    return tf.reduce_sum(numerator / denominator, axis=-1)


@tf.function
def give_me_Lorentzian_batched(omega, poles_batch, B_batch, width_batch):
    omega = tf.convert_to_tensor(omega, dtype=tf.float32)
    poles_batch = tf.convert_to_tensor(poles_batch, dtype=tf.float32)
    B_batch = tf.convert_to_tensor(B_batch, dtype=tf.float32)
    width_batch = tf.reshape(tf.convert_to_tensor(width_batch, dtype=tf.float32), (-1, 1, 1))

    omega_exp = tf.expand_dims(omega, axis=1)
    omega_exp = tf.tile(omega_exp, [tf.shape(poles_batch)[0], 1, 1])
    poles_exp = tf.expand_dims(poles_batch, axis=-1)
    B_exp = tf.expand_dims(B_batch, axis=-1)

    numerator = B_exp * width_batch / np.pi
    denominator = tf.square(omega_exp - poles_exp) + tf.square(width_batch)
    return tf.reduce_sum(numerator / denominator, axis=1)



def fit_strength_with_tf_lorentzian(omega, y, n, eta, grid_M=None, min_spacing=0.2, l2=0.0):
    omega_np = np.asarray(omega, np.float32)
    y_np = np.asarray(y, np.float32) #
    wmin, wmax = float(omega_np.min()), float(omega_np.max())
    if grid_M is None:
        grid_M = len(omega_np)

    E_grid = np.linspace(wmin + 1e-6, wmax - 1e-6, grid_M, dtype=np.float32) #
    A = 1.0 / ((omega_np[:, None] - E_grid[None, :]) ** 2 + (eta ** 2) / 4.0) * (eta / (2 * np.pi))
    coeff, _ = nnls(A, y_np)
    coeff = coeff.astype(np.float32)
    idx = np.argsort(coeff)[-n:]
    E0 = np.sort(E_grid[idx]).astype(np.float32)
    B0 = coeff[idx][np.argsort(E_grid[idx])].astype(np.float32)

    for k in range(1, n):
        if E0[k] - E0[k - 1] < min_spacing:
            E0[k] = E0[k - 1] + min_spacing
    z0 = _pack_poles_and_strengths_np(E0, B0, wmin, min_spacing)

    def residuals(z):
        E_tf, B_tf = _unpack_poles_and_strengths_tf(z, n, tf.constant(wmin, tf.float32), tf.constant(min_spacing, tf.float32))
        yhat_tf = give_me_Lorentzian(omega_np, E_tf, B_tf, tf.constant(eta, tf.float32))
        r = yhat_tf.numpy() - y_np
        if l2 > 0:
            r = np.concatenate([r, np.sqrt(l2) * np.asarray(z, dtype=np.float32)])
        return r

    res = least_squares(residuals, z0, method="trf", max_nfev=5000, xtol=1e-10, ftol=1e-10, gtol=1e-10)
    E_tf, B_tf = _unpack_poles_and_strengths_tf(res.x, n, tf.constant(wmin, tf.float32), tf.constant(min_spacing, tf.float32))
    yhat_tf = give_me_Lorentzian(omega_np, E_tf, B_tf, tf.constant(eta, tf.float32))
    return E_tf.numpy(), B_tf.numpy(), yhat_tf.numpy()


# -----------------------------------------------------------------------------
# Generic ansatz packing / unpacking
# -----------------------------------------------------------------------------
def _n_basis_from_config(config: AnsatzConfig) -> int:
    p = int(config.n_params)
    if config.ansatz == "linear":
        return p
    if config.ansatz == "linear_exp":
        return 2 * p
    if config.ansatz == "quadratic":
        return p + p * (p + 1) // 2
    raise ValueError(f"Unknown ansatz {config.ansatz!r}.")



def get_packed_layout(config: AnsatzConfig) -> PackedLayout:
    n = int(config.n)
    p = int(config.n_params)
    n_upper = n * (n + 1) // 2
    n_basis = _n_basis_from_config(config)

    idx = 0
    eta_size = 1
    idx += eta_size

    v0_slice = slice(idx, idx + n)
    idx += n

    v_linear_size = p * n if config.use_vector_terms else 0
    v_linear_slice = slice(idx, idx + v_linear_size)
    idx += v_linear_size

    d_diag_slice = slice(idx, idx + n)
    idx += n

    basis_slice = slice(idx, idx + n_basis * n_upper)
    idx += n_basis * n_upper

    width_bias_slice = slice(idx, idx + 1)
    idx += 1

    width_linear_size = p if config.width_model == "affine" else 0
    width_linear_slice = slice(idx, idx + width_linear_size)
    idx += width_linear_size

    feature_param_size = p if config.ansatz == "linear_exp" else 0
    feature_param_slice = slice(idx, idx + feature_param_size)
    idx += feature_param_size

    return PackedLayout(
        eta_size=eta_size,
        v0_slice=v0_slice,
        v_linear_slice=v_linear_slice,
        d_diag_slice=d_diag_slice,
        basis_slice=basis_slice,
        width_bias_slice=width_bias_slice,
        width_linear_slice=width_linear_slice,
        feature_param_slice=feature_param_slice,
        n_upper=n_upper,
        n_basis=n_basis,
        total_size=idx,
    )



def _sym_from_upper(flat_upper: tf.Tensor, n: int) -> tf.Tensor:
    flat_upper = tf.cast(flat_upper, tf.float32)
    upper_idx = np.triu_indices(n)
    indices = tf.constant(np.column_stack(upper_idx), dtype=tf.int32)
    mat = tf.tensor_scatter_nd_update(tf.zeros((n, n), dtype=tf.float32), indices, flat_upper)
    return mat + tf.transpose(mat) - tf.linalg.diag(tf.linalg.diag_part(mat))



def unpack_trainable_parameters(params: tf.Tensor, config: AnsatzConfig) -> Dict[str, tf.Tensor]:
    params = tf.convert_to_tensor(params, dtype=tf.float32)
    layout = get_packed_layout(config)
    n, p = int(config.n), int(config.n_params)

    eta0 = params[0]
    v0 = params[layout.v0_slice]

    if config.use_vector_terms and layout.v_linear_slice.stop > layout.v_linear_slice.start:
        v_linear = tf.reshape(params[layout.v_linear_slice], (p, n))
    else:
        v_linear = tf.zeros((p, n), dtype=tf.float32)

    d_diag = params[layout.d_diag_slice]
    D = tf.linalg.diag(d_diag)

    basis_flat = params[layout.basis_slice]
    basis_mats = tf.reshape(basis_flat, (layout.n_basis, layout.n_upper))
    basis_mats = tf.map_fn(lambda x: _sym_from_upper(x, n), basis_mats, fn_output_signature=tf.float32)

    width_bias = params[layout.width_bias_slice][0]
    if config.width_model == "affine" and layout.width_linear_slice.stop > layout.width_linear_slice.start:
        width_linear = params[layout.width_linear_slice]
    else:
        width_linear = tf.zeros((p,), dtype=tf.float32)

    if config.ansatz == "linear_exp":
        feature_params = tf.nn.softplus(params[layout.feature_param_slice])
    else:
        feature_params = tf.zeros((0,), dtype=tf.float32)

    return {
        "eta0": eta0,
        "v0": v0,
        "v_linear": v_linear,
        "D": D,
        "d_diag": d_diag,
        "basis_mats": basis_mats,
        "width_bias": width_bias,
        "width_linear": width_linear,
        "feature_params": feature_params,
        "layout": layout,
    }



def compute_ansatz_features(param_shifts: tf.Tensor, config: AnsatzConfig, feature_params: Optional[tf.Tensor] = None) -> tf.Tensor:
    dx = tf.convert_to_tensor(param_shifts, dtype=tf.float32)
    if dx.shape.rank == 1:
        dx = dx[None, :]
    p = int(config.n_params)

    if config.ansatz == "linear":
        return dx

    if config.ansatz == "linear_exp":
        if feature_params is None:
            feature_params = tf.ones((p,), dtype=tf.float32)
        decay = tf.reshape(feature_params, (1, p))
        damped = dx * tf.exp(-decay * tf.abs(dx))
        return tf.concat([dx, damped], axis=1)

    if config.ansatz == "quadratic":
        feats = [dx]
        quad_terms = []
        for i in range(p):
            for j in range(i, p):
                quad_terms.append(dx[:, i] * dx[:, j])
        if quad_terms:
            feats.append(tf.stack(quad_terms, axis=1))
        return tf.concat(feats, axis=1)

    raise ValueError(f"Unknown ansatz {config.ansatz!r}.")



def build_model_matrices_and_vectors(
    params: tf.Tensor,
    config: AnsatzConfig,
    param_values: tf.Tensor,
    central_point: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    unpacked = unpack_trainable_parameters(params, config)
    dx = tf.cast(param_values, tf.float32) - tf.cast(central_point[None, :], tf.float32)
    features = compute_ansatz_features(dx, config, unpacked["feature_params"])

    M_batch = unpacked["D"][None, :, :] + tf.einsum('bf,fij->bij', features, unpacked["basis_mats"])
    v_batch = unpacked["v0"][None, :] + tf.einsum('bp,pn->bn', dx, unpacked["v_linear"])

    if config.width_model == "constant":
        eta_batch = tf.fill((tf.shape(dx)[0],), tf.abs(unpacked["eta0"]))
    else:
        width_affine = unpacked["width_bias"] + tf.einsum('bp,p->b', dx, unpacked["width_linear"])
        eta_batch = tf.sqrt(tf.square(unpacked["eta0"]) + tf.square(width_affine))

    return M_batch, v_batch, eta_batch, features



def make_random_initial_guess(config: AnsatzConfig, seed: Optional[int] = None, fold: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    layout = get_packed_layout(config)
    vec = rng.normal(scale=0.05, size=layout.total_size).astype(np.float32)
    vec[0] = np.float32(fold)
    vec[layout.width_bias_slice] = np.array([0.0], dtype=np.float32)
    if config.ansatz == "linear_exp":
        vec[layout.feature_param_slice] = np.full(layout.feature_param_slice.stop - layout.feature_param_slice.start, 0.2, dtype=np.float32)
    return vec



def encode_initial_guess(random_initial_guess, E, B, config_or_n, retain):
    """
    Generic version of the old initializer.

    It fills eta, D diagonal, and v0 based on the fitted central spectrum while leaving
    the ansatz-specific correction terms untouched.
    """
    if isinstance(config_or_n, AnsatzConfig):
        config = config_or_n
    else:
        config = AnsatzConfig(n=int(config_or_n), n_params=2, ansatz="linear_exp")

    layout = get_packed_layout(config)
    n = int(config.n)
    params = np.asarray(random_initial_guess, dtype=np.float32).copy()

    k_keep = int(round(float(retain) * n))
    k_keep = max(1, min(k_keep, n))
    left = (n - k_keep) // 2
    right = left + k_keep

    E = np.asarray(E, dtype=np.float32).reshape(-1)
    B = np.asarray(B, dtype=np.float32).reshape(-1)
    order = np.argsort(E)
    E, B = E[order], B[order]
    if len(E) < k_keep:
        raise ValueError(f"E,B need at least k_keep={k_keep} entries (got {len(E)}).")

    start = (len(E) - k_keep) // 2
    E_sel = E[start:start + k_keep]
    B_sel = B[start:start + k_keep]

    D_full = np.empty(n, dtype=np.float32)
    D_full[left:right] = E_sel
    cur = E_sel[0]
    for i in range(left - 1, -1, -1):
        cur -= 2.0
        D_full[i] = cur
    cur = E_sel[-1]
    for i in range(right, n):
        cur += 2.0
        D_full[i] = cur

    v0_full = np.zeros(n, dtype=np.float32)
    v0_full[left:right] = np.sqrt(np.maximum(B_sel, 0.0)).astype(np.float32)

    params[0] = params[0] if params.size > 0 else 0.0
    params[layout.v0_slice] = v0_full
    params[layout.d_diag_slice] = D_full
    return params


# -----------------------------------------------------------------------------
# Losses and compatibility wrappers
# -----------------------------------------------------------------------------
@tf.function
def calculate_alphaD(eigenvalues, B):
    mask = tf.cast(eigenvalues > 1.0, dtype=tf.float32)
    B = B * mask
    fac = tf.constant(ALPHAD_FAC, dtype=tf.float32)
    return tf.reduce_sum(B / tf.maximum(eigenvalues, 1e-6)) * fac


@tf.function
def cost_function_batched_generic(
    params,
    config: AnsatzConfig,
    param_values,
    strength_true,
    alphaD_true,
    central_point,
    retain,
    w_strength,
    w_mminus1,
    w_mplus1,
    m1_target=875.0,
    eps=1e-8,
):
    M_batch, v_batch, eta_batch, _ = build_model_matrices_and_vectors(params, config, param_values, central_point)
    eigenvalues, eigenvectors = tf.linalg.eigh(M_batch)

    n_i = tf.shape(eigenvalues)[1]
    k_keep = tf.cast(tf.round(tf.cast(retain, tf.float32) * tf.cast(n_i, tf.float32)), tf.int32)
    k_keep = tf.clip_by_value(k_keep, 1, n_i)
    left = (n_i - k_keep) // 2
    right = left + k_keep

    eigenvalues_kept = eigenvalues[:, left:right]
    eigvecsT_full = tf.transpose(eigenvectors, [0, 2, 1])
    eigvecsT = eigvecsT_full[:, left:right, :]

    proj = tf.matmul(eigvecsT, v_batch[:, :, None])
    proj = tf.squeeze(proj, axis=-1)
    B_batch = tf.square(proj)

    omega_tensor = tf.cast(strength_true[0][:, 0], tf.float32)
    strength_true_tensor = tf.stack([tf.cast(s[:, 1], tf.float32) for s in strength_true], axis=0)
    Lor_batch = give_me_Lorentzian_batched(omega_tensor[None, :], eigenvalues_kept, B_batch, eta_batch / 2.0)

    d = omega_tensor[1:] - omega_tensor[:-1]
    w0 = d[0] / 2.0
    wN = d[-1] / 2.0
    w_mid = (omega_tensor[2:] - omega_tensor[:-2]) / 2.0 if tf.shape(omega_tensor)[0] > 2 else tf.zeros([0], tf.float32)
    trap_w = tf.concat([[w0], w_mid, [wN]], axis=0)
    trap_w = tf.maximum(trap_w, 0.0)

    diff = Lor_batch - strength_true_tensor
    num = tf.reduce_sum(tf.square(diff) * trap_w[None, :], axis=1)
    den = tf.reduce_sum(tf.square(strength_true_tensor) * trap_w[None, :], axis=1) + eps
    L_strength = tf.reduce_mean(num / den)

    mask = tf.cast(eigenvalues_kept > 1.0, tf.float32)
    alphaD_calc = tf.reduce_sum((B_batch * mask) / tf.maximum(eigenvalues_kept, 1e-6), axis=1) * tf.constant(ALPHAD_FAC, tf.float32)
    alphaD_true_tensor = tf.cast(alphaD_true, tf.float32)
    alphaD_scale = tf.math.reduce_std(alphaD_true_tensor) + eps
    L_mminus1 = tf.reduce_mean(tf.square((alphaD_calc - alphaD_true_tensor) / alphaD_scale))

    m1_calc = tf.reduce_sum(B_batch * eigenvalues_kept * mask, axis=1)
    m1_scale = tf.maximum(tf.abs(tf.cast(m1_target, tf.float32)), 1.0)
    L_mplus1 = tf.reduce_mean(tf.square((m1_calc - tf.cast(m1_target, tf.float32)) / m1_scale))

    total_cost = tf.cast(w_strength, tf.float32) * L_strength + tf.cast(w_mminus1, tf.float32) * L_mminus1 + tf.cast(w_mplus1, tf.float32) * L_mplus1
    strength_cost = tf.cast(w_strength, tf.float32) * L_strength
    alphaD_cost = tf.cast(w_mminus1, tf.float32) * L_mminus1
    m1_cost = tf.cast(w_mplus1, tf.float32) * L_mplus1

    return total_cost, strength_cost, alphaD_cost, m1_cost, Lor_batch[-1], strength_true_tensor[-1], omega_tensor, alphaD_calc, B_batch[-1], eigenvalues_kept[-1]


@tf.function
def cost_function_batched_mixed(
    params,
    n,
    fmt_data,
    strength_true,
    alphaD_true,
    central_point,
    retain,
    w_strength,
    w_mminus1,
    w_mplus1,
    m1_target=875.0,
    eps=1e-8,
):
    """Backward-compatible wrapper for the old 2-parameter project."""
    if isinstance(fmt_data, (list, tuple)) and len(fmt_data) > 0 and not isinstance(fmt_data[0], (float, int, np.floating)):
        param_values = np.asarray([[float(x) for x in row] for row in fmt_data], dtype=np.float32)
    else:
        param_values = np.asarray(fmt_data, dtype=np.float32)

    config = AnsatzConfig(n=int(n), n_params=param_values.shape[1], ansatz="linear_exp", width_model="affine", use_vector_terms=True)
    return cost_function_batched_generic(
        params=params,
        config=config,
        param_values=tf.convert_to_tensor(param_values, tf.float32),
        strength_true=strength_true,
        alphaD_true=alphaD_true,
        central_point=tf.convert_to_tensor(central_point, tf.float32),
        retain=retain,
        w_strength=w_strength,
        w_mminus1=w_mminus1,
        w_mplus1=w_mplus1,
        m1_target=m1_target,
        eps=eps,
    )


# -----------------------------------------------------------------------------
# Compatibility helpers used elsewhere in older notebooks/scripts
# -----------------------------------------------------------------------------
def nec_mat(n):
    D = np.diag(np.random.uniform(1, 10, n))
    A = np.random.uniform(1, 10, (n, n))
    S1 = np.abs(A + A.T) / 2
    S2 = np.abs(A + A.T) / 2
    return D, S1, S2



def data_table(fmt_data, strength_dir='../dipoles_data_all/total_strength/', alphaD_dir='../dipoles_data_all/total_alphaD/'):
    strength = []
    alphaD = []
    for frmt in fmt_data:
        key = [str(x) for x in frmt]
        strength_file = os.path.join(strength_dir, 'strength_' + '_'.join(key[::-1]) + '.out')
        alphaD_file = os.path.join(alphaD_dir, 'alphaD_' + '_'.join(key[::-1]) + '.out')
        strength.append(np.loadtxt(strength_file))
        alphaD.append(np.loadtxt(alphaD_file))
    return strength, np.asarray(alphaD)



def count_trainable_parameters(config: AnsatzConfig) -> int:
    return get_packed_layout(config).total_size



def summarize_config(config: AnsatzConfig) -> Dict[str, int | str | bool]:
    layout = get_packed_layout(config)
    return {
        "n": config.n,
        "n_params": config.n_params,
        "ansatz": config.ansatz,
        "width_model": config.width_model,
        "use_vector_terms": config.use_vector_terms,
        "n_basis": layout.n_basis,
        "n_upper": layout.n_upper,
        "n_trainable": layout.total_size,
    }


# -----------------------------------------------------------------------------
# AlphaD-only emulator helpers
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class AlphaDOnlyConfig:
    n: int
    n_params: int
    ansatz: str = "linear"
    alphaD_mode: str = "mid_eigenvalue"   # mid_eigenvalue | sum_inverse_positive



def load_generic_alphaD_dataset(
    strength_dir: str,
    alphaD_dir: Optional[str] = None,
    strength_regex: str = r"strength_(?P<p2>[0-9.]+)_(?P<p1>[0-9.]+)\.out",
    alphaD_regex: Optional[str] = None,
    filter_ranges: Optional[str | Dict[str, Sequence[float]]] = None,
    central_point: Optional[Sequence[float] | str] = None,
) -> GenericDataset:
    if isinstance(central_point, str):
        central_point = json.loads(central_point)
    return load_dataset(
        strength_dir=strength_dir,
        alphaD_dir=alphaD_dir,
        strength_regex=strength_regex,
        alphaD_regex=alphaD_regex,
        filter_ranges=filter_ranges,
        central_point=central_point,
    )



def summarize_alphaD_only_config(config: AlphaDOnlyConfig) -> Dict[str, int | str]:
    return {
        "n": config.n,
        "n_params": config.n_params,
        "ansatz": config.ansatz,
        "alphaD_mode": config.alphaD_mode,
        "n_basis": _n_basis_from_config(AnsatzConfig(n=config.n, n_params=config.n_params, ansatz=config.ansatz)),
        "n_trainable": count_alphaD_only_parameters(config),
    }



def count_alphaD_only_parameters(config: AlphaDOnlyConfig) -> int:
    n_upper = config.n * (config.n + 1) // 2
    n_basis = _n_basis_from_config(AnsatzConfig(n=config.n, n_params=config.n_params, ansatz=config.ansatz))
    feature_param_size = config.n_params if config.ansatz == "linear_exp" else 0
    return config.n + n_basis * n_upper + feature_param_size



def count_em2_parameters_generic(n: int, n_params: int, ansatz: str = "linear_exp") -> int:
    config = AlphaDOnlyConfig(n=n, n_params=n_params, ansatz=ansatz)
    return count_alphaD_only_parameters(config)



def unpack_alphaD_only_parameters(params: tf.Tensor, config: AlphaDOnlyConfig) -> Dict[str, tf.Tensor]:
    params = tf.convert_to_tensor(params, dtype=tf.float32)
    n = int(config.n)
    n_upper = n * (n + 1) // 2
    ansatz_config = AnsatzConfig(n=n, n_params=int(config.n_params), ansatz=config.ansatz)
    n_basis = _n_basis_from_config(ansatz_config)

    idx = 0
    d_diag = params[idx:idx + n]
    idx += n

    basis_flat = params[idx:idx + n_basis * n_upper]
    idx += n_basis * n_upper
    basis_mats = tf.reshape(basis_flat, (n_basis, n_upper))
    basis_mats = tf.map_fn(lambda x: _sym_from_upper(x, n), basis_mats, fn_output_signature=tf.float32)

    if config.ansatz == "linear_exp":
        feature_params = tf.nn.softplus(params[idx:idx + config.n_params])
    else:
        feature_params = tf.zeros((0,), dtype=tf.float32)

    return {
        "D": tf.linalg.diag(d_diag),
        "d_diag": d_diag,
        "basis_mats": basis_mats,
        "feature_params": feature_params,
    }



def build_alphaD_only_matrices(
    params: tf.Tensor,
    config: AlphaDOnlyConfig,
    param_values: tf.Tensor,
    central_point: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    unpacked = unpack_alphaD_only_parameters(params, config)
    dx = tf.cast(param_values, tf.float32) - tf.cast(central_point[None, :], tf.float32)
    ansatz_config = AnsatzConfig(n=config.n, n_params=config.n_params, ansatz=config.ansatz)
    features = compute_ansatz_features(dx, ansatz_config, unpacked["feature_params"])
    M_batch = unpacked["D"][None, :, :] + tf.einsum('bf,fij->bij', features, unpacked["basis_mats"])
    return M_batch, features



def alphaD_from_eigenvalues_batch(eigenvalues: tf.Tensor, mode: str, eps: float = 1.0e-8) -> tf.Tensor:
    eigenvalues = tf.cast(eigenvalues, tf.float32)
    if mode == "mid_eigenvalue":
        n_i = tf.shape(eigenvalues)[1]
        mid_idx = tf.maximum(0, (n_i - 1) // 2)
        return eigenvalues[:, mid_idx]
    if mode == "sum_inverse_positive":
        mask = tf.cast(eigenvalues > 1.0, tf.float32)
        return tf.reduce_sum(mask / tf.maximum(eigenvalues, eps), axis=1)
    raise ValueError(f"Unknown alphaD_mode {mode!r}.")



def make_alphaD_only_loss_fn_generic(
    n: int,
    param_values,
    alphaD_true,
    central_point,
    ansatz: str = "linear_exp",
    alphaD_mode: str = "mid_eigenvalue",
    l2_reg: float = 0.0,
):
    param_values_tf = tf.convert_to_tensor(param_values, dtype=tf.float32)
    alphaD_true_tf = tf.convert_to_tensor(alphaD_true, dtype=tf.float32)
    central_point_tf = tf.convert_to_tensor(central_point, dtype=tf.float32)
    config = AlphaDOnlyConfig(
        n=int(n),
        n_params=int(param_values_tf.shape[1]),
        ansatz=ansatz,
        alphaD_mode=alphaD_mode,
    )

    def loss_fn(params: tf.Tensor):
        M_batch, _ = build_alphaD_only_matrices(params, config, param_values_tf, central_point_tf)
        eigenvalues, _ = tf.linalg.eigh(M_batch)
        alphaD_pred = alphaD_from_eigenvalues_batch(eigenvalues, config.alphaD_mode)

        scale = tf.math.reduce_std(alphaD_true_tf) + tf.constant(1.0e-8, tf.float32)
        mse = tf.reduce_mean(tf.square((alphaD_pred - alphaD_true_tf) / scale))
        if l2_reg > 0:
            mse = mse + tf.cast(l2_reg, tf.float32) * tf.reduce_mean(tf.square(tf.cast(params, tf.float32)))
        return mse, alphaD_pred

    return loss_fn
