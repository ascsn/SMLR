@dataclass(frozen=True)
class ParameterNormalizer:
    method: str
    offset: np.ndarray
    scale: np.ndarray

fit_normalizer(params, method="???")
transform_params(params, normalizer)
inverse_transform_params(x_norm, normalizer)
# purpose is to stabilize parameter coordiante representation for loaded datasets,
# and for proper initialization for the emulator training.