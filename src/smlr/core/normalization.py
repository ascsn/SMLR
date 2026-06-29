fit_normalizer(params, method="???")
transform_params(params, normalizer)
inverse_transform_params(x_norm, normalizer)
# purpose is to stabilize parameter coordiante representation for loaded datasets,
# and for proper initialization for the emulator training.