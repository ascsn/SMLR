# Reproducibility

## Generate Synthetic Training Data

Create synthetic data that mimics the structure of high-fidelity
training inputs used in the SMLR workflow.

## Visualize Training Data

Use the included plotting scripts and notebooks to inspect training
data quality and parameter coverage.

## Train Surrogate Model

Train the emulator models from the `Beta_decay/` and
`Dipole_polarizability/` directories using the provided training
scripts.

## Leave-One-Out CV

Perform leave-one-out cross-validation to verify emulator performance
and identify possible overfitting.

## Prediction at New Parameter Point

Use trained surrogate models to make predictions at parameter points
that were not included in the training set.

## Conclusion

Summarize the reproducibility workflow, including the steps needed to
reproduce results and the expected outputs from the training and
prediction stages.
