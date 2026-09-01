"""Human-Informed Neural Networks (HINN)."""

from .simulation import (
    BNNHuman,
    RNNMachine,
    expected_calibration_error,
    generate_financial_dataset,
    main,
    make_labels,
    train_hinn,
    trust_lambda,
)

__all__ = [
    "BNNHuman",
    "RNNMachine",
    "expected_calibration_error",
    "generate_financial_dataset",
    "main",
    "make_labels",
    "train_hinn",
    "trust_lambda",
]
