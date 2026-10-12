"""Processors package."""

from src.processors import (
    BASIS_GATES,
    PROCESSOR_A_COUPLING_MAP,
    PROCESSOR_A_NUM_QUBITS,
    PROCESSOR_B_COUPLING_MAP,
    PROCESSOR_B_NUM_QUBITS,
    apply_noise_model,
    apply_noise_model_to_both,
    build_noise_model,
    configure_both_processors,
    construct_and_apply_noise_model_to_both,
    construct_noise_model,
    create_noise_model,
    create_processor,
    get_processor_a,
    get_processor_b,
)

__all__ = [
    "BASIS_GATES",
    "PROCESSOR_A_COUPLING_MAP",
    "PROCESSOR_A_NUM_QUBITS",
    "PROCESSOR_B_COUPLING_MAP",
    "PROCESSOR_B_NUM_QUBITS",
    "construct_noise_model",
    "build_noise_model",
    "create_noise_model",
    "apply_noise_model",
    "create_processor",
    "get_processor_a",
    "get_processor_b",
    "configure_both_processors",
    "apply_noise_model_to_both",
    "construct_and_apply_noise_model_to_both",
]
