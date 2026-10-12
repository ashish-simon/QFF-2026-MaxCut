"""Custom backend processor models and synthetic noise models using Qiskit Aer.

This module defines:
1. Processor A: 5-qubit backend with linear chain coupling map [(0,1), (1,2), (2,3), (3,4)].
2. Processor B: 9-qubit backend with heavy-hex-inspired coupling map
   [(0,1), (1,2), (1,6), (2,3), (2,6), (2,5), (3,4), (4,5), (5,6), (6,7), (6,8), (8,0)].
3. Synthetic Noise Model:
   - 0.1% (0.001) 1-qubit depolarizing error on {rz, sx, x}
   - 0.5% (0.005) 2-qubit depolarizing error on CX
   - 1.0% (0.01) symmetric readout error on measurement
4. Functions:
   - construct_noise_model()
   - apply_noise_model(backend, noise_model)
   - get_processor_a(with_noise=True)
   - get_processor_b(with_noise=True)
   - configure_both_processors(noise_model)
"""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple, Union

from qiskit.circuit import Measure, Parameter
from qiskit.circuit.library import CXGate, RZGate, SXGate, XGate
from qiskit.transpiler import CouplingMap, Target
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

# Basis gates specification
BASIS_GATES: List[str] = ["rz", "sx", "x", "cx"]

# Processor A specifications: 5 qubits, linear chain
PROCESSOR_A_NUM_QUBITS: int = 5
PROCESSOR_A_COUPLING_MAP: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
]

# Processor B specifications: 9 qubits, sparse heavy-hex-inspired
PROCESSOR_B_NUM_QUBITS: int = 9
PROCESSOR_B_COUPLING_MAP: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (1, 6),
    (2, 3),
    (2, 6),
    (2, 5),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (6, 8),
    (8, 0),
]


def construct_noise_model(
    p1_error: float = 0.001,
    p2_error: float = 0.005,
    p_readout: float = 0.01,
    single_qubit_gates: Optional[List[str]] = None,
    two_qubit_gates: Optional[List[str]] = None,
) -> NoiseModel:
    """Construct a synthetic Qiskit Aer NoiseModel.

    Includes:
    - 0.1% (0.001) 1-qubit depolarizing error on single-qubit gates
    - 0.5% (0.005) 2-qubit depolarizing error on CX gates
    - 1.0% (0.01) symmetric readout error on measurement

    Parameters
    ----------
    p1_error : float, default=0.001 (0.1%)
        Depolarizing error probability for 1-qubit gates.
    p2_error : float, default=0.005 (0.5%)
        Depolarizing error probability for 2-qubit CX gates.
    p_readout : float, default=0.01 (1.0%)
        Symmetric readout error probability per qubit.
    single_qubit_gates : list of str, optional
        List of single-qubit gates to apply depolarizing error to.
        Defaults to ['rz', 'sx', 'x'].
    two_qubit_gates : list of str, optional
        List of two-qubit gates to apply depolarizing error to.
        Defaults to ['cx'].

    Returns
    -------
    NoiseModel
        Configured Qiskit Aer NoiseModel.
    """
    if single_qubit_gates is None:
        single_qubit_gates = ["rz", "sx", "x"]
    if two_qubit_gates is None:
        two_qubit_gates = ["cx"]

    noise_model = NoiseModel(basis_gates=BASIS_GATES)

    # 1-qubit depolarizing error (0.1%)
    if p1_error > 0:
        err_1q = depolarizing_error(p1_error, 1)
        noise_model.add_all_qubit_quantum_error(err_1q, single_qubit_gates)

    # 2-qubit CX depolarizing error (0.5%)
    if p2_error > 0:
        err_2q = depolarizing_error(p2_error, 2)
        noise_model.add_all_qubit_quantum_error(err_2q, two_qubit_gates)

    # Readout error (1.0%)
    if p_readout > 0:
        ro_probs = [
            [1.0 - p_readout, p_readout],
            [p_readout, 1.0 - p_readout],
        ]
        ro_error = ReadoutError(ro_probs)
        noise_model.add_all_qubit_readout_error(ro_error)

    return noise_model


# Aliases
build_noise_model = construct_noise_model
create_noise_model = construct_noise_model


def build_processor_target(
    num_qubits: int,
    coupling_map: List[Tuple[int, int]],
) -> Target:
    """Build a Qiskit Target object from scratch with {rz, sx, x, cx} and measure."""
    target = Target(num_qubits=num_qubits)
    phi = Parameter("phi")
    target.add_instruction(RZGate(phi))
    target.add_instruction(SXGate())
    target.add_instruction(XGate())
    target.add_instruction(Measure())

    cx_props = {tuple(edge): None for edge in coupling_map}
    target.add_instruction(CXGate(), cx_props)
    return target


def create_processor(
    num_qubits: int,
    coupling_map: List[Tuple[int, int]],
    backend_name: str = "custom_processor",
    noise_model: Optional[NoiseModel] = None,
    with_noise: bool = True,
) -> AerSimulator:
    """Build a custom AerSimulator backend model from scratch.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the processor.
    coupling_map : list of (int, int)
        Coupling map edge list.
    backend_name : str, default='custom_processor'
        Name identifier for the backend.
    noise_model : NoiseModel, optional
        Pre-built noise model to attach. If None and with_noise=True,
        construct_noise_model() is called.
    with_noise : bool, default=True
        Whether to apply the synthetic noise model.

    Returns
    -------
    AerSimulator
        The custom configured backend simulator.
    """
    target = build_processor_target(num_qubits=num_qubits, coupling_map=coupling_map)

    selected_noise_model = None
    if with_noise:
        selected_noise_model = (
            noise_model if noise_model is not None else construct_noise_model()
        )

    backend = AerSimulator(
        target=target,
        coupling_map=coupling_map,
        basis_gates=BASIS_GATES,
        noise_model=selected_noise_model,
    )

    # Configure backend metadata
    backend._configuration.n_qubits = num_qubits
    backend._configuration.coupling_map = [list(e) for e in coupling_map]
    backend._configuration.backend_name = backend_name
    backend._configuration.basis_gates = list(BASIS_GATES)

    # Attach convenient attributes for direct inspection
    setattr(backend, "custom_coupling_map", coupling_map)
    setattr(backend, "custom_num_qubits", num_qubits)
    setattr(backend, "custom_basis_gates", list(BASIS_GATES))

    return backend


def apply_noise_model(
    backend: AerSimulator,
    noise_model: Optional[NoiseModel] = None,
) -> AerSimulator:
    """Apply a synthetic noise model to an existing AerSimulator processor backend.

    Parameters
    ----------
    backend : AerSimulator
        The processor backend.
    noise_model : NoiseModel, optional
        The noise model to apply. If None, the default synthetic noise model is used.

    Returns
    -------
    AerSimulator
        The backend with noise model configured.
    """
    if noise_model is None:
        noise_model = construct_noise_model()
    backend.set_options(noise_model=noise_model)
    return backend


def get_processor_a(with_noise: bool = True) -> AerSimulator:
    """Return Processor A (5 qubits, linear chain coupling map, {rz, sx, x, cx}).

    Linear chain coupling map: [(0,1), (1,2), (2,3), (3,4)].

    Parameters
    ----------
    with_noise : bool, default=True
        If True, applies the synthetic noise model (0.1% 1Q, 0.5% 2Q CX, 1.0% readout).

    Returns
    -------
    AerSimulator
        Configured backend for Processor A.
    """
    return create_processor(
        num_qubits=PROCESSOR_A_NUM_QUBITS,
        coupling_map=PROCESSOR_A_COUPLING_MAP,
        backend_name="processor_a",
        with_noise=with_noise,
    )


def get_processor_b(with_noise: bool = True) -> AerSimulator:
    """Return Processor B (9 qubits, heavy-hex-inspired coupling map, {rz, sx, x, cx}).

    Coupling map: [(0,1), (1,2), (1,6), (2,3), (2,6), (2,5), (3,4), (4,5), (5,6), (6,7), (6,8), (8,0)].

    Parameters
    ----------
    with_noise : bool, default=True
        If True, applies the synthetic noise model (0.1% 1Q, 0.5% 2Q CX, 1.0% readout).

    Returns
    -------
    AerSimulator
        Configured backend for Processor B.
    """
    return create_processor(
        num_qubits=PROCESSOR_B_NUM_QUBITS,
        coupling_map=PROCESSOR_B_COUPLING_MAP,
        backend_name="processor_b",
        with_noise=with_noise,
    )


def configure_both_processors(
    noise_model: Optional[NoiseModel] = None,
) -> Tuple[AerSimulator, AerSimulator]:
    """Construct and apply the synthetic noise model to BOTH Processor A and Processor B.

    Parameters
    ----------
    noise_model : NoiseModel, optional
        Pre-built noise model. If None, construct_noise_model() is called.

    Returns
    -------
    Tuple[AerSimulator, AerSimulator]
        (processor_a, processor_b) with the synthetic noise model applied.
    """
    nm = noise_model if noise_model is not None else construct_noise_model()
    proc_a = get_processor_a(with_noise=False)
    proc_b = get_processor_b(with_noise=False)
    apply_noise_model(proc_a, nm)
    apply_noise_model(proc_b, nm)
    return proc_a, proc_b


# Aliases for constructing and applying noise model to both processors
apply_noise_model_to_both = configure_both_processors
construct_and_apply_noise_model_to_both = configure_both_processors
build_and_apply_noise_model = configure_both_processors
