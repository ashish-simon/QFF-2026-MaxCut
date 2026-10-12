"""QAOA circuit construction and transpilation for routing evaluation.

This module provides:
1. build_qaoa_circuit(): Builds a QAOA circuit with p=1 from the cost Hamiltonian
   (or input graph) using standard elementary gates and optional measurement.
2. transpile_qaoa_circuit(): Transpiles the QAOA circuit to a target backend
   (Processor A or B) using optimization_level=1 and seed_transpiler=42.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

import networkx as nx
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

from src.problem import construct_cost_hamiltonian


def build_qaoa_circuit(
    cost_hamiltonian: Union[SparsePauliOp, nx.Graph],
    p: int = 1,
    gamma: Union[float, Parameter, List[Union[float, Parameter]]] = 0.3,
    beta: Union[float, Parameter, List[Union[float, Parameter]]] = 1.2,
    insert_measurements: bool = True,
) -> QuantumCircuit:
    """Build a QAOA quantum circuit with p=1 using the cost Hamiltonian.

    Parameters
    ----------
    cost_hamiltonian : SparsePauliOp or nx.Graph
        The cost Hamiltonian operator (or NetworkX graph). If a graph is provided,
        the anchored 5-qubit cost Hamiltonian is constructed automatically.
    p : int, default=1
        Number of QAOA alternating operator layers (reps).
    gamma : float, Parameter, or list, default=0.3
        Cost layer rotation angle(s).
    beta : float, Parameter, or list, default=1.2
        Mixer layer rotation angle(s).
    insert_measurements : bool, default=True
        Whether to append measure_all() at the end of the circuit.

    Returns
    -------
    QuantumCircuit
        The constructed QAOA circuit.
    """
    if isinstance(cost_hamiltonian, nx.Graph):
        cost_hamiltonian = construct_cost_hamiltonian(cost_hamiltonian, include_offset=False)

    num_qubits = cost_hamiltonian.num_qubits

    # Format parameter lists for p layers
    gamma_list = gamma if isinstance(gamma, list) else [gamma] * p
    beta_list = beta if isinstance(beta, list) else [beta] * p

    qc = QuantumCircuit(num_qubits, name=f"QAOA_p{p}")

    # Initial state: |+>^n superposition
    for q in range(num_qubits):
        qc.h(q)

    # Alternate Cost and Mixer layers for p steps
    for step in range(p):
        g = gamma_list[step]
        b = beta_list[step]

        # --- Cost Layer U(C, gamma) = exp(-i * gamma * H_C) ---
        for pauli_str, coeff in cost_hamiltonian.to_list():
            # In Qiskit, index 0 is rightmost character: num_qubits - 1 - i
            z_qubits = [num_qubits - 1 - i for i, char in enumerate(pauli_str) if char == "Z"]
            weight = coeff.real

            if len(z_qubits) == 1:
                # Single-qubit Z term from anchored Node 0: exp(-i * gamma * w * Z) -> RZ(2 * gamma * w)
                q = z_qubits[0]
                qc.rz(2 * g * weight, q)
            elif len(z_qubits) == 2:
                # Two-qubit ZZ term: exp(-i * gamma * w * ZZ) -> CX - RZ(2 * gamma * w) - CX
                u, v = z_qubits
                qc.cx(u, v)
                qc.rz(2 * g * weight, v)
                qc.cx(u, v)
            # Identity term ('I' * num_qubits) is a global phase and omitted

        # --- Mixer Layer U(B, beta) = exp(-i * beta * sum X_j) ---
        for q in range(num_qubits):
            qc.rx(2 * b, q)

    if insert_measurements:
        qc.measure_all()

    qc.metadata = {
        "p": p,
        "gamma": gamma_list,
        "beta": beta_list,
        "num_qubits": num_qubits,
    }

    return qc


# Alias for intuitive discovery
create_qaoa_circuit = build_qaoa_circuit


def transpile_qaoa_circuit(
    circuit: QuantumCircuit,
    backend: Any,
    optimization_level: int = 1,
    seed_transpiler: int = 42,
    **transpile_kwargs: Any,
) -> QuantumCircuit:
    """Transpile a QAOA circuit for a target processor backend.

    Parameters
    ----------
    circuit : QuantumCircuit
        The QAOA circuit to transpile.
    backend : Backend or AerSimulator
        Target processor model (Processor A or Processor B).
    optimization_level : int, default=1
        Transpilation optimization level (0, 1, 2, or 3).
    seed_transpiler : int, default=42
        Random seed for stochastic routing/layout passes.
    **transpile_kwargs
        Additional arguments passed to qiskit.transpile().

    Returns
    -------
    QuantumCircuit
        The transpiled quantum circuit mapped to the backend basis gates and coupling map.
    """
    return transpile(
        circuit,
        backend=backend,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        **transpile_kwargs,
    )


# Aliases
transpile_circuit = transpile_qaoa_circuit
transpile_to_backend = transpile_qaoa_circuit
