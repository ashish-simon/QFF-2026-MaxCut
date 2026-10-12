"""Circuit metric extraction and Max-Cut evaluation.

This module provides:
1. Metric extractors:
   - get_circuit_depth(): Extracts total circuit depth.
   - get_cx_count(): Extracts total 2-qubit CX gate count.
   - get_swap_count(): Extracts total SWAP gate count (including routed SWAPs synthesized to CX).
   - extract_routing_metrics(): Returns a comprehensive dictionary of routing overhead metrics.
2. Execution and Evaluation:
   - execute_and_evaluate(): Executes a transpiled circuit on a backend (using 4096 shots),
     calculates the expected Max-Cut value, and evaluates the approximation ratio against
     the exact classical brute-force baseline.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import networkx as nx
from qiskit import QuantumCircuit

from src.problem import compute_cut_value, generate_graph, solve_maxcut_brute_force


def get_circuit_depth(circuit: QuantumCircuit) -> int:
    """Extract the circuit depth of a transpiled circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The quantum circuit.

    Returns
    -------
    int
        Circuit depth.
    """
    return circuit.depth()


def get_cx_count(circuit: QuantumCircuit) -> int:
    """Extract the total CX gate count from a circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The quantum circuit.

    Returns
    -------
    int
        Total number of CX gates.
    """
    return int(circuit.count_ops().get("cx", 0))


def get_swap_count(
    circuit: QuantumCircuit,
    base_cx_count: int = 12,
) -> int:
    """Extract the total SWAP gate count to evaluate routing overhead.

    If explicit 'swap' gates exist in the circuit, their count is returned.
    When transpiling to backends with basis gate set {rz, sx, x, cx}, SWAP gates
    inserted during routing are synthesized into 3 CX gates. In this case,
    the number of routed SWAP gates is inferred from the CX overhead:
        inferred_swaps = max(0, (cx_count - base_cx_count) // 3)

    Parameters
    ----------
    circuit : QuantumCircuit
        The transpiled quantum circuit.
    base_cx_count : int, default=12
        The baseline number of CX gates in the unrouted QAOA circuit
        (for the 6-node anchored graph with p=1, exactly 6 ZZ terms -> 12 CX gates).

    Returns
    -------
    int
        The total count of SWAP gates (explicit or inferred from routing).
    """
    ops = circuit.count_ops()
    explicit_swaps = int(ops.get("swap", 0))
    if explicit_swaps > 0:
        return explicit_swaps

    cx_count = int(ops.get("cx", 0))
    if cx_count > base_cx_count:
        return (cx_count - base_cx_count) // 3
    return 0


def extract_routing_metrics(
    circuit: QuantumCircuit,
    base_cx_count: int = 12,
) -> Dict[str, Union[int, float]]:
    """Extract routing overhead metrics from a transpiled circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The transpiled quantum circuit.
    base_cx_count : int, default=12
        Baseline unrouted CX count for overhead calculation.

    Returns
    -------
    dict
        Dictionary containing depth, cx_count, swap_count, and routing_overhead_cx.
    """
    depth = get_circuit_depth(circuit)
    cx_count = get_cx_count(circuit)
    swap_count = get_swap_count(circuit, base_cx_count=base_cx_count)
    overhead_cx = max(0, cx_count - base_cx_count)

    return {
        "depth": depth,
        "cx_count": cx_count,
        "swap_count": swap_count,
        "routing_overhead_cx": overhead_cx,
        "base_cx_count": base_cx_count,
    }


class ExecutionResult(dict):
    """Container for execution and Max-Cut evaluation results.

    Supports both dictionary key lookup `res['expected_cut']`
    and attribute access `res.expected_cut`.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'ExecutionResult' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def execute_and_evaluate(
    transpiled_circuit: QuantumCircuit,
    backend: Any,
    graph: Optional[nx.Graph] = None,
    shots: int = 4096,
    seed_simulator: Optional[int] = 42,
) -> ExecutionResult:
    """Run a transpiled QAOA circuit on a backend and evaluate Max-Cut performance.

    Executes using the specified number of shots (default 4096), decodes measured
    bitstrings to graph partitions (with Node 0 anchored to state 0), calculates
    the expected Max-Cut value, and evaluates the approximation ratio against
    the classical brute-force baseline.

    Parameters
    ----------
    transpiled_circuit : QuantumCircuit
        The transpiled QAOA circuit with measurements.
    backend : Backend or AerSimulator
        Target processor backend.
    graph : nx.Graph, optional
        The problem graph. If None, generate_graph() is called with default parameters.
    shots : int, default=4096
        Number of measurement shots.
    seed_simulator : int or None, default=42
        Simulation random seed for reproducibility.

    Returns
    -------
    ExecutionResult
        Dictionary-like result containing:
        - expected_cut: float
        - max_cut_baseline: float
        - approximation_ratio: float
        - shots: int
        - counts: dict of {bitstring: count}
        - best_cut_observed: float
        - best_bitstring_observed: str
    """
    if graph is None:
        graph = generate_graph()

    # Calculate exact classical baseline
    baseline_result = solve_maxcut_brute_force(graph)
    exact_max_cut = float(baseline_result)

    # Run circuit on backend
    run_kwargs: Dict[str, Any] = {"shots": shots}
    if seed_simulator is not None:
        run_kwargs["seed_simulator"] = seed_simulator

    job = backend.run(transpiled_circuit, **run_kwargs)
    result = job.result()
    counts = result.get_counts()

    total_cut = 0.0
    best_cut_observed = -1.0
    best_bitstring_observed = ""

    num_qubits = 5  # Anchored graph maps 5 non-anchor nodes to 5 qubits

    for raw_bitstring, count in counts.items():
        # Remove any spaces from multi-register formatting
        bitstring = raw_bitstring.replace(" ", "")

        # In Qiskit, bitstring is reversed: bitstring[-1] is qubit 0
        reversed_bits = [int(b) for b in reversed(bitstring)]

        # Map to 6-node partition: Node 0 is anchored to 0
        partition = {0: 0}
        for q in range(min(num_qubits, len(reversed_bits))):
            partition[q + 1] = reversed_bits[q]

        cut_val = compute_cut_value(graph, partition)
        total_cut += cut_val * count

        if cut_val > best_cut_observed:
            best_cut_observed = cut_val
            best_bitstring_observed = bitstring

    expected_cut = total_cut / float(shots)
    approximation_ratio = expected_cut / exact_max_cut if exact_max_cut > 0 else 0.0

    return ExecutionResult({
        "expected_cut": expected_cut,
        "max_cut_baseline": exact_max_cut,
        "approximation_ratio": approximation_ratio,
        "shots": shots,
        "counts": counts,
        "best_cut_observed": best_cut_observed,
        "best_bitstring_observed": best_bitstring_observed,
    })


# Aliases
run_and_evaluate_maxcut = execute_and_evaluate
evaluate_maxcut_execution = execute_and_evaluate
execute_circuit = execute_and_evaluate
