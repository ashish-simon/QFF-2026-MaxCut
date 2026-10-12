"""Tests for src/problem.py."""

import numpy as np
import networkx as nx
from qiskit.quantum_info import SparsePauliOp

from src.problem import (
    generate_graph,
    generate_random_graph,
    construct_cost_hamiltonian,
    construct_maxcut_hamiltonian,
    solve_maxcut_brute_force,
    compute_cut_value,
    MaxCutResult,
)


def test_graph_generation():
    """Verify 6-node graph generation and edge weights."""
    G = generate_graph()
    assert len(G.nodes) == 6, f"Expected 6 nodes, got {len(G.nodes)}"
    assert set(G.nodes) == {0, 1, 2, 3, 4, 5}

    # Compare with reference nx.erdos_renyi_graph(n=6, p=0.5, seed=42)
    G_ref = nx.erdos_renyi_graph(n=6, p=0.5, seed=42)
    assert set(G.edges()) == set(G_ref.edges()), "Edges must match Erdos-Renyi graph"

    for u, v, data in G.edges(data=True):
        assert "weight" in data, f"Edge ({u}, {v}) missing weight"
        w = data["weight"]
        assert isinstance(w, int), f"Weight {w} should be integer"
        assert 1 <= w <= 5, f"Weight {w} should be in [1, 5]"


def test_cost_hamiltonian_and_anchoring():
    """Verify Hamiltonian has 5 qubits and accurately reflects cut values."""
    G = generate_graph()
    H = construct_cost_hamiltonian(G, anchor_node=0, anchor_state=0)

    assert H.num_qubits == 5, f"Expected 5 qubits, got {H.num_qubits}"

    # Verify Hamiltonian with offset matches cut values on all 32 basis states
    H_full = construct_cost_hamiltonian(G, anchor_node=0, anchor_state=0, include_offset=True)
    matrix = H_full.to_matrix()
    eigvals = np.linalg.eigvalsh(matrix)
    min_energy = min(eigvals)

    # Brute force cut
    res = solve_maxcut_brute_force(G)
    max_cut_val = float(res)

    # Ground state energy must equal -max_cut_val
    assert np.isclose(min_energy, -max_cut_val), (
        f"Ground state energy {min_energy} does not equal -MaxCut {-max_cut_val}"
    )

    # Test state-by-state cut mapping
    # Basis state |b4 b3 b2 b1 b0> where q_k is bit for node (k+1)
    for idx in range(32):
        # bitstring of 5 qubits
        bits_5 = [(idx >> k) & 1 for k in range(5)]  # bits_5[k] is bit for qubit k (node k+1)
        partition = {0: 0}
        for k in range(5):
            partition[k + 1] = bits_5[k]

        cut = compute_cut_value(G, partition)
        # Expected energy is -cut
        diag_energy = matrix[idx, idx].real
        assert np.isclose(diag_energy, -cut), (
            f"State {idx} (bits {bits_5}): diagonal energy {diag_energy} != -cut {-cut}"
        )


def test_brute_force_solver():
    """Verify brute-force solver output and interface."""
    G = generate_graph()
    res = solve_maxcut_brute_force(G)

    assert isinstance(res, MaxCutResult)
    assert isinstance(res, float)
    assert res > 0, "Max cut should be positive"
    assert res == res.max_cut
    assert len(res.best_partition) == 6
    assert res.best_partition[0] == 0
    assert len(res.best_bitstring) == 6

    # Verify partition evaluation matches max_cut
    cut = compute_cut_value(G, res.best_partition)
    assert cut == res.max_cut

    # Test tuple unpacking
    val, part = res
    assert val == res.max_cut
    assert part == res.best_partition


if __name__ == "__main__":
    test_graph_generation()
    test_cost_hamiltonian_and_anchoring()
    test_brute_force_solver()
    print("All tests passed successfully!")
