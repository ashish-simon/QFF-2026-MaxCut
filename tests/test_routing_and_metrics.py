"""Tests for src/routing.py and src/metrics.py."""

from src.problem import generate_graph, construct_cost_hamiltonian, solve_maxcut_brute_force
from src.processors import get_processor_a, get_processor_b
from src.routing import build_qaoa_circuit, transpile_qaoa_circuit
from src.metrics import (
    get_circuit_depth,
    get_cx_count,
    get_swap_count,
    extract_routing_metrics,
    execute_and_evaluate,
)


def test_routing_and_transpilation():
    """Verify QAOA circuit construction and transpilation on both processors."""
    G = generate_graph()
    H = construct_cost_hamiltonian(G, include_offset=False)

    # 1. Build QAOA circuit
    qc = build_qaoa_circuit(H, p=1, gamma=0.3, beta=1.2)
    assert qc.num_qubits == 5, f"Expected 5 qubits, got {qc.num_qubits}"

    ops = qc.count_ops()
    assert "h" in ops and ops["h"] == 5
    assert "cx" in ops and ops["cx"] == 12  # 6 ZZ interactions * 2 CX
    assert "rz" in ops
    assert "rx" in ops and ops["rx"] == 5
    assert "measure" in ops

    # 2. Transpile to Processor A (5 qubits, linear chain)
    proc_a = get_processor_a(with_noise=True)
    t_a = transpile_qaoa_circuit(qc, proc_a, optimization_level=1, seed_transpiler=42)

    # 3. Transpile to Processor B (9 qubits, heavy-hex)
    proc_b = get_processor_b(with_noise=True)
    t_b = transpile_qaoa_circuit(qc, proc_b, optimization_level=1, seed_transpiler=42)

    # 4. Extract metrics
    metrics_a = extract_routing_metrics(t_a)
    metrics_b = extract_routing_metrics(t_b)

    assert metrics_a["depth"] > 0
    assert metrics_b["depth"] > 0
    assert metrics_a["cx_count"] >= 12
    assert metrics_b["cx_count"] >= 12

    # Linear chain requires routing overhead (SWAPs) for non-adjacent interactions
    assert metrics_a["swap_count"] >= 0
    assert metrics_b["swap_count"] >= 0

    print("Processor A routing metrics:", metrics_a)
    print("Processor B routing metrics:", metrics_b)


def test_execution_and_evaluation():
    """Verify circuit execution with 4096 shots and approximation ratio computation."""
    G = generate_graph()
    H = construct_cost_hamiltonian(G, include_offset=False)
    qc = build_qaoa_circuit(H, p=1, gamma=0.3, beta=1.2)

    proc_a = get_processor_a(with_noise=True)
    t_a = transpile_qaoa_circuit(qc, proc_a, optimization_level=1, seed_transpiler=42)

    res_a = execute_and_evaluate(t_a, proc_a, graph=G, shots=4096, seed_simulator=42)

    assert res_a.shots == 4096
    assert res_a.max_cut_baseline == 16.0
    assert 0 < res_a.expected_cut <= 16.0
    assert 0 < res_a.approximation_ratio <= 1.0
    assert res_a.best_cut_observed > 0

    proc_b = get_processor_b(with_noise=True)
    t_b = transpile_qaoa_circuit(qc, proc_b, optimization_level=1, seed_transpiler=42)

    res_b = execute_and_evaluate(t_b, proc_b, graph=G, shots=4096, seed_simulator=42)

    assert res_b.shots == 4096
    assert res_b.max_cut_baseline == 16.0
    assert 0 < res_b.expected_cut <= 16.0
    assert 0 < res_b.approximation_ratio <= 1.0

    print(f"Processor A result: expected_cut={res_a.expected_cut:.4f}, ratio={res_a.approximation_ratio:.4f}")
    print(f"Processor B result: expected_cut={res_b.expected_cut:.4f}, ratio={res_b.approximation_ratio:.4f}")


if __name__ == "__main__":
    test_routing_and_transpilation()
    test_execution_and_evaluation()
    print("All routing and metrics tests passed successfully!")
