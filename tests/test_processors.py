"""Tests for src/processors.py."""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from src.processors import (
    BASIS_GATES,
    PROCESSOR_A_COUPLING_MAP,
    PROCESSOR_A_NUM_QUBITS,
    PROCESSOR_B_COUPLING_MAP,
    PROCESSOR_B_NUM_QUBITS,
    apply_noise_model,
    configure_both_processors,
    construct_noise_model,
    get_processor_a,
    get_processor_b,
)


def test_noise_model_construction():
    """Verify the synthetic noise model parameters and errors."""
    nm = construct_noise_model()
    assert isinstance(nm, NoiseModel)

    # Check basis gates in noise model
    for gate in ["rz", "sx", "x", "cx"]:
        assert gate in nm.basis_gates

    # Test custom parameters
    nm_custom = construct_noise_model(p1_error=0.002, p2_error=0.01, p_readout=0.02)
    assert isinstance(nm_custom, NoiseModel)


def test_processor_a():
    """Verify Processor A configuration, coupling map, basis gates, and simulation."""
    proc_a = get_processor_a(with_noise=True)
    assert isinstance(proc_a, AerSimulator)
    assert proc_a.num_qubits == 5
    assert proc_a.configuration().n_qubits == 5

    # Check coupling map
    expected_edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    cmap_edges = [tuple(e) for e in proc_a.coupling_map.get_edges()]
    assert set(cmap_edges) == set(expected_edges)

    # Check basis gates in target
    target_ops = set(proc_a.target.operation_names)
    for gate in ["rz", "sx", "x", "cx"]:
        assert gate in target_ops, f"Gate {gate} missing from target"

    # Test circuit compilation and noisy run
    qc = QuantumCircuit(5, 5)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(range(2), range(2))

    t_qc = transpile(qc, proc_a, seed_transpiler=42)
    ops = set(t_qc.count_ops().keys())
    assert ops.issubset({"rz", "sx", "x", "cx", "measure", "barrier"})

    result = proc_a.run(t_qc, shots=100).result()
    counts = result.get_counts()
    assert len(counts) > 0


def test_processor_b():
    """Verify Processor B configuration, heavy-hex coupling map, and simulation."""
    proc_b = get_processor_b(with_noise=True)
    assert isinstance(proc_b, AerSimulator)
    assert proc_b.num_qubits == 9
    assert proc_b.configuration().n_qubits == 9

    # Check coupling map
    expected_edges = [
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
    cmap_edges = [tuple(e) for e in proc_b.coupling_map.get_edges()]
    assert set(cmap_edges) == set(expected_edges)

    # Check basis gates in target
    target_ops = set(proc_b.target.operation_names)
    for gate in ["rz", "sx", "x", "cx"]:
        assert gate in target_ops, f"Gate {gate} missing from target"

    # Test circuit compilation and noisy run
    qc = QuantumCircuit(9, 9)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 6)
    qc.cx(6, 8)
    qc.cx(8, 0)
    qc.measure(range(9), range(9))

    t_qc = transpile(qc, proc_b, seed_transpiler=42)
    ops = set(t_qc.count_ops().keys())
    assert ops.issubset({"rz", "sx", "x", "cx", "measure", "barrier"})

    result = proc_b.run(t_qc, shots=100).result()
    counts = result.get_counts()
    assert len(counts) > 0


def test_configure_both_processors():
    """Verify function that constructs and applies noise model to both processors."""
    p_a, p_b = configure_both_processors()
    assert p_a.num_qubits == 5
    assert p_b.num_qubits == 9
    assert p_a.options.noise_model is not None
    assert p_b.options.noise_model is not None


if __name__ == "__main__":
    test_noise_model_construction()
    test_processor_a()
    test_processor_b()
    test_configure_both_processors()
    print("All processor tests passed successfully!")
