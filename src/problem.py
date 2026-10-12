"""Problem formulation for 6-node Max-Cut on a 5-qubit quantum processor.

This module provides:
1. Graph generation: Erdős–Rényi 6-node graph with random integer edge weights (1-5).
2. Qiskit cost Hamiltonian construction: Anchoring Node 0 to state '0' to reduce
   the problem to 5 quantum variables for a 5-qubit processor.
3. Classical brute-force solver: Exact evaluation of all partitions to determine
   the baseline maximum cut value.
"""

from __future__ import annotations

import itertools
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
from qiskit.quantum_info import SparsePauliOp


def generate_graph(
    n: int = 6,
    p: float = 0.5,
    seed: int = 42,
    weight_seed: Optional[int] = 42,
    weight_range: Tuple[int, int] = (1, 5),
) -> nx.Graph:
    """Generate a random Erdős–Rényi graph with integer edge weights.

    Parameters
    ----------
    n : int, default=6
        Number of nodes in the graph.
    p : float, default=0.5
        Probability for edge creation.
    seed : int, default=42
        Seed for nx.erdos_renyi_graph generation.
    weight_seed : int or None, default=42
        Seed for the random integer edge weights. If None, system randomness is used.
    weight_range : tuple of (int, int), default=(1, 5)
        Range (min_weight, max_weight) inclusive for random integer edge weights.

    Returns
    -------
    nx.Graph
        The generated NetworkX graph with integer 'weight' attributes on edges.
    """
    G = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
    rng = random.Random(weight_seed) if weight_seed is not None else random.Random()
    min_w, max_w = weight_range

    for u, v in G.edges():
        G[u][v]["weight"] = rng.randint(min_w, max_w)

    return G


# Alias for intuitive discovery
generate_random_graph = generate_graph


def construct_cost_hamiltonian(
    G: nx.Graph,
    anchor_node: int = 0,
    anchor_state: int = 0,
    include_offset: bool = False,
    return_offset: bool = False,
) -> Union[SparsePauliOp, Tuple[SparsePauliOp, float]]:
    r"""Construct the Qiskit SparsePauliOp cost Hamiltonian for Max-Cut on the graph.

    By anchoring `anchor_node` (default 0) to `anchor_state` (default '0'),
    the $\mathbb{Z}_2$ bit-flip symmetry of Max-Cut is broken and the problem is
    reduced from $n$ variables to $n - 1$ variables. For a 6-node graph, this yields
    exactly 5 quantum variables fitting onto a 5-qubit processor.

    Mathematical Formulation:
    -------------------------
    The Max-Cut objective function for partition $x_v \in \{0, 1\}$ is:
        $C(\vec{x}) = \sum_{(u, v) \in E} w_{uv} (x_u + x_v - 2 x_u x_v)$

    With spin mapping $s_v = 1 - 2 x_v \in \{+1, -1\}$ where $x_v = 0 \iff s_v = +1$:
        $x_u + x_v - 2 x_u x_v = \frac{1 - s_u s_v}{2}$

    Anchoring Node 0 to state '0' sets $s_0 = +1$:
    - For edge $(0, v)$ ($v \ge 1$, mapped to qubit $q = v - 1$):
        $\frac{1 - s_0 s_v}{2} = \frac{1 - (+1) s_v}{2} = \frac{1 - Z_{v-1}}{2}$
    - For edge $(u, v)$ with $u, v \ge 1$ (mapped to qubits $q_u = u - 1, q_v = v - 1$):
        $\frac{1 - s_u s_v}{2} = \frac{1 - Z_{u-1} Z_{v-1}}{2}$

    In quantum optimization (QAOA / VQE), we minimize the cost Hamiltonian:
        $H_{\text{cost}} = - C = H_{\text{Ising}} + \text{offset} \cdot I$
    where:
        $H_{\text{Ising}} = \sum_{(0, v) \in E} \frac{w_{0v}}{2} Z_{v-1} + \sum_{u, v \ge 1, (u, v) \in E} \frac{w_{uv}}{2} Z_{u-1} Z_{v-1}$
        $\text{offset} = -\frac{1}{2} \sum_{(u, v) \in E} w_{uv}$

    Parameters
    ----------
    G : nx.Graph
        The input weighted graph.
    anchor_node : int, default=0
        The node to anchor to eliminate bit-flip symmetry.
    anchor_state : int, default=0
        The fixed binary state of the anchored node (0 or 1).
    include_offset : bool, default=False
        If True, includes the constant identity term `offset * I` in the SparsePauliOp.
        When included, eigenvalues of $H$ directly equal $- \text{Cut}(\vec{x})$.
        When False (standard for QAOA ansatz), only non-identity Pauli terms are returned.
    return_offset : bool, default=False
        If True, returns a tuple `(SparsePauliOp, offset)`.

    Returns
    -------
    SparsePauliOp or Tuple[SparsePauliOp, float]
        The 5-qubit cost Hamiltonian SparsePauliOp (and optionally the constant offset float).
        The returned SparsePauliOp also carries `.offset` as an attribute.
    """
    nodes = sorted(list(G.nodes()))
    if anchor_node not in nodes:
        raise ValueError(f"Anchor node {anchor_node} not in graph nodes {nodes}.")

    remaining_nodes = [v for v in nodes if v != anchor_node]
    num_qubits = len(remaining_nodes)
    node_to_qubit = {node: idx for idx, node in enumerate(remaining_nodes)}

    s_anchor = 1 if anchor_state == 0 else -1

    pauli_list: List[Tuple[str, complex]] = []
    total_weight = 0.0

    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        total_weight += w

        if u == anchor_node or v == anchor_node:
            other = v if u == anchor_node else u
            q = node_to_qubit[other]
            # Qiskit Pauli string order: qubit 0 is rightmost (index num_qubits - 1 - q)
            z_chars = ["I"] * num_qubits
            z_chars[num_qubits - 1 - q] = "Z"
            pauli_str = "".join(z_chars)
            coeff = (w / 2.0) * s_anchor
            pauli_list.append((pauli_str, coeff))
        else:
            qu = node_to_qubit[u]
            qv = node_to_qubit[v]
            z_chars = ["I"] * num_qubits
            z_chars[num_qubits - 1 - qu] = "Z"
            z_chars[num_qubits - 1 - qv] = "Z"
            pauli_str = "".join(z_chars)
            coeff = w / 2.0
            pauli_list.append((pauli_str, coeff))

    offset = -0.5 * total_weight

    if include_offset:
        pauli_list.append(("I" * num_qubits, offset))

    hamiltonian = SparsePauliOp.from_list(pauli_list).simplify()
    setattr(hamiltonian, "offset", offset)
    setattr(hamiltonian, "total_weight", total_weight)
    setattr(hamiltonian, "node_to_qubit", node_to_qubit)
    setattr(hamiltonian, "anchor_node", anchor_node)
    setattr(hamiltonian, "anchor_state", anchor_state)

    if return_offset:
        return hamiltonian, offset
    return hamiltonian


# Aliases for flexible naming
construct_maxcut_hamiltonian = construct_cost_hamiltonian
get_cost_hamiltonian = construct_cost_hamiltonian
get_maxcut_hamiltonian = construct_cost_hamiltonian


class MaxCutResult(float):
    """Result container for Max-Cut brute-force solver.

    Subclasses float so it can be used directly as a numeric value:
        assert solver(G) == max_cut_val
    while supporting tuple unpacking:
        val, partition = solver(G)
    and attribute access:
        res.max_cut, res.best_partition, res.best_bitstring
    """

    def __new__(
        cls,
        max_cut: float,
        best_bitstring: str = "",
        best_partition: Optional[Dict[int, int]] = None,
        all_optimal_partitions: Optional[List[Dict[int, int]]] = None,
    ):
        obj = super().__new__(cls, max_cut)
        obj.max_cut = float(max_cut)
        obj.best_bitstring = best_bitstring
        obj.best_partition = best_partition if best_partition is not None else {}
        obj.all_optimal_partitions = (
            all_optimal_partitions if all_optimal_partitions is not None else []
        )
        return obj

    def __iter__(self):
        return iter((self.max_cut, self.best_partition))

    def __getitem__(self, item):
        return (self.max_cut, self.best_partition)[item]

    def __repr__(self) -> str:
        return (
            f"MaxCutResult(max_cut={self.max_cut}, "
            f"best_bitstring='{self.best_bitstring}', "
            f"best_partition={self.best_partition})"
        )


def compute_cut_value(G: nx.Graph, partition: Dict[int, int]) -> float:
    """Compute the cut value of a graph given a partition dict {node: 0 or 1}."""
    cut = 0.0
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        if partition.get(u, 0) != partition.get(v, 0):
            cut += w
    return cut


def solve_maxcut_brute_force(
    G: nx.Graph,
    anchor_node: Optional[int] = 0,
    anchor_state: int = 0,
) -> MaxCutResult:
    """Find the exact Max-Cut value and partition by classical brute force.

    Parameters
    ----------
    G : nx.Graph
        The input graph with edge weights.
    anchor_node : int or None, default=0
        If specified, anchors this node to `anchor_state` to halve the search space
        (exploiting Z_2 symmetry). If None, searches all 2^n configurations.
    anchor_state : int, default=0
        State for the anchored node (0 or 1).

    Returns
    -------
    MaxCutResult
        A result object representing the maximum cut value (behaves as float,
        can be unpacked as (max_cut, best_partition)).
    """
    nodes = sorted(list(G.nodes()))
    n = len(nodes)

    if anchor_node is not None and anchor_node in nodes:
        free_nodes = [v for v in nodes if v != anchor_node]
        best_cut = -1.0
        best_partition: Dict[int, int] = {}
        all_optimal: List[Dict[int, int]] = []

        for bits in itertools.product([0, 1], repeat=len(free_nodes)):
            part = {anchor_node: anchor_state}
            for node, b in zip(free_nodes, bits):
                part[node] = b

            cut_val = compute_cut_value(G, part)
            if cut_val > best_cut:
                best_cut = cut_val
                best_partition = part
                all_optimal = [part]
            elif cut_val == best_cut:
                all_optimal.append(part)

        bitstring = "".join(str(best_partition[node]) for node in nodes)
        return MaxCutResult(
            max_cut=best_cut,
            best_bitstring=bitstring,
            best_partition=best_partition,
            all_optimal_partitions=all_optimal,
        )
    else:
        best_cut = -1.0
        best_partition = {}
        all_optimal = []

        for bits in itertools.product([0, 1], repeat=n):
            part = {node: b for node, b in zip(nodes, bits)}
            cut_val = compute_cut_value(G, part)
            if cut_val > best_cut:
                best_cut = cut_val
                best_partition = part
                all_optimal = [part]
            elif cut_val == best_cut:
                all_optimal.append(part)

        bitstring = "".join(str(best_partition[node]) for node in nodes)
        return MaxCutResult(
            max_cut=best_cut,
            best_bitstring=bitstring,
            best_partition=best_partition,
            all_optimal_partitions=all_optimal,
        )


# Aliases for classical brute-force solver
brute_force_maxcut = solve_maxcut_brute_force
classical_maxcut_brute_force = solve_maxcut_brute_force
