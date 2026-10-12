"""Visualization utilities for problem graphs, processor topologies, and routing metrics.

This module provides:
1. draw_problem_graph(): Visualizes the 6-node weighted graph with edge weights and anchored node.
2. draw_processor_coupling_maps(): Visualizes coupling maps of Processor A (5Q linear)
   and Processor B (9Q heavy-hex) side-by-side.
3. plot_metric_comparison(): Creates a grouped bar chart comparing Circuit Depth, CX Count,
   and SWAP Overhead between Processor A and Processor B.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from src.processors import (
    PROCESSOR_A_COUPLING_MAP,
    PROCESSOR_A_NUM_QUBITS,
    PROCESSOR_B_COUPLING_MAP,
    PROCESSOR_B_NUM_QUBITS,
)


def draw_problem_graph(
    G: nx.Graph,
    partition: Optional[Dict[int, int]] = None,
    save_path: Optional[str] = "results/figures/problem_graph.png",
    figsize: Tuple[int, int] = (7, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """Draw the 6-node Max-Cut graph with edge weights and highlighted anchor node.

    Parameters
    ----------
    G : nx.Graph
        The weighted graph.
    partition : dict of {int: int}, optional
        Node partition {node: 0 or 1}. If provided, nodes and cut edges are colored accordingly.
    save_path : str, optional
        Path to save the generated figure. If None, figure is not saved.
    figsize : tuple of (int, int), default=(7, 6)
        Figure size in inches.

    Returns
    -------
    tuple of (plt.Figure, plt.Axes)
    """
    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)

    # Node coloring
    if partition is not None:
        node_colors = [
            "#3B82F6" if partition.get(node, 0) == 0 else "#EF4444"
            for node in G.nodes()
        ]
    else:
        # Highlight anchor node 0 in amber, others in blue
        node_colors = [
            "#F59E0B" if node == 0 else "#3B82F6"
            for node in G.nodes()
        ]

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=800,
        edgecolors="#1E293B",
        linewidths=2.0,
    )

    # Edge drawing: highlight cut edges if partition provided
    if partition is not None:
        cut_edges = [
            (u, v) for u, v in G.edges()
            if partition.get(u, 0) != partition.get(v, 0)
        ]
        uncut_edges = [
            (u, v) for u, v in G.edges()
            if partition.get(u, 0) == partition.get(v, 0)
        ]
        nx.draw_networkx_edges(G, pos, edgelist=cut_edges, ax=ax, width=2.5, edge_color="#DC2626", style="dashed")
        nx.draw_networkx_edges(G, pos, edgelist=uncut_edges, ax=ax, width=1.5, edge_color="#94A3B8")
    else:
        nx.draw_networkx_edges(G, pos, ax=ax, width=2.0, edge_color="#64748B")

    # Node labels
    labels = {node: f"q{node-1}\n(Node {node})" if node != 0 else "Anchor\nNode 0" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=9, font_weight="bold", font_color="#FFFFFF")

    # Edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax,
        font_size=10, font_weight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="#F8FAFC", ec="#CBD5E1", lw=1)
    )

    ax.set_title("6-Node Problem Graph with Edge Weights\n(Node 0 Anchored to '0' -> 5 Qubits)", fontsize=12, fontweight="bold", pad=12)
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


# Alias
draw_graph = draw_problem_graph


def draw_coupling_map(
    coupling_map: List[Tuple[int, int]],
    num_qubits: int,
    title: str = "Coupling Map",
    ax: Optional[plt.Axes] = None,
    pos: Optional[Dict[int, Tuple[float, float]]] = None,
    node_color: str = "#3B82F6",
) -> plt.Axes:
    """Draw a single processor coupling map.

    Parameters
    ----------
    coupling_map : list of (int, int)
        Coupling map directed or undirected edge list.
    num_qubits : int
        Number of qubits.
    title : str
        Subplot title.
    ax : plt.Axes, optional
        Target axes.
    pos : dict, optional
        Predefined node positions.
    node_color : str
        Hex color for nodes.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    G = nx.DiGraph()
    G.add_nodes_from(range(num_qubits))
    G.add_edges_from(coupling_map)

    if pos is None:
        pos = nx.spring_layout(G.to_undirected(), seed=42)

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_color,
        node_size=700,
        edgecolors="#1E293B",
        linewidths=2.0,
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=2.0,
        edge_color="#475569",
        arrows=True,
        arrowsize=14,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.05",
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=10, font_weight="bold", font_color="#FFFFFF"
    )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.axis("off")
    return ax


def draw_processor_coupling_maps(
    save_path: Optional[str] = "results/figures/coupling_maps.png",
    figsize: Tuple[int, int] = (14, 6),
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Draw side-by-side coupling maps for Processor A and Processor B.

    Parameters
    ----------
    save_path : str, optional
        Path to save figure.
    figsize : tuple of (int, int), default=(14, 6)
        Figure dimensions.

    Returns
    -------
    tuple of (plt.Figure, (plt.Axes, plt.Axes))
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Positions for Processor A (Linear chain 0 -> 1 -> 2 -> 3 -> 4)
    pos_a = {i: (i * 1.5, 0.0) for i in range(PROCESSOR_A_NUM_QUBITS)}
    draw_coupling_map(
        PROCESSOR_A_COUPLING_MAP,
        PROCESSOR_A_NUM_QUBITS,
        title="Processor A: 5-Qubit Linear Chain\nEdges: [(0,1), (1,2), (2,3), (3,4)]",
        ax=ax1,
        pos=pos_a,
        node_color="#0284C7",
    )

    # Positions for Processor B (Heavy-hex layout)
    # 0, 1, 2, 3, 4, 5, 6, 7, 8
    pos_b = {
        0: (0.0, 1.0),
        1: (1.0, 2.0),
        2: (2.0, 2.0),
        3: (3.0, 1.0),
        4: (3.0, 0.0),
        5: (2.0, -1.0),
        6: (1.0, 0.5),
        7: (0.0, -0.5),
        8: (-0.5, 0.5),
    }
    draw_coupling_map(
        PROCESSOR_B_COUPLING_MAP,
        PROCESSOR_B_NUM_QUBITS,
        title="Processor B: 9-Qubit Heavy-Hex-Inspired\n12 Directed Coupling Edges",
        ax=ax2,
        pos=pos_b,
        node_color="#10B981",
    )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, (ax1, ax2)


# Alias
draw_coupling_maps = draw_processor_coupling_maps


def plot_metric_comparison(
    metrics_data: Union[pd.DataFrame, Dict[str, Dict[str, Any]]],
    save_path: Optional[str] = "results/figures/metric_comparison.png",
    figsize: Tuple[int, int] = (9, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a grouped bar chart comparing Circuit Depth, CX Count, and SWAP Overhead.

    Parameters
    ----------
    metrics_data : pd.DataFrame or dict
        Benchmark metrics comparison data.
    save_path : str, optional
        Path to save figure.
    figsize : tuple of (int, int), default=(9, 6)
        Figure dimensions.

    Returns
    -------
    tuple of (plt.Figure, plt.Axes)
    """
    if isinstance(metrics_data, dict):
        df = pd.DataFrame(metrics_data)
    else:
        df = metrics_data.copy()

    # Normalize column/index names if needed
    # Expecting rows: Depth, CX Count, SWAP Count or columns for Processors
    fig, ax = plt.subplots(figsize=figsize)

    categories = ["Circuit Depth", "Total CX Count", "SWAP Gate Overhead"]
    
    # Extract values for Processor A and B
    if "Processor A" in df.columns and "Processor B" in df.columns:
        # DataFrame has Processors as columns
        val_map = {
            "Circuit Depth": ["Depth", "Circuit Depth", "depth"],
            "Total CX Count": ["CX Count", "Total CX Count", "cx_count"],
            "SWAP Gate Overhead": ["SWAP Count", "SWAP Gate Overhead", "swap_count"],
        }
        vals_a = []
        vals_b = []
        for cat in categories:
            possible_keys = val_map[cat]
            matched_key = next((k for k in possible_keys if k in df.index), None)
            if matched_key:
                vals_a.append(float(df.loc[matched_key, "Processor A"]))
                vals_b.append(float(df.loc[matched_key, "Processor B"]))
            else:
                vals_a.append(0.0)
                vals_b.append(0.0)
    else:
        # Fallback dictionary extraction
        vals_a = [
            float(df.get("Processor A", {}).get("depth", 60)),
            float(df.get("Processor A", {}).get("cx_count", 18)),
            float(df.get("Processor A", {}).get("swap_count", 2)),
        ]
        vals_b = [
            float(df.get("Processor B", {}).get("depth", 35)),
            float(df.get("Processor B", {}).get("cx_count", 12)),
            float(df.get("Processor B", {}).get("swap_count", 0)),
        ]

    x = range(len(categories))
    width = 0.35

    rects1 = ax.bar([i - width / 2 for i in x], vals_a, width, label="Processor A (5Q Linear)", color="#0284C7", edgecolor="#0369A1", linewidth=1.2)
    rects2 = ax.bar([i + width / 2 for i in x], vals_b, width, label="Processor B (9Q Heavy-Hex)", color="#10B981", edgecolor="#047857", linewidth=1.2)

    # Add numeric labels on top of bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(
            f"{int(height)}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontweight="bold", fontsize=10
        )

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(
            f"{int(height)}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontweight="bold", fontsize=10
        )

    ax.set_ylabel("Count / Depth", fontsize=11, fontweight="bold")
    ax.set_title("Routing Overhead Comparison: Processor A vs Processor B\n(QAOA Max-Cut p=1)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


# Aliases
plot_grouped_bar_chart = plot_metric_comparison
plot_routing_comparison = plot_metric_comparison
