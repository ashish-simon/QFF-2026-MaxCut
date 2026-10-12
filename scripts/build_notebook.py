"""Script to generate and execute main.ipynb."""

import json
import os

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Project Q: 6-Node Max-Cut QAOA Benchmark\n",
            "### Evaluating Routing Overhead and Noise Sensitivity across Custom Processor Topologies\n",
            "\n",
            "This notebook provides an end-to-end reproducible workflow for:\n",
            "1. **Problem Formulation**: 6-node Erd\u0151s\u2013R\u00e9nyi graph ($n=6, p=0.5, \\text{seed}=42$) with integer edge weights $\\in [1, 5]$.\n",
            "2. **Classical Baseline**: Exact brute-force solver finding the baseline Max-Cut value ($C_{\\text{max}} = 16.0$).\n",
            "3. **Quantum Formulation**: Anchoring Node 0 to state '0' to reduce the problem to exactly 5 quantum variables ($N=5$).\n",
            "4. **Processor Topologies**:\n",
            "   - **Processor A**: 5-qubit linear chain `[(0,1), (1,2), (2,3), (3,4)]`\n",
            "   - **Processor B**: 9-qubit heavy-hex-inspired `[(0,1), (1,2), (1,6), (2,3), (2,6), (2,5), (3,4), (4,5), (5,6), (6,7), (6,8), (8,0)]`\n",
            "   - **Noise Model**: 0.1% 1Q depolarizing, 0.5% 2Q CX depolarizing, 1.0% readout error.\n",
            "5. **QAOA Routing & Transpilation**: Building $p=1$ QAOA circuits and compiling with `optimization_level=1, seed_transpiler=42`.\n",
            "6. **Execution & Metrics**: 4096-shot noisy simulations, metric extraction (Depth, CX Count, SWAP Count, Approximation Ratio), and export to `results/tables/metrics.csv` and `results/figures/`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 1: Imports and Environment Setup\n",
            "import os\n",
            "import sys\n",
            "\n",
            "# Ensure src is in Python path\n",
            "sys.path.insert(0, os.path.abspath('.'))\n",
            "\n",
            "import matplotlib.pyplot as plt\n",
            "import networkx as nx\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "from qiskit import QuantumCircuit\n",
            "\n",
            "from src.problem import (\n",
            "    generate_graph,\n",
            "    construct_cost_hamiltonian,\n",
            "    solve_maxcut_brute_force,\n",
            "    compute_cut_value,\n",
            ")\n",
            "from src.processors import (\n",
            "    get_processor_a,\n",
            "    get_processor_b,\n",
            "    configure_both_processors,\n",
            ")\n",
            "from src.routing import build_qaoa_circuit, transpile_qaoa_circuit\n",
            "from src.metrics import extract_routing_metrics, execute_and_evaluate\n",
            "from src.visualization import (\n",
            "    draw_problem_graph,\n",
            "    draw_processor_coupling_maps,\n",
            "    plot_metric_comparison,\n",
            ")\n",
            "\n",
            "print('All modules successfully imported!')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Problem Formulation & Exact Classical Baseline\n",
            "\n",
            "We generate a 6-node Erd\u0151s\u2013R\u00e9nyi random graph ($p=0.5, \\text{seed}=42$) and assign random integer edge weights between 1 and 5.\n",
            "Then we solve for the exact maximum cut value using the classical brute-force solver."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 2: Graph Generation and Classical Brute-Force Solve\n",
            "G = generate_graph(n=6, p=0.5, seed=42, weight_seed=42)\n",
            "print(f'Graph nodes: {list(G.nodes())}')\n",
            "print(f'Graph edges with weights: {list(G.edges(data=True))}')\n",
            "\n",
            "# Exact brute-force solver\n",
            "baseline_res = solve_maxcut_brute_force(G)\n",
            "max_cut_baseline = float(baseline_res)\n",
            "best_partition = baseline_res.best_partition\n",
            "best_bitstring = baseline_res.best_bitstring\n",
            "\n",
            "print('\\n' + '=' * 50)\n",
            "print(f'EXACT MAX-CUT BASELINE: {max_cut_baseline}')\n",
            "print(f'Optimal Partition: {best_partition}')\n",
            "print(f'Optimal Bitstring (Node 0 to 5): {best_bitstring}')\n",
            "print('=' * 50)\n",
            "\n",
            "# Visualize and save problem graph\n",
            "fig_graph, ax_graph = draw_problem_graph(G, partition=best_partition, save_path='results/figures/problem_graph.png')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Quantum Cost Hamiltonian Construction (Anchoring Node 0)\n",
            "\n",
            "Max-Cut has global $\\mathbb{Z}_2$ spin-flip symmetry ($x \\leftrightarrow 1 - x$).\n",
            "By anchoring **Node 0** to state '0' ($s_0 = +1$), we eliminate the symmetry and reduce the problem from 6 variables to exactly 5 quantum variables ($q_0 \\dots q_4$), allowing it to fit directly on a 5-qubit processor."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 3: Construct Cost Hamiltonian\n",
            "cost_hamiltonian, offset = construct_cost_hamiltonian(\n",
            "    G, anchor_node=0, anchor_state=0, return_offset=True\n",
            ")\n",
            "\n",
            "print(f'Number of qubits: {cost_hamiltonian.num_qubits}')\n",
            "print(f'Number of Pauli terms: {len(cost_hamiltonian)}')\n",
            "print(f'Constant energy offset: {offset}')\n",
            "print('\\nHamiltonian Pauli terms:')\n",
            "for pauli, coeff in cost_hamiltonian.to_list():\n",
            "    print(f'  {pauli} : {coeff.real:+.2f}')\n",
            "\n",
            "# Verify ground state energy matches -MaxCut\n",
            "H_full = construct_cost_hamiltonian(G, anchor_node=0, anchor_state=0, include_offset=True)\n",
            "eigvals = np.linalg.eigvalsh(H_full.to_matrix())\n",
            "ground_energy = min(eigvals)\n",
            "print(f'\\nGround State Energy: {ground_energy:.4f}')\n",
            "print(f'Verified: Ground State Energy equals -MaxCut? {np.isclose(ground_energy, -max_cut_baseline)}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Hardware Processor Models & Synthetic Noise\n",
            "\n",
            "We configure:\n",
            "- **Processor A**: 5 qubits, linear chain coupling map `[(0,1), (1,2), (2,3), (3,4)]`\n",
            "- **Processor B**: 9 qubits, heavy-hex-inspired coupling map `[(0,1), (1,2), (1,6), (2,3), (2,6), (2,5), (3,4), (4,5), (5,6), (6,7), (6,8), (8,0)]`\n",
            "- **Synthetic Noise Model**: 0.1% 1Q depolarizing, 0.5% 2Q CX depolarizing, 1.0% readout error."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 4: Initialize Processors and Visualize Coupling Maps\n",
            "proc_a, proc_b = configure_both_processors()\n",
            "\n",
            "print(f'Processor A: {proc_a.num_qubits} qubits, basis: {proc_a.configuration().basis_gates}')\n",
            "print(f'Processor B: {proc_b.num_qubits} qubits, basis: {proc_b.configuration().basis_gates}')\n",
            "\n",
            "# Visualize and save coupling maps\n",
            "fig_maps, axs_maps = draw_processor_coupling_maps(save_path='results/figures/coupling_maps.png')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. QAOA Circuit Construction ($p=1$) & Transpilation\n",
            "\n",
            "We construct the QAOA ansatz circuit for $p=1$ with parameters $\\gamma = 0.3$ and $\\beta = 1.2$, then transpile to both backends with `optimization_level=1` and `seed_transpiler=42`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 5: Build QAOA Circuit and Transpile\n",
            "qaoa_circuit = build_qaoa_circuit(\n",
            "    cost_hamiltonian, p=1, gamma=0.3, beta=1.2, insert_measurements=True\n",
            ")\n",
            "print('Unrouted QAOA Circuit Summary:')\n",
            "print(f'  Qubits: {qaoa_circuit.num_qubits}')\n",
            "print(f'  Operations: {dict(qaoa_circuit.count_ops())}')\n",
            "print(f'  Depth: {qaoa_circuit.depth()}')\n",
            "\n",
            "# Transpile to Processor A\n",
            "transpiled_a = transpile_qaoa_circuit(\n",
            "    qaoa_circuit, proc_a, optimization_level=1, seed_transpiler=42\n",
            ")\n",
            "\n",
            "# Transpile to Processor B\n",
            "transpiled_b = transpile_qaoa_circuit(\n",
            "    qaoa_circuit, proc_b, optimization_level=1, seed_transpiler=42\n",
            ")\n",
            "\n",
            "# Extract routing metrics\n",
            "metrics_a = extract_routing_metrics(transpiled_a)\n",
            "metrics_b = extract_routing_metrics(transpiled_b)\n",
            "\n",
            "print('\\nRouting Metrics:')\n",
            "print(f\"  Processor A (5Q Linear):    Depth={metrics_a['depth']}, CX={metrics_a['cx_count']}, SWAPs={metrics_a['swap_count']}\")\n",
            "print(f\"  Processor B (9Q Heavy-Hex): Depth={metrics_b['depth']}, CX={metrics_b['cx_count']}, SWAPs={metrics_b['swap_count']}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Noisy Execution & Benchmark Evaluation (4096 Shots)\n",
            "\n",
            "We execute the transpiled circuits on both noisy backends using 4096 shots, decode the measured partitions, compute the expected cut value $\\langle C \\rangle$, and calculate the approximation ratio $r = \\langle C \\rangle / C_{\\text{max}}$."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 6: Execute and Evaluate on Noisy Backends\n",
            "exec_res_a = execute_and_evaluate(transpiled_a, proc_a, graph=G, shots=4096, seed_simulator=42)\n",
            "exec_res_b = execute_and_evaluate(transpiled_b, proc_b, graph=G, shots=4096, seed_simulator=42)\n",
            "\n",
            "print('=' * 60)\n",
            "print(f'Processor A Execution (Linear Chain):')\n",
            "print(f'  Expected Cut:        {exec_res_a.expected_cut:.4f} / {max_cut_baseline}')\n",
            "print(f'  Approximation Ratio: {exec_res_a.approximation_ratio:.4%}')\n",
            "print(f'  Best Sampled Cut:    {exec_res_a.best_cut_observed}')\n",
            "\n",
            "print(f'\\nProcessor B Execution (Heavy-Hex):')\n",
            "print(f'  Expected Cut:        {exec_res_b.expected_cut:.4f} / {max_cut_baseline}')\n",
            "print(f'  Approximation Ratio: {exec_res_b.approximation_ratio:.4%}')\n",
            "print(f'  Best Sampled Cut:    {exec_res_b.best_cut_observed}')\n",
            "print('=' * 60)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Comparative Benchmark Metrics & Export\n",
            "\n",
            "We compile the comparative benchmark metrics into a `pandas.DataFrame`, export it to `results/tables/metrics.csv`, and visualize the routing overhead using a grouped bar chart saved to `results/figures/metric_comparison.png`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 7: Compile DataFrame, Export Table, and Plot Comparison\n",
            "metrics_df = pd.DataFrame(\n",
            "    {\n",
            "        'Processor A (5Q Linear)': [\n",
            "            metrics_a['depth'],\n",
            "            metrics_a['cx_count'],\n",
            "            metrics_a['swap_count'],\n",
            "            metrics_a['routing_overhead_cx'],\n",
            "            f\"{exec_res_a.expected_cut:.4f}\",\n",
            "            max_cut_baseline,\n",
            "            f\"{exec_res_a.approximation_ratio:.4%}\",\n",
            "        ],\n",
            "        'Processor B (9Q Heavy-Hex)': [\n",
            "            metrics_b['depth'],\n",
            "            metrics_b['cx_count'],\n",
            "            metrics_b['swap_count'],\n",
            "            metrics_b['routing_overhead_cx'],\n",
            "            f\"{exec_res_b.expected_cut:.4f}\",\n",
            "            max_cut_baseline,\n",
            "            f\"{exec_res_b.approximation_ratio:.4%}\",\n",
            "        ],\n",
            "    },\n",
            "    index=[\n",
            "        'Circuit Depth',\n",
            "        'Total CX Count',\n",
            "        'SWAP Gate Count',\n",
            "        'Routing CX Overhead',\n",
            "        'Expected Cut',\n",
            "        'Max-Cut Baseline',\n",
            "        'Approximation Ratio',\n",
            "    ],\n",
            ")\n",
            "\n",
            "# Export to CSV\n",
            "os.makedirs('results/tables', exist_ok=True)\n",
            "csv_path = 'results/tables/metrics.csv'\n",
            "metrics_df.to_csv(csv_path)\n",
            "print(f'Benchmark metrics table exported successfully to {csv_path}!\\n')\n",
            "\n",
            "# Display formatted table\n",
            "print(metrics_df)\n",
            "\n",
            "# Generate and save grouped bar chart\n",
            "fig_comp, ax_comp = plot_metric_comparison(\n",
            "    metrics_df, save_path='results/figures/metric_comparison.png'\n",
            ")\n",
            "plt.show()"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

output_path = "e:/Project_Q/main.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"{output_path} successfully written!")
