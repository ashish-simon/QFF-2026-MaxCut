# Qiskit Fall Fest 2026 — Phase 1 Challenge Submission

### Track 2: Quantum Optimization (QAOA / QUBO)
### Problem Statement: PS-O1: Max-Cut on Weighted Graphs

---

## Table of Contents
- [Challenge & Track Identification](#challenge--track-identification)
- [Project Overview](#project-overview)
- [Authoritative Repository Structure](#authoritative-repository-structure)
- [Setup & Environment](#setup--environment)
- [Problem Formulation](#problem-formulation)
  - [Graph Generation](#graph-generation)
  - [Node Anchoring (Qubit Reduction)](#node-anchoring-qubit-reduction)
  - [Hamiltonian Representation](#hamiltonian-representation)
  - [Classical Brute-Force Baseline](#classical-brute-force-baseline)
- [Hardware Topologies & Architectural Justification](#hardware-topologies--architectural-justification)
- [Synthetic Noise Model](#synthetic-noise-model)
- [QAOA Routing & Transpilation Pipeline](#qaoa-routing--transpilation-pipeline)
- [Results & Benchmarking Metrics](#results--benchmarking-metrics)
- [Comprehensive Architectural Report](#comprehensive-architectural-report)
- [Reproducibility & Verification](#reproducibility--verification)

---

## Challenge & Track Identification

* **Institution**: SRM University-AP
* **Hackathon**: Qiskit Fall Fest 2026 (QFF 2026) — Phase 1
* **Selected Track**: **Track 2: Quantum Optimization (QAOA / QUBO)**
* **Problem Statement**: **PS-O1: Max-Cut on Weighted Graphs**
* **Task**: Solve Max-Cut on a 6-node weighted graph using QAOA ($p=1$). Compare cut value, approximation ratio, circuit depth, and SWAP routing overhead on **Processor A vs. Processor B**.

---

## Project Overview

Project Q investigates the algorithmic routing overhead and noise vulnerability of the Quantum Approximate Optimization Algorithm (QAOA) for solving the NP-hard Max-Cut problem. Using a standardized 6-node weighted random graph, we compare a minimal 5-qubit linear chain against a 9-qubit sparse heavy-hex-inspired processor architecture under identical synthetic noise channels.

---

## Authoritative Repository Structure

Strictly formatted according to Page 3 of the Phase 1 Master Guide:

```text
Project_Q/
├── README.md                           # Problem statement selected, setup, methodology, key results
├── main.ipynb                          # End-to-end reproducible experiment notebook
├── requirements.txt                    # Software dependencies (qiskit, qiskit-aer, networkx, etc.)
├── report.md                           # Architectural analysis, trade-off interpretation, limitations
├── data/
│   ├── raw/
│   │   └── graph_edges.json            # Raw generated 6-node weighted graph instance
│   └── processed/
│       └── hamiltonian.json            # Cleaned cost Hamiltonian Pauli operators & offset
├── processors/
│   ├── __init__.py                     # Processors package re-exports
│   ├── processor_A.json                # Participant-defined Processor A coupling map & config
│   └── processor_B.json                # Participant-defined Processor B coupling map & config
├── src/
│   ├── __init__.py
│   ├── problem.py                      # Problem formulation & cost Hamiltonian construction
│   ├── processors.py                   # Topology graph definition & noise model loaders
│   ├── routing.py                      # Transpilation & layout routing routines
│   ├── metrics.py                      # Circuit metrics (depth, CX, SWAP, fidelity)
│   └── visualization.py                # Plotting & coupling graph visualizers
├── results/
│   ├── tables/
│   │   └── metrics.csv                 # Exported CSV benchmark comparison table
│   └── figures/
│       ├── coupling_maps.png           # Exported topology visualizer (Processors A & B)
│       ├── metric_comparison.png       # Exported grouped bar chart (Depth, CX, SWAP overhead)
│       └── problem_graph.png           # Exported weighted problem graph & optimal partition
├── assets/
│   └── qff_phase1_latex_images/        # LaTeX formula renders matching master guide
│       ├── eq_bell_fidelity.png
│       ├── eq_bell_state.png
│       ├── eq_graph.png
│       ├── eq_qubit_state.png
│       ├── eq_standard_error.png
│       └── eq_zz_expectation.png
├── scripts/
│   ├── build_notebook.py               # Notebook generator script
│   └── run_notebook.py                 # Automated notebook execution runner
└── tests/
    ├── test_problem.py
    ├── test_processors.py
    └── test_routing_and_metrics.py
```

---

## Setup & Environment

To preserve system drive capacity (e.g., restricted space on `C:`), all packages and the virtual environment are strictly contained on drive `E:`.

### 1. Prerequisites
- Python 3.11 (64-bit)
- PowerShell (Windows) or bash

### 2. Virtual Environment Setup
```powershell
# Create virtual environment on drive E:
py -3.11 -m venv e:\Project_Q\.venv

# Install dependencies using local cache
& "e:\Project_Q\.venv\Scripts\python.exe" -m pip install --cache-dir "E:\pip_cache" -r requirements.txt
```

### 3. Dependencies ([`requirements.txt`](file:///e:/Project_Q/requirements.txt))
* **`qiskit`** (`2.5.2`): Quantum circuit synthesis, operator algebra, and transpiler pass managers.
* **`qiskit-aer`** (`0.17.2`): C++ simulation engine with custom noise models.
* **`networkx`** (`3.6.1`): Graph generation, topology mapping, and cut evaluation.
* **`scikit-learn`** (`1.9.1`): Numerical and statistical evaluation tools.
* **`pandas`** (`3.0.6`): Tabular benchmark metrics handling and CSV export.
* **`matplotlib`** (`3.11.2`): Figure generation and publication-grade charts.

---

## Problem Formulation

### Graph Generation
Implemented in [`src/problem.py`](file:///e:/Project_Q/src/problem.py#L22-L52), the problem instance is an Erdős–Rényi random graph generated via `nx.erdos_renyi_graph(n=6, p=0.5, seed=42)`. Random integer weights $w_{uv} \in [1, 5]$ are assigned deterministically (`weight_seed=42`).

The generated graph has 6 vertices ($V = \{0, 1, 2, 3, 4, 5\}$) and 9 edges:
* $(0, 2): 1$, $(0, 3): 1$, $(0, 4): 3$
* $(1, 4): 2$, $(1, 5): 2$
* $(2, 3): 2$, $(2, 4): 1$, $(3, 4): 5$, $(3, 5): 1$
* Total edge weight: $\sum w_{uv} = 18.0$

### Node Anchoring (Qubit Reduction)
The classical Max-Cut objective function for partition $x_v \in \{0, 1\}$ is:
$$C(\vec{x}) = \sum_{(u, v) \in E} w_{uv} (x_u + x_v - 2x_u x_v)$$

Because Max-Cut exhibits global bit-flip ($\mathbb{Z}_2$) symmetry ($x_v \leftrightarrow 1 - x_v$), every cut partition has an identical twin partition. We anchor **Node 0** to state `'0'` ($x_0 = 0$, spin $s_0 = +1$).

This breaks the $\mathbb{Z}_2$ symmetry without altering the maximum cut value, eliminating one degree of freedom and reducing the search space from $2^6 = 64$ to $2^5 = 32$. The remaining 5 nodes $\{1, 2, 3, 4, 5\}$ map directly to **5 quantum variables** ($q_0, q_1, q_2, q_3, q_4$ via $q = v - 1$), fitting onto a 5-qubit processor.

### Hamiltonian Representation
Mapping spins $s_v = 1 - 2x_v \in \{+1, -1\}$ to Pauli $Z$ operators:
* Edges connected to anchor Node 0: $\frac{w_{0v}}{2}(I - s_0 s_v) \xrightarrow{s_0 = +1} \frac{w_{0v}}{2}(I - Z_{v-1})$
* Edges between unanchored nodes ($u, v \ge 1$): $\frac{w_{uv}}{2}(I - Z_{u-1} Z_{v-1})$

The cost Hamiltonian to minimize (such that ground state energy equals $-C_{\text{max}}$) is:
$$H_{\text{cost}} = \sum_{(0, v) \in E} \frac{w_{0v}}{2} Z_{v-1} + \sum_{u, v \ge 1, (u, v) \in E} \frac{w_{uv}}{2} Z_{u-1} Z_{v-1} - \frac{1}{2}\left(\sum_{(u,v)\in E} w_{uv}\right) I$$

Constructed as a 5-qubit Qiskit `SparsePauliOp` with 9 terms and offset $-9.0$.

### Classical Brute-Force Baseline
Implemented in [`src/problem.py`](file:///e:/Project_Q/src/problem.py#L225-L290), `solve_maxcut_brute_force(G)` enumerates all configurations to establish exact ground truth:
* **Exact Max-Cut Baseline**: **$16.0$** (out of total graph weight $18.0$)
* **Optimal Partition**: $\{0: 0, 1: 0, 2: 1, 3: 0, 4: 1, 5: 1\}$ (bitstring `001011`)
* **Ground State Energy Verification**: $\min \text{eig}(H_{\text{cost}}) = -16.0000$ (verified mathematically).

---

## Hardware Topologies & Architectural Justification

### Topological Justification (Processor A vs. Processor B)
> [!IMPORTANT]
> **Architectural Justification for Benchmarking Topologies**
> The selection of a 5-qubit linear chain coupling map `[(0,1), (1,2), (2,3), (3,4)]` for **Processor A** ([`processors/processor_A.json`](file:///e:/Project_Q/processors/processor_A.json)) and the 12-edge heavy-hex-inspired graph `[(0,1), (1,2), (1,6), (2,3), (2,6), (2,5), (3,4), (4,5), (5,6), (6,7), (6,8), (8,0)]` across 9 qubits for **Processor B** ([`processors/processor_B.json`](file:///e:/Project_Q/processors/processor_B.json)) is deliberately designed to contrast severe routing bottlenecks against native planar subgraph embedding. In Processor A, the 1D linear chain imposes a nearest-neighbor constraint with an average node degree of only $1.6$ and an interaction diameter of $4$. Because the problem graph's non-anchored sub-topology contains non-local edges (e.g., interactions $(q_0, q_4)$ and $(q_1, q_3)$), executing two-qubit CX gates between distant qubits requires inserting SWAP gates into the compiled circuit. Each SWAP gate must be decomposed into 3 two-qubit CX operations (under the $\{rz, sx, x, cx\}$ basis gate set), creating severe routing overhead, substantial circuit dilation (depth increasing from 16 to 60), and accumulating depolarizing noise. Conversely, Processor B employs a 9-qubit heavy-hex-inspired lattice with an average node degree of $2.67$ and cyclic pathways containing central degree-5 hub nodes (qubits 2 and 6). This expanded connectivity allows the transpiler's layout pass to embed all 6 problem interactions directly onto adjacent hardware edges without inserting a single SWAP gate ($\text{SWAP count} = 0$), maintaining minimum two-qubit gate count ($12$ CXs) and reducing circuit depth to $35$ ($41.7\%$ reduction). **This comparison explicitly fulfills the Phase 1 benchmarking requirement. Custom Processor D was intentionally excluded from this stage as it is strictly reserved for Phase 2 co-design and architectural exploration.**

---

## Synthetic Noise Model

Implemented in [`src/processors.py`](file:///e:/Project_Q/src/processors.py#L52-L105), the noise model simulates NISQ hardware decoherence:
1. **1-Qubit Depolarizing Error**: $0.1\%$ ($p_1 = 0.001$) applied to single-qubit basis gates $\{rz, sx, x\}$.
2. **2-Qubit CX Depolarizing Error**: $0.5\%$ ($p_2 = 0.005$) applied to entangling CX gates.
3. **Readout Error**: $1.0\%$ ($p_{\text{ro}} = 0.01$) symmetric measurement error:
   $$\begin{pmatrix} P(0|0) & P(1|0) \\ P(0|1) & P(1|1) \end{pmatrix} = \begin{pmatrix} 0.99 & 0.01 \\ 0.01 & 0.99 \end{pmatrix}$$

---

## QAOA Routing & Transpilation Pipeline

Implemented in [`src/routing.py`](file:///e:/Project_Q/src/routing.py):
1. **Ansatz Synthesis**: `build_qaoa_circuit(cost_hamiltonian, p=1, gamma=0.3, beta=1.2)` prepares $|+\rangle^{\otimes 5}$, applies cost unitary $e^{-i \gamma H_C}$ (9 $RZ$ gates, 6 $ZZ$ interactions requiring 12 unrouted CXs), followed by mixer unitary $\prod RX(2 \beta)$, and final measurement.
2. **Compilation**: `transpile_qaoa_circuit(qc, backend, optimization_level=1, seed_transpiler=42)` maps virtual qubits to physical hardware under the target coupling map.

---

## Results & Benchmarking Metrics

All metrics were gathered from 4096-shot executions under the synthetic noise model and exported to [`results/tables/metrics.csv`](file:///e:/Project_Q/results/tables/metrics.csv):

| Metric | Baseline (Ideal Unrouted) | Processor A (5Q Linear Chain) | Processor B (9Q Heavy-Hex) | Performance Delta |
| :--- | :---: | :---: | :---: | :--- |
| **Physical Qubits** | 5 | 5 | 9 | +4 spectator/routing nodes |
| **Coupling Edges** | All-to-all virtual | 4 (linear) | 12 (heavy-hex) | $3\times$ edge density |
| **Circuit Depth** | 16 | **60** | **35** | **$41.7\%$ shallower** |
| **Total CX Count** | 12 | **18** | **12** | **$33.3\%$ fewer 2Q gates** |
| **SWAP Gate Count** | 0 | **2** | **0** | **$100\%$ SWAP elimination** |
| **Routing CX Overhead** | 0 | **+6 CX** | **+0 CX** | **Zero routing overhead** |
| **Expected Cut $\langle C \rangle$** | — | **10.8625** | **11.0037** | Higher solution fidelity |
| **Max-Cut Baseline** | 16.0 | 16.0 | 16.0 | Exact classical ground truth |
| **Approximation Ratio** | — | **$67.89\%$** | **$68.77\%$** | Improved under noise |

---

## Comprehensive Architectural Report

For an in-depth treatment of routing bottlenecks, cumulative two-qubit error scaling, NISQ trade-offs, and mathematical spectrum analysis, see the dedicated report:

 [`report.md`](file:///e:/Project_Q/report.md)

---

## Reproducibility & Verification

### Running the End-to-End Notebook
Open and execute [`main.ipynb`](file:///e:/Project_Q/main.ipynb) in Jupyter or VS Code.

### Automated Test Suite Execution
```powershell
$env:PYTHONPATH="."
& "e:\Project_Q\.venv\Scripts\python.exe" tests\test_problem.py
& "e:\Project_Q\.venv\Scripts\python.exe" tests\test_processors.py
& "e:\Project_Q\.venv\Scripts\python.exe" tests\test_routing_and_metrics.py
```

### Reproducing Benchmark Tables & Figures
```powershell
$env:PYTHONPATH="."
& "e:\Project_Q\.venv\Scripts\python.exe" scripts\run_notebook.py
```
Output tables and figures are stored in `results/tables/metrics.csv` and `results/figures/`.
