# Bitcoin OTC Trust Network — Comprehensive Analysis

A Python script for in-depth graph analysis of the **Bitcoin OTC Trust Network** dataset, covering structural metrics, centrality, flow algorithms, bipartite representations, sentiment analysis, and rich visualizations.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [Usage](#usage)
- [Analysis Modules](#analysis-modules)
- [Output Files](#output-files)
---

## Overview

This script performs a comprehensive network analysis of the Bitcoin OTC signed trust graph — a real-world directed, weighted network where users rate each other on a scale from -10 (total distrust) to +10 (total trust).

The analysis covers:

1. Basic network statistics (density, connectivity, degree distributions)
2. Hub detection and clique analysis
3. Centrality measures (Degree, PageRank)
4. Eulerian graph checks
5. Maximum flow computation (Ford-Fulkerson)
6. Bipartite graph representation
7. Sentiment analysis of ratings
8. Adjacency and incidence matrix representations
9. Network visualizations (5 PNG files)
10. Additional metrics (assortativity, reciprocity, k-core, bridges)

---

## Requirements

- Python 3.8+
- `networkx`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`

---

## Installation

```bash
pip install networkx pandas numpy matplotlib seaborn
```

---

## Dataset

The script uses the **Bitcoin OTC Trust Weighted Signed Network** from SNAP (Stanford Network Analysis Project).

- **Download:** [https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html)
- **File format:** `.csv.gz` (compressed CSV)
- **Columns:** `source`, `target`, `rating`, `timestamp`
- **Rating scale:** -10 (total distrust) to +10 (total trust)

---

## Configuration

At the top of the script, set the paths for your environment:

```python
INPUT_FILE = r'path/to/soc-sign-bitcoinotc.csv.gz'
OUTPUT_DIR = r'path/to/output/directory/'
```

Make sure `OUTPUT_DIR` exists before running, or create it:

```bash
mkdir -p path/to/output/directory
```

---

## Usage

```bash
python bitcoin_otc_analysis.py
```

The script will run all analysis modules sequentially and print results to the console, then save five visualization files to `OUTPUT_DIR`.

---

## Analysis Modules

### 1. Data Loading
Reads the gzip-compressed CSV and builds a pandas DataFrame with edge data.

### 2. Graph Construction
Builds a directed `networkx.DiGraph` with edge weights (ratings) and timestamps.

### 3. Basic Statistics
Reports connectivity (strong/weak), number of connected components, and average in/out degrees.

### 4. Hubs & Cliques
- Identifies the **top 10 hubs** by total degree.
- Finds **maximum cliques** in the undirected projection (uses a 500-node sample for large graphs).
- Computes the **average clustering coefficient**.

### 5. Centrality Analysis
- **Degree Centrality** — top 5 nodes
- **PageRank** — top 5 nodes

### 6. Eulerian Graph Check
Tests both the directed and undirected graphs for Eulerian cycles and paths. Reports the number of odd-degree vertices if neither condition is met.

### 7. Maximum Flow (Ford-Fulkerson)
Builds a capacity graph using only positive ratings, then computes maximum flow between three selected pairs of top hubs. Reports flow values and the five heaviest flow edges per pair.

### 8. Bipartite Graph
Constructs a bipartite graph separating **raters** (left partition) from **rated users** (right partition) using only positive-rating edges.

### 9. Sentiment Analysis
Breaks down the rating distribution into positive, negative, and neutral counts, and prints a text-based histogram.

### 10. Matrix Representations
Generates the **adjacency matrix** and **incidence matrix** for a 100-node subgraph and prints a 5×5 preview.

### 11. Additional Analyses
| Metric | Description |
|---|---|
| Assortativity | Whether high-degree nodes connect to other high-degree nodes |
| Reciprocity | Fraction of edges with a reverse edge |
| Diameter & Avg. path length | Computed on a 300-node sample |
| K-core decomposition | Maximum core number and size of the main k-core |
| Bridges | Critical edges whose removal disconnects the graph |
| Directionality | Breakdown of nodes by in-only / out-only / bidirectional |

---

## Output Files

All files are saved to `OUTPUT_DIR`:

| File | Description |
|---|---|
| `viz1_distributions.png` | In/out degree distributions, rating histogram, top-20 hubs bar chart |
| `viz2_hub_network.png` | Spring-layout graph of top hubs and their neighbors (green = positive, red = negative edges) |
| `viz3_adjacency_matrix.png` | Heatmap of the 100-node adjacency matrix |
| `viz4_bipartite_graph.png` | Bipartite layout: raters (blue) vs rated users (orange) |
| `viz5_bipartite_stats.png` | Degree distributions and top-15 active nodes for the bipartite graph |

---

## Notes

- Clique detection and diameter computation are approximated on subgraphs for performance on large inputs.
- Maximum flow uses only positive-weight edges as capacities; negative ratings are excluded.
- The bipartite graph uses only positive-rating interactions.
