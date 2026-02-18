# GraphUniverse: Enabling Systematic Evaluation of Inductive Generalization

[![PyPI](https://img.shields.io/pypi/v/graph-universe)](https://pypi.org/project/graph-universe/)
[![Python](https://img.shields.io/pypi/pyversions/graph-universe)](https://pypi.org/project/graph-universe/)
[![License](https://img.shields.io/pypi/l/graph-universe)](https://github.com/LouisVanLangendonck/GraphUniverse/blob/main/LICENSE)
[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue)]()

**Generate families of graphs with finely controllable properties for systematic evaluation of inductive graph learning models.**

[Quick Start](#quick-start) | [Interactive UI](#interactive-ui) | [Validation](#validation--analysis) | [Paper Experiments](#for-researchers--contributors)

![Example Graph Family][graphplot]

[graphplot]: https://github.com/LouisVanLangendonck/GraphUniverse/blob/main/assets/ExampleGraphFamily.png "Example Graph Family Visualization"

## Key Features

Current graph learning benchmarks are limited to **single-graph, transductive settings**. GraphUniverse enables the first systematic evaluation of **inductive generalization** by generating entire families of graphs with:

- **Consistent Semantics**: Communities maintain stable identities across graphs
- **Fine-grained Control**: Tune homophily, degree distributions, community structure
- **Scalable Generation**: Linear scaling, thousands of graphs per minute  
- **Validated Framework**: Comprehensive parameter sensitivity analysis
- **Interactive Tools**: Web-based exploration and visualization

![GraphUniverse Methodology Graphical Overview][logo]

[logo]: https://github.com/LouisVanLangendonck/GraphUniverse/blob/main/assets/GraphUniverseMethodologyClean.png "Methodology Overview"

---

## Installation

Install from PyPI:
```bash
pip install graph-universe
```

**For the interactive UI and visualization tools:**
```bash
pip install graph-universe[viz]
```

**Optional extras:**
- `[viz]` - Streamlit UI + seaborn visualization tools
- `[dev]` - Development dependencies (testing, linting)
- `[notebook]` - Jupyter notebook support
- `[all]` - Everything (includes documentation tools)

**Install from source (for contributors):**
```bash
git clone https://github.com/LouisVanLangendonck/GraphUniverse.git
cd GraphUniverse
pip install -e ".[dev]"
```

---

## Interactive UI

After installing with `[viz]`, launch the interactive dashboard:
```bash
graph-universe-ui
```

**This opens a browser-based UI where you can:**
- 🎛️ Configure all generation parameters with real-time preview
- 📊 Generate graph families and visualize their properties
- 🔍 Analyze community structure, homophily, and degree distributions
- 💾 Download datasets directly in PyTorch Geometric format

**No coding required!** Perfect for exploration and quick dataset generation.

**Hosted demo:** Try it online at [graphuniverse.streamlit.app](https://graphuniverse.streamlit.app/)

**Launch from Python:**
```python
from graph_universe import launch_ui
launch_ui()  # Opens browser, press Ctrl+C to stop
```

---

## Quick Start

### Option 1: Python API with Individual Classes
```python
from graph_universe import GraphUniverse, GraphFamilyGenerator

# Create universe with 8 communities and 10-dimensional features
universe = GraphUniverse(K=8, edge_propensity_variance=0.3, feature_dim=10)

# Generate family with full parameter control
family = GraphFamilyGenerator(
    universe=universe,
    n_nodes_range=(35, 50),
    n_communities_range=(2, 6),
    homophily_range=(0.2, 0.8),
    avg_degree_range=(2.0, 10.0),
    power_law_exponent_range=(2.0, 5.0),
    degree_separation_range=(0.1, 0.7),
    seed=42
)

# Generate 30 graphs
family.generate_family(n_graphs=30, show_progress=True)

print(f"Generated {len(family.graphs)} graphs!")

# Convert to PyTorch Geometric format for training
pyg_graphs = family.to_pyg_graphs(task="community_detection")
```

### Option 2: Config-Driven Workflow

Create `config.yaml`:
```yaml
universe_parameters:
  K: 10
  edge_propensity_variance: 0.5
  feature_dim: 16
  center_variance: 1.0
  cluster_variance: 0.3
  seed: 42

family_parameters:
  n_graphs: 100
  n_nodes_range: [25, 200]
  n_communities_range: [3, 7]
  homophily_range: [0.1, 0.9]
  avg_degree_range: [2.0, 8.0]
  power_law_exponent_range: [2.0, 3.0]
  degree_separation_range: [0.4, 0.8]
  seed: 42

task: "community_detection"
```

Then load and generate:
```python
import yaml
from graph_universe import GraphUniverseDataset

with open("config.yaml") as f:
    config = yaml.safe_load(f)

dataset = GraphUniverseDataset(root="./data", parameters=config)
print(f"Generated dataset with {len(dataset)} graphs!")
```

---

## Validation & Analysis

GraphUniverse includes built-in validation to ensure generated graphs match target properties:
```python
# Validate standard graph properties
family_properties = family.analyze_graph_family_properties()
for property_name in ['node_counts', 'avg_degrees', 'homophily_levels']:
    values = family_properties[property_name]
    print(f"{property_name}: mean={np.mean(values):.3f}")

# Analyze within-graph community signals (fits Random Forest per graph)
family_signals = family.analyze_graph_family_signals()
for signal in ['structure_signal', 'feature_signal', 'degree_signal']:
    values = family_signals[signal]
    print(f"{signal}: mean={np.mean(values):.3f}")

# Measure between-graph consistency
family_consistency = family.analyze_graph_family_consistency()
for metric in ['structure_consistency', 'feature_consistency', 'degree_consistency']:
    value = family_consistency[metric]
    print(f"{metric}: {value:.3f}")
```

---

## Documentation & Support

- **GitHub Repository**: https://github.com/LouisVanLangendonck/GraphUniverse
- **PyPI Package**: https://pypi.org/project/graph-universe/
- **Issue Tracker**: https://github.com/LouisVanLangendonck/GraphUniverse/issues
- **Changelog**: https://github.com/LouisVanLangendonck/GraphUniverse/blob/main/CHANGELOG.md

---

## Citation

If you use GraphUniverse in your research, please cite:
```bibtex
@inproceedings{vanlangendonck2026graphuniverse,
  title={GraphUniverse: Enabling Systematic Evaluation of Inductive Generalization},
  author={Van Langendonck, Louis and Bernardez, Guillermo},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```

---

## For Researchers & Contributors

The sections below contain resources for reproducing paper experiments and contributing to development.

### Reproducing Paper Experiments

Clone the repository to access validation and experiment scripts:
```bash
git clone https://github.com/LouisVanLangendonck/GraphUniverse.git
cd GraphUniverse
pip install -e ".[dev]"
```

**Run parameter sensitivity validation (reproduces paper results):**
```bash
python experiments/validate_parameter_sensitivity.py --n-random-samples 100 --n-graphs 30
```

**Run scalability experiments:**
```bash
python experiments/scalability_experiment.py
```

### Example Gallery & Additional Resources

Visit the [GitHub repository](https://github.com/LouisVanLangendonck/GraphUniverse) for:
- Complete example scripts in `examples/` directory
- Detailed methodology diagrams and visualizations
- Additional documentation and tutorials

### Contributing

We welcome contributions! To get started:

1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest test/`
5. Submit a pull request

See the repository for detailed contribution guidelines.

---

## License

MIT License - see [LICENSE](https://github.com/LouisVanLangendonck/GraphUniverse/blob/main/LICENSE) for details.

Copyright (c) 2025 Louis Van Langendonck and Guillermo Bernardez