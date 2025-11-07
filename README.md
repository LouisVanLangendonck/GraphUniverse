# GraphUniverse: Enabling Systematic Evaluation of Inductive Generalization

**Generate families of graphs with finely controllable properties for systematic evaluation of inductive graph learning models.**

[Quick Start](#quick-start) | [Documentation](#documentation) | [Interactive Demo](https://graphuniverse.streamlit.app/) | [Paper](link-to-paper)

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

## Quick Start

### Installation
```bash
pip install graphuniverse
```

### Basic Usage

#### Option 1: Via individual classes

```python
from graph_universe import GraphUniverse, GraphFamilyGenerator

# Create universe with detailed parameters
universe = GraphUniverse(K=8, edge_propensity_variance=0.3, feature_dim=10)

# Generate family with full parameter control
family = GraphFamilyGenerator(
    universe=universe,
    min_n_nodes=25, 
    max_n_nodes=50,
    min_communities=2,
    max_communities=7,
    homophily_range=(0.2, 0.8),
    avg_degree_range=(2.0, 10.0),
    degree_distribution="power_law",
    power_law_exponent_range=(2.0, 5.0),
    degree_separation_range=(0.1, 0.7),
    seed=42
)

# Generate graphs (stores in family.graphs)
family.generate_family(n_graphs=30, show_progress=True)

# Access generated graphs and convert to PyG format
print(f"Generated {len(family.graphs)} graphs!")
pyg_graphs = family.to_pyg_graphs(tasks=["community_detection"])
```

#### Option 2: Via YAML config file

```yaml
# configs/experiment.yaml
universe_parameters:
  K: 10
  edge_propensity_variance: 0.5
  feature_dim: 16
  center_variance: 1.0
  cluster_variance: 0.3
  seed: 42

family_parameters:
  n_graphs: 100
  min_n_nodes: 25
  max_n_nodes: 200
  min_communities: 3
  max_communities: 7
  homophily_range: [0.1, 0.9]
  avg_degree_range: [2.0, 8.0]
  degree_distribution: "power_law"
  power_law_exponent_range: [2.0, 3.0]
  degree_separation_range: [0.4, 0.8]
  seed: 42

tasks: ["community_detection", "triangle_counting"]
```

```python
# Use config-driven workflow
import yaml
from graph_universe import GraphUniverseDataset

with open("configs/experiment.yaml") as f:
    config = yaml.safe_load(f)

dataset = GraphUniverseDataset(root="./data", parameters=config)
print(f"Generated dataset with {len(dataset)} graphs!")
```

#### Option 3: Interactive Demo
Try GraphUniverse in your browser with real-time parameter tuning and direct dataset download:
**[https://graphuniverse.streamlit.app/](https://graphuniverse.streamlit.app/)**

### Validation & Quality

GraphUniverse includes comprehensive metrics to validate property realization and quantify learnable community signals:

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



