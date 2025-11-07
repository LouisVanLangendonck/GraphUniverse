# GraphUniverse: Enabling Systematic Evaluation of Inductive Generalization

**Generate families of graphs with finely controllable properties for systematic evaluation of inductive graph learning models.**

[Quick Start](#quick-start) | [Documentation](#documentation) | [Interactive Demo](https://graphuniverse.streamlit.app/) | [Paper](link-to-paper)

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

### Basic Usage

#### Option 1: Via individual classes

```python
from graph_universe import GraphUniverse, GraphFamilyGenerator

# Create universe
universe = GraphUniverse(K=5, edge_propensity_variance=0.3, feature_dim=10)

# Generate family  
family = GraphFamilyGenerator(
    universe=universe,
    min_n_nodes=50, 
    max_n_nodes=150,
    homophily_range=(0.2, 0.8)
)

graphs = family.generate_family(
    n_graphs=100,
)

# Convert to PyG graphs, ready for training on community detection task
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
  n_graphs: 1000
  min_n_nodes: 50
  max_n_nodes: 200
  min_communities: 3
  max_communities: 7
  homophily_range: [0.1, 0.9]
  avg_degree_range: [2.0, 8.0]
  degree_distribution: "power_law"
  power_law_exponent_range: [2.0, 3.0]
  degree_separation_range: [0.4, 0.8]
  seed: 42
```

```python
# Use config-driven workflow
import yaml

with open("configs/experiment.yaml") as f:
    config = yaml.safe_load(f)

dataset = GraphUniverseDataset(root="./data", parameters=config)
```

#### Option 3: Via interactive Demo in Browser
Try GraphUniverse in your browser and directly download dataset ready-to-train-on (Pytorch Geometric format):
**[https://graphuniverse.streamlit.app/](https://graphuniverse.streamlit.app/)**


### Validation & Quality

GraphUniverse includes comprehensive metrics to validate desired property realization and quantify the learnable community-related signals both within graphs ("graph family signals") and between graphs ("graph family consistency"). 'Community-related' metrics since we encourage eventual graph learning tasks to rely on community-detection or a derivation of it, since all learnable info (diversity in features, graph structure and node degrees) are stored in those latent 'communities'. 

```python
# Validate Standard Graph Property Generation
family_properties = family.analyze_graph_family_properties()
for property in ['node_counts', 'avg_degrees', 'homophily_levels', 'mean_edge_probability_deviation']: # More Calculateable Properties Available
    print(family_properties[property])

# Calculate Within-graph Community-related Signals
family_signals = family.analyze_graph_family_signals()
for signal_metric in ['structure_signal', 'feature_signal', 'degree_signal']:
    print(family_signals[signal_metric])

# Calculate Between-graph Community-related Consistency
family_consistency = family.analyze_graph_family_consistency()
for consistency_metric in ['structure_consistency', 'feature_consistency', 'degree_consistency']:
    print(family_consistency[consistency_metric])
```



