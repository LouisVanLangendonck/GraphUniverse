# Changelog

## [0.1.1] - 2025-02-18

### Changed
- Improved README with clearer installation instructions
- Highlighted interactive UI (`graph-universe-ui`) in documentation
- Restructured README for better PyPI presentation

### Fixed
- Clarified that `[viz]` extra is required for UI

## [0.1.0] - 18-02-2025

### Added
- Initial release of GraphUniverse
- Core graph generation functionality with controllable community structure
- `GraphUniverse` class for defining graph generation universes
- `GraphFamilyGenerator` for creating families of related graphs
- `GraphSample` class for individual graph instances
- `FeatureGenerator` for synthetic node features with configurable variance
- Support for power-law degree distributions
- Wide range of configurable graph properties: Homophily, average degree, power-law distribution, graph size, etc.
- PyTorch Geometric dataset integration
- Interactive Streamlit UI for graph exploration and generation
- Command-line interface via `graph-universe-ui`
- Comprehensive test suite
- Example scripts and documentation

### Features
- Generate graph families with consistent community semantics
- Fine-grained control over graph properties (homophily, degree distribution, community structure)
- Scalable generation
- Optional visualization tools and interactive UI (seaborn, streamlit)

[0.1.0]: https://github.com/LouisVanLangendonck/GraphUniverse/releases/tag/v0.1.0