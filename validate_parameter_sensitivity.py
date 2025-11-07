"""
Script to analyze how individual parameters affect signal and consistency metrics.
Tests each parameter with narrow ranges/values while randomizing all others.
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

from graph_universe.graph_family import GraphFamilyGenerator
from graph_universe.graph_universe import GraphUniverse

# Parameters for random analysis
PARAMS_OF_INTEREST = [
    "edge_propensity_variance",
    "cluster_variance",
    "homophily_range",
    "avg_degree_range",
    "min_n_nodes",
    "min_communities",
    "degree_separation_range",
    "power_law_exponent_range",
]

# Fixed settings
GRAPHS_PER_FAMILY = 30
UNIVERSE_K = 15

# All parameters that can be varied
ALL_VARIABLE_PARAMS = {
    # Universe parameters
    "edge_propensity_variance": {
        "type": "continuous",
        "test_values": [0.0, 0.10, 0.20, 0.30, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "random_range": (0.0, 1.0),
        "level": "universe",
    },
    "feature_dim": {
        "type": "discrete",
        "test_values": [10, 50, 100],
        "random_range": (10, 100),
        "level": "universe",
    },
    "center_variance": {
        "type": "continuous",
        "test_values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "random_range": (0.1, 1.0),
        "level": "universe",
    },
    "cluster_variance": {
        "type": "continuous",
        "test_values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "random_range": (0.1, 1.0),
        "level": "universe",
    },
    # Family generator parameters
    "min_n_nodes": {
        "type": "discrete",
        "test_values": [50, 100, 250, 500, 750],
        "random_range": (50, 400),
        "level": "family",
        "paired_with": "max_n_nodes",
    },
    "max_n_nodes": {
        "type": "discrete",
        "test_values": [100, 200, 400, 800],
        "random_range": (100, 1000),
        "level": "family",
        "paired_with": "min_n_nodes",
    },
    "min_communities": {
        "type": "discrete",
        "test_values": [2, 4, 6, 10, 15],
        "random_range": (2, 15),
        "level": "family",
        "paired_with": "max_communities",
    },
    "max_communities": {
        "type": "discrete",
        "test_values": [4, 6, 8],
        "random_range": (4, 8),
        "level": "family",
        "paired_with": "min_communities",
        "max_value": UNIVERSE_K,
    },
    "homophily_range": {
        "type": "range",
        "test_values": [(0.0, 0.05), (0.2, 0.25), (0.4, 0.45), (0.6, 0.65), (0.8, 0.85)],
        "random_range": (0.0, 1.0),
        "level": "family",
    },
    "avg_degree_range": {
        "type": "range",
        "test_values": [(2.0, 2.5), (4.0, 4.5), (6.0, 6.5), (8.0, 8.5), (10.0, 10.5)],
        "random_range": (2.0, 20.0),
        "level": "family",
    },
    "degree_separation_range": {
        "type": "range",
        "test_values": [
            (0.0, 0.05),
            (0.1, 0.15),
            (0.2, 0.25),
            (0.4, 0.45),
            (0.6, 0.65),
            (0.8, 0.85),
            (0.9, 0.95),
        ],
        "random_range": (0.0, 1.0),
        "level": "family",
    },
    "degree_distribution": {
        "type": "categorical",
        "test_values": ["power_law"],
        "random_range": ["power_law"],
        "level": "family",
    },
    "power_law_exponent_range": {
        "type": "range",
        "test_values": [(1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (4.0, 4.5), (4.5, 5.0)],
        "random_range": (1.5, 4.5),
        "level": "family",
    },
}

# Dictionary for renaming parameters and metrics in the heatmap
HEATMAP_DISPLAY_NAMES = {
    # Parameters
    "edge_propensity_variance": r"Edge Propensity Variance $\epsilon$",
    "cluster_variance": r"Cluster Variance $\sigma^2$",
    "min_n_nodes": r"Mean Node Count $\bar{n}$",
    "min_communities": r"Mean Communities Participating $\bar{k}$",
    "homophily_range": r"Mean Homophily $\bar{h}$",
    "avg_degree_range": r"Mean Average Degree $\bar{d}$",
    "degree_separation_range": r"Mean Degree Separation $\bar{\rho}$",
    "power_law_exponent_range": r"Mean Power Law Exponent $\bar{\alpha}$",
    # Signal Metrics
    "feature_signal": "Feature Signal",
    "degree_signal": "Degree Signal",
    "structure_signal": "Structure Signal",
    # Consistency Metrics
    "feature_consistency": "Feature Consistency",
    "degree_consistency": "Degree Consistency",
    "structure_consistency": "Structure Consistency",
    # Main Property Metrics
    "homophily_levels": "Mean Homophily Levels",
    "avg_degrees": "Mean Average Degree",
    "tail_ratio_99": "Mean Tail Ratio 99",
    # Technical Metrics
    "mean_edge_probability_deviation": "Mean Edge Probability Deviation",
    "graph_generation_times": "Mean Graph Generation Time",
}

# Metrics to calculate
SIGNAL_METRICS = ["feature_signal", "degree_signal", "structure_signal"]
CONSISTENCY_METRICS = [
    "feature_consistency",
    "degree_consistency",
    "structure_consistency",
]
MAIN_PROPERTY_METRICS = ["homophily_levels", "avg_degrees", "tail_ratio_99"]
TECHNICAL_METRICS = ["mean_edge_probability_deviation", "graph_generation_times"]
PROPERTY_METRICS = MAIN_PROPERTY_METRICS + TECHNICAL_METRICS

def generate_random_baseline_params(n_samples=100, seed=None):
    """
    Generate n_samples random parameter combinations from the allowed ranges.
    We don't fix any specific parameter.

    Args:
        n_samples: Number of random samples to generate
        seed: Random seed for reproducibility

    Returns:
        list: List of parameter dictionaries
    """
    if seed is not None:
        np.random.seed(seed)

    all_samples = []

    for sample_idx in range(n_samples):
        # Generate a new seed for each sample to ensure independence
        sample_seed = abs(hash(f"random_baseline_{sample_idx}")) % (2**32)
        np.random.seed(sample_seed)

        params = {"universe": {}, "family": {}}

        # Generate random universe parameters (only for parameters in PARAMS_OF_INTEREST)
        for param_name, param_config in ALL_VARIABLE_PARAMS.items():
            if param_config["level"] == "universe" and param_name in PARAMS_OF_INTEREST:
                if param_config["type"] == "continuous":
                    min_val, max_val = param_config["random_range"]
                    params["universe"][param_name] = np.random.uniform(min_val, max_val)
                elif param_config["type"] == "discrete":
                    params["universe"][param_name] = int(
                        np.random.uniform(
                            param_config["random_range"][0], param_config["random_range"][1]
                        )
                    )
                elif param_config["type"] == "categorical":
                    params["universe"][param_name] = np.random.choice(param_config["random_range"])
                elif param_config["type"] == "range":
                    min_val, max_val = param_config["random_range"]
                    range_size = np.random.uniform(0.05, 0.2)  # Random range size
                    start_val = np.random.uniform(min_val, max_val - range_size)
                    end_val = start_val + range_size
                    params["universe"][param_name] = (start_val, end_val)
            elif param_config["level"] == "universe" and param_name not in PARAMS_OF_INTEREST:
                # For excluded parameters, use reasonable default values
                if param_name == "feature_dim":
                    params["universe"][param_name] = 15  # Default feature dimension
                elif param_name == "center_variance":
                    params["universe"][param_name] = 0.2  # Default center variance

        # Generate random family parameters (only for parameters in PARAMS_OF_INTEREST)
        for param_name, param_config in ALL_VARIABLE_PARAMS.items():
            if param_config["level"] == "family" and param_name in PARAMS_OF_INTEREST:
                if param_config["type"] == "continuous":
                    min_val, max_val = param_config["random_range"]
                    params["family"][param_name] = np.random.uniform(min_val, max_val)
                elif param_config["type"] == "discrete":
                    params["family"][param_name] = int(
                        np.random.uniform(
                            param_config["random_range"][0], param_config["random_range"][1]
                        )
                    )
                elif param_config["type"] == "categorical":
                    params["family"][param_name] = np.random.choice(param_config["random_range"])
                elif param_config["type"] == "range":
                    min_val, max_val = param_config["random_range"]
                    range_size = np.random.uniform(0.05, 0.2)  # Random range size
                    start_val = np.random.uniform(min_val, max_val - range_size)
                    end_val = start_val + range_size
                    params["family"][param_name] = (start_val, end_val)

        # Handle paired parameters
        for param_name, param_config in ALL_VARIABLE_PARAMS.items():
            if "paired_with" in param_config:
                paired_param = param_config["paired_with"]

                if param_name == "min_n_nodes" and param_name in params["family"]:
                    # Ensure max > min
                    params["family"][paired_param] = params["family"][param_name] + np.random.uniform(50, 200)
                elif param_name == "min_communities" and param_name in params["family"]:
                    # Ensure max > min
                    params["family"][paired_param] = min(
                        params["family"][param_name] + np.random.randint(1, 4), UNIVERSE_K
                    )

        # Always set degree_distribution to "power_law"
        params["family"]["degree_distribution"] = "power_law"

        all_samples.append(params)

    return all_samples


def run_random_baseline_analysis(
    n_samples=100, n_repeats_per_sample=3, output_dir="parameter_analysis_results"
):
    """
    Run random baseline analysis by generating n_samples random parameter combinations
    and running each n_repeats_per_sample times.

    Args:
        n_samples: Number of random samples to generate
        n_repeats_per_sample: Number of repeats per random sample
        output_dir: Directory to save results

    Returns:
        dict: Dictionary containing random baseline results
    """
    os.makedirs(output_dir, exist_ok=True)

    print(
        f"\nRunning random baseline analysis with {n_samples} samples, {n_repeats_per_sample} repeats each"
    )

    # Generate random parameter samples
    random_param_samples = generate_random_baseline_params(n_samples=n_samples)

    # Initialize result structure
    random_results = {
        "random_samples": random_param_samples,
        "signal_metrics": {metric: [] for metric in SIGNAL_METRICS},
        "consistency_metrics": {metric: [] for metric in CONSISTENCY_METRICS},
        "property_metrics": {metric: [] for metric in PROPERTY_METRICS},
        "metadata": [],
    }

    # Run analysis for each sample
    successful_samples = []
    successful_sample_indices = []

    for sample_idx, sample_params in enumerate(
        tqdm(random_param_samples, desc="Processing random samples")
    ):
        print(f"\nProcessing sample {sample_idx + 1}/{n_samples}")

        sample_signals = {metric: [] for metric in SIGNAL_METRICS}
        sample_consistency = {metric: [] for metric in CONSISTENCY_METRICS}
        sample_properties = {metric: [] for metric in PROPERTY_METRICS}
        sample_metadata = []
        sample_successful = False

        for repeat in range(n_repeats_per_sample):
            seed = abs(hash(f"random_baseline_{sample_idx}_{repeat}")) % (2**32)

            try:
                # Create universe with random params
                universe_params = sample_params["universe"]
                universe = GraphUniverse(
                    K=UNIVERSE_K,
                    edge_propensity_variance=universe_params["edge_propensity_variance"],
                    feature_dim=universe_params["feature_dim"],
                    center_variance=universe_params["center_variance"],
                    cluster_variance=universe_params["cluster_variance"],
                    seed=seed,
                )

                # Create family generator with random params
                family_params = sample_params["family"]
                generator = GraphFamilyGenerator(
                    universe=universe,
                    min_n_nodes=family_params["min_n_nodes"],
                    max_n_nodes=family_params["max_n_nodes"],
                    min_communities=family_params["min_communities"],
                    max_communities=family_params["max_communities"],
                    homophily_range=family_params["homophily_range"],
                    avg_degree_range=family_params["avg_degree_range"],
                    degree_separation_range=family_params["degree_separation_range"],
                    degree_distribution=family_params["degree_distribution"],
                    power_law_exponent_range=family_params["power_law_exponent_range"],
                    seed=seed,
                )

                # Generate family
                generator.generate_family(n_graphs=GRAPHS_PER_FAMILY, show_progress=False)

                # Calculate signals
                signals = generator.analyze_graph_family_signals()
                for metric in SIGNAL_METRICS:
                    if metric in signals:
                        metric_values = [v for v in signals[metric] if v is not None]
                        if metric_values:
                            sample_signals[metric].append(metric_values)

                # Calculate consistency
                consistency = generator.analyze_graph_family_consistency()
                for metric in CONSISTENCY_METRICS:
                    if metric in consistency:
                        if isinstance(consistency[metric], list):
                            if consistency[metric]:
                                sample_consistency[metric].append(consistency[metric])
                        else:
                            sample_consistency[metric].append([consistency[metric]])

                # Calculate properties
                properties = generator.analyze_graph_family_properties()
                for metric in PROPERTY_METRICS:
                    if metric in properties:
                        if isinstance(properties[metric], list):
                            if properties[metric]:
                                sample_properties[metric].append(properties[metric])
                        else:
                            sample_properties[metric].append([properties[metric]])

                # Store metadata
                sample_metadata.append(
                    {
                        "sample_idx": sample_idx,
                        "repeat": repeat,
                        "seed": seed,
                        "params": sample_params.copy(),
                    }
                )

                sample_successful = True

            except Exception as e:
                print(f"Error processing sample {sample_idx}, repeat {repeat}: {e}")
                continue

        # Only store results if at least one repeat was successful
        if sample_successful:
            successful_samples.append(sample_params)
            successful_sample_indices.append(sample_idx)

            # Store results for this sample
            for metric in SIGNAL_METRICS:
                random_results["signal_metrics"][metric].append(sample_signals[metric])
            for metric in CONSISTENCY_METRICS:
                random_results["consistency_metrics"][metric].append(sample_consistency[metric])
            for metric in PROPERTY_METRICS:
                random_results["property_metrics"][metric].append(sample_properties[metric])
            random_results["metadata"].extend(sample_metadata)
        else:
            print(f"Sample {sample_idx} failed completely - skipping")

    # Update results structure with successful samples info
    random_results["random_samples"] = successful_samples
    random_results["successful_sample_indices"] = successful_sample_indices
    random_results["n_successful_samples"] = len(successful_samples)
    random_results["n_total_samples"] = n_samples

    # Save results
    output_file = os.path.join(output_dir, "random_baseline_analysis.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(random_results, f)

    print(
        f"Random baseline analysis completed. Successfully processed {len(successful_samples)}/{n_samples} samples."
    )
    print(f"Results saved to {output_file}")
    return random_results


def is_invalid_value(value):
    """
    Check if a value is invalid (None, NaN, or inf)

    Args:
        value: The value to check

    Returns:
        bool: True if the value is invalid, False otherwise
    """
    if value is None:
        return True

    try:
        return np.isnan(value) or np.isinf(value)
    except (TypeError, ValueError):
        # For values that can't be checked with numpy functions
        return False


def apply_publication_style(axis):
    axis.tick_params(axis="both", which="major", labelsize=11, length=6, width=1.2)
    axis.tick_params(axis="both", which="minor", labelsize=9, length=3, width=1.0)
    for spine in axis.spines.values():
        spine.set_linewidth(1.2)
    axis.grid(True, alpha=0.25, linewidth=0.8)
    axis.set_facecolor("white")
    axis.margins(x=0.05)


def create_wide_boxplot(axis, box_data, positions, widths=0.65, show_fliers=False):
    """Create a wider, cleaner boxplot with consistent styling."""
    bp = axis.boxplot(
        box_data,
        positions=positions,
        widths=widths,
        patch_artist=True,
        showfliers=show_fliers,
        boxprops={"linewidth": 1.4},
        whiskerprops={"linewidth": 1.4},
        capprops={"linewidth": 1.4},
        medianprops={"linewidth": 2.0},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
    )
    # Tighten x-limits to make boxplots fill more of the axis
    if positions:
        axis.set_xlim(min(positions) - 0.6, max(positions) + 0.6)
    return bp


def create_categorical_boxplot_with_tests(
    ax, box_data, box_positions, x_labels, colors, annotation_color="lightgray"
):
    """
    Create categorical boxplot with statistical testing using classical legend approach.

    Args:
        ax: matplotlib axis
        box_data: list of data arrays for each group
        box_positions: list of x positions for boxes
        x_labels: list of x-axis labels
        colors: dict with keys 'face', 'median', 'whiskers' for styling
        annotation_color: color for annotation boxes (default: 'lightgray' for baseline, 'lightcoral' for randomized)
    """
    if not box_data:
        return

    # Create boxplot
    bp = create_wide_boxplot(ax, box_data, box_positions, widths=0.7, show_fliers=False)

    # Color the boxes
    for patch in bp["boxes"]:
        patch.set_facecolor(colors["face"])
        patch.set_alpha(0.7)

    # Style the median line
    for median in bp["medians"]:
        median.set_color(colors["median"])
        median.set_linewidth(2)

    # Style the whiskers and caps
    for element in ["whiskers", "caps"]:
        for line in bp[element]:
            line.set_color(colors["whiskers"])
            line.set_linewidth(1)

    # Style the fliers (outliers)
    for flier in bp["fliers"]:
        flier.set_marker("o")
        flier.set_markerfacecolor(colors["median"])

    # Set x-ticks and labels
    ax.set_xticks(box_positions)
    ax.set_xticklabels(x_labels)

    # Add horizontal grid lines for easier reading
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)  # Put grid lines behind data points

def calculate_correlation_with_significance(x, y, param_type="continuous", family_ids=None):
    """
    Calculate correlation with direction and statistical significance.
    Always uses family means when family_ids are provided.

    Args:
        x: Independent variable values
        y: Dependent variable values
        param_type: Type of parameter ('continuous', 'categorical', 'boolean', 'range')
        family_ids: Array of family identifiers for family mean calculation

    Returns:
        For continuous: (correlation_value, direction, significance, ci_lower, ci_upper)
        For categorical: (correlation_value, direction, significance, ci_lower, ci_upper, variance_test_name, variance_p_value, variance_significance)
    """
    # Remove NaN values first
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    # Check for sufficient data and non-constant arrays after removing NaN
    if len(x) < 3:
        return 0.0, "none", "ns", 0.0, 0.0

    # Check if arrays are constant (all values the same) after removing NaN
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0, "none", "ns", 0.0, 0.0

    # Calculate Pearson correlation coefficient
    correlation, p_value = pearsonr(x, y)

    # Check if correlation is nan
    if np.isnan(correlation):
        return 0.0, "none", "ns", 0.0, 0.0

    # No confidence intervals needed for plotting
    ci_lower, ci_upper = 0.0, 0.0

    # Determine direction
    if correlation > 0:
        direction = "positive"
    elif correlation < 0:
        direction = "negative"
    else:
        direction = "none"

    # Determine significance based on p-value
    if p_value < 0.001:
        significance = "***"
    elif p_value < 0.01:
        significance = "**"
    elif p_value < 0.05:
        significance = "*"
    else:
        significance = "ns"

    return correlation, direction, significance, ci_lower, ci_upper

def format_correlation_display(
    correlation_value, direction, significance, ci_lower, ci_upper, add_confidence_interval=False
):
    """
    Format correlation value for display in heatmap.

    Args:
        correlation_value: Correlation coefficient, Cohen's d, or eta-squared value
        direction: 'positive', 'negative', or 'none'
        significance: '***', '**', '*', or 'ns'

    Returns:
        str: Formatted string for display
    """
    if significance == "ns":
        return "NS"

    # Format the correlation value
    if abs(correlation_value) < 0.01:
        formatted_value = "<0.01"
    else:
        formatted_value = f"{correlation_value:.2f}"

    # Add direction indicator
    if add_confidence_interval:
        if direction == "positive":
            return f"{formatted_value} ± {abs(ci_upper - correlation_value):.2f} ({significance})"
        elif direction == "negative":
            return f"-{formatted_value} ± {abs(correlation_value - ci_lower):.2f} ({significance})"
        else:
            return f"{formatted_value} ± {abs(ci_upper - correlation_value):.2f} ({significance})"
    else:
        return f"{formatted_value} ({significance})"

def get_plot_param_name(param_name):
    """
    Get the display name for a parameter in plots.
    Removes underscores and capitalizes words, with special handling for specific parameters.

    Args:
        param_name: Original parameter name

    Returns:
        str: Formatted parameter name for plotting
    """
    # Special handling for specific parameters
    plot_param_name_dict = {
        "degree_center_method": "Use Degree Community Coupling",
        "use_dccc_sbm": "Use DCCC SBM",
        "edge_propensity_variance": "Edge Propensity Variance",
    }
    return plot_param_name_dict.get(param_name, param_name.replace("_", " ").title())

def create_summary_heatmap(
    results_dict, save_path="parameter_sensitivity_heatmap.png", display_names=HEATMAP_DISPLAY_NAMES
):
    """
    Create split heatmaps showing correlation values for continuous parameters and significance levels for categorical parameters.
    Shows statistical significance levels: * (p < 0.05), ** (p < 0.01), and *** (p < 0.001).

    Args:
        results_dict: Dictionary containing results
        save_path: Path to save the heatmap image
        display_names: Dictionary mapping parameter and metric names to display names (e.g., LaTeX formatted)
    """
    params = list(results_dict.keys())
    # Reorder metrics: Property metrics first, then signal metrics, then consistency metrics
    metrics = MAIN_PROPERTY_METRICS + TECHNICAL_METRICS + SIGNAL_METRICS + CONSISTENCY_METRICS

    # Separate continuous and categorical parameters
    continuous_params = params
    categorical_params = []

    # Create matrices for both parameter types
    continuous_correlation_matrix = np.zeros((len(continuous_params), len(metrics)))
    continuous_significance_matrix = np.empty((len(continuous_params), len(metrics)), dtype=object)

    np.zeros(
        (len(categorical_params), len(metrics))
    )  # Numeric for colormap
    np.empty((len(categorical_params), len(metrics)), dtype=object)

    # Process continuous parameters
    for i, param_name in enumerate(continuous_params):
        param_results = results_dict[param_name]
        test_values = param_results["test_values"]
        param_config = ALL_VARIABLE_PARAMS[param_name]

        # Get x values using helper function
        x_values = test_values
        if isinstance(test_values[0], tuple) and len(test_values[0]) == 2:
            x_values = [(v[0] + v[1]) / 2 for v in test_values]

        # Calculate correlation for each metric
        for j, metric in enumerate(metrics):
            if metric in SIGNAL_METRICS:
                metric_data = param_results["signal_metrics"][metric]
            elif metric in CONSISTENCY_METRICS:
                metric_data = param_results["consistency_metrics"][metric]
            else:  # PROPERTY_METRICS
                metric_data = param_results["property_metrics"][metric]

            # Use all individual data points, not just means
            all_x_values = []
            all_y_values = []
            all_family_ids = []

            for k, data_point in enumerate(metric_data):
                if data_point["values"]:
                    all_graph_values = data_point["values"]
                    if all_graph_values:
                        all_x_values.extend([x_values[k]] * len(all_graph_values))
                        all_y_values.extend(all_graph_values)
                        # Create family IDs - each test value represents a different family
                        all_family_ids.extend([f"family_{k}_{param_name}"] * len(all_graph_values))

            if len(all_x_values) > 2 and np.std(all_x_values) > 0:
                result_tuple = calculate_correlation_with_significance(
                    np.array(all_x_values),
                    np.array(all_y_values),
                    param_config["type"],
                    family_ids=np.array(all_family_ids),
                )
                correlation_value, _direction, significance = result_tuple[0], result_tuple[1], result_tuple[2]

                continuous_correlation_matrix[i, j] = correlation_value
                continuous_significance_matrix[i, j] = significance
            else:
                continuous_correlation_matrix[i, j] = 0.0
                continuous_significance_matrix[i, j] = "ns"

    # Create the heatmap
    _fig, ax = plt.subplots(figsize=(12, 10))

    # Use a diverging colormap centered at 0
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(-1, 1)

    # Create the heatmap
    heatmap = ax.imshow(continuous_correlation_matrix, cmap=cmap, norm=norm, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15)

    # Add significance markers
    for i in range(len(continuous_params)):
        for j in range(len(metrics)):
            significance = continuous_significance_matrix[i, j]
            if significance != "ns":
                ax.text(j, i, significance, ha="center", va="center", color="black", fontweight="bold")

    # Set tick labels
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(continuous_params)))

    # Use display names if available
    metric_labels = [display_names.get(metric, metric) for metric in metrics]
    param_labels = [display_names.get(param, param) for param in continuous_params]

    ax.set_xticklabels(metric_labels, rotation=45, ha="right")
    ax.set_yticklabels(param_labels)

    # Add grid lines
    ax.set_xticks(np.arange(-.5, len(metrics), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(continuous_params), 1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)

    # Set title and adjust layout
    plt.title("Parameter Sensitivity Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def convert_random_baseline_to_summary_format(param_values, metric_values):
    """
    Convert random baseline data format (param_values, metric_values) to the format expected by create_summary_heatmap.

    Args:
        param_values: Dict of parameter_name -> list of values
        metric_values: Dict of metric_name -> list of values

    Returns:
        results_dict: Dict in the format expected by create_summary_heatmap
    """
    results_dict = {}

    # Only process parameters that are in PARAMS_OF_INTEREST
    for param_name in PARAMS_OF_INTEREST:
        if param_name not in param_values:
            continue

        param_config = ALL_VARIABLE_PARAMS.get(param_name, {})
        param_vals = param_values[param_name]

        # For range parameters, we need to reconstruct approximate ranges from the midpoints
        # For other parameters, use the values as-is
        if param_config.get("type") == "range":
            # For range parameters, the param_vals are already midpoints
            # Instead of creating fake ranges, use the midpoints directly as test values
            unique_midpoints = sorted(set(param_vals))
            unique_param_vals = unique_midpoints
        else:
            # For non-range parameters, use the values directly
            unique_param_vals = sorted(set(param_vals))

        # Create the structure expected by create_summary_heatmap
        param_results = {
            "test_values": unique_param_vals,
            "signal_metrics": {metric: [] for metric in SIGNAL_METRICS},
            "consistency_metrics": {metric: [] for metric in CONSISTENCY_METRICS},
            "property_metrics": {metric: [] for metric in PROPERTY_METRICS},
            "metadata": [],
        }

        # For each unique parameter value, collect the corresponding metric values
        for test_value in unique_param_vals:
            # Find all indices where this parameter value occurs
            if param_config.get("type") == "range":
                # For range parameters, test_value is now a midpoint (float), and param_vals contains midpoints
                # Find indices where the midpoint matches this test_value
                value_indices = [
                    i for i, val in enumerate(param_vals) if abs(val - test_value) < 1e-6
                ]
            else:
                # For non-range parameters, match exactly
                value_indices = [i for i, val in enumerate(param_vals) if val == test_value]

            # Skip if no indices found
            if not value_indices:
                # Add empty entries for all metrics
                for metric in SIGNAL_METRICS:
                    param_results["signal_metrics"][metric].append(
                        {"mean": np.nan, "std": np.nan, "values": []}
                    )
                for metric in CONSISTENCY_METRICS:
                    param_results["consistency_metrics"][metric].append(
                        {"mean": np.nan, "std": np.nan, "values": []}
                    )
                for metric in PROPERTY_METRICS:
                    param_results["property_metrics"][metric].append(
                        {"mean": np.nan, "std": np.nan, "values": []}
                    )
                continue

            # For each metric type, collect the values at these indices
            for metric in SIGNAL_METRICS:
                if metric in metric_values and len(metric_values[metric]) > max(value_indices):
                    metric_vals_for_this_param = [
                        metric_values[metric][i]
                        for i in value_indices
                        if i < len(metric_values[metric])
                        and not is_invalid_value(metric_values[metric][i])
                    ]
                    param_results["signal_metrics"][metric].append(
                        {
                            "mean": np.mean(metric_vals_for_this_param)
                            if metric_vals_for_this_param
                            else np.nan,
                            "std": np.std(metric_vals_for_this_param)
                            if metric_vals_for_this_param
                            else np.nan,
                            "values": metric_vals_for_this_param,
                        }
                    )
                else:
                    param_results["signal_metrics"][metric].append(
                        {"mean": np.nan, "std": np.nan, "values": []}
                    )

            for metric in CONSISTENCY_METRICS:
                if metric in metric_values and len(metric_values[metric]) > max(value_indices):
                    metric_vals_for_this_param = [
                        metric_values[metric][i]
                        for i in value_indices
                        if i < len(metric_values[metric])
                        and not is_invalid_value(metric_values[metric][i])
                    ]
                    param_results["consistency_metrics"][metric].append(
                        {
                            "mean": np.mean(metric_vals_for_this_param)
                            if metric_vals_for_this_param
                            else np.nan,
                            "std": np.std(metric_vals_for_this_param)
                            if metric_vals_for_this_param
                            else np.nan,
                            "values": metric_vals_for_this_param,
                        }
                    )
                else:
                    param_results["consistency_metrics"][metric].append(
                        {"mean": np.nan, "std": np.nan, "values": []}
                    )

            for metric in PROPERTY_METRICS:
                if metric in metric_values and len(metric_values[metric]) > max(value_indices):
                    metric_vals_for_this_param = [
                        metric_values[metric][i]
                        for i in value_indices
                        if i < len(metric_values[metric])
                        and not is_invalid_value(metric_values[metric][i])
                    ]
                    param_results["property_metrics"][metric].append(
                        {
                            "mean": np.mean(metric_vals_for_this_param)
                            if metric_vals_for_this_param
                            else np.nan,
                            "std": np.std(metric_vals_for_this_param)
                            if metric_vals_for_this_param
                            else np.nan,
                            "values": metric_vals_for_this_param,
                        }
                    )
                else:
                    param_results["property_metrics"][metric].append(
                        {"mean": np.nan, "std": np.nan, "values": []}
                    )

        # Add to results dictionary
        results_dict[param_name] = param_results

    return results_dict

def plot_random_baseline_results(
    random_results, save_dir="random_baseline_plots", display_names=HEATMAP_DISPLAY_NAMES
):
    """
    Create plots for random baseline results showing each parameter of interest vs all metrics.
    For categorical parameters: use boxplots with statistical testing
    For continuous parameters: use scatter plots with dark gray dotted correlation lines and error bars

    Args:
        random_results: Dictionary containing random baseline results
        save_dir: Directory to save plots
        display_names: Dictionary mapping parameter and metric names to display names
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"Creating random baseline plots in {save_dir}")

    # Verify data consistency
    n_samples = len(random_results["random_samples"])
    for metric in SIGNAL_METRICS:
        if metric in random_results["signal_metrics"]:
            n_metric_samples = len(random_results["signal_metrics"][metric])
            if n_metric_samples != n_samples:
                print(
                    f"WARNING: Mismatch in {metric}: {n_metric_samples} metric samples vs {n_samples} parameter samples"
                )

    # Extract parameter values from the random samples (only for PARAMS_OF_INTEREST)
    param_values = {}
    for param_name in PARAMS_OF_INTEREST:
        param_values[param_name] = []

    # Extract metric values and their standard deviations
    metric_values = {}
    metric_stds = {}
    for metric in SIGNAL_METRICS + CONSISTENCY_METRICS + PROPERTY_METRICS:
        metric_values[metric] = []
        metric_stds[metric] = []

    # Process the results to extract parameter and metric values
    # Iterate over all samples directly instead of relying on metric data
    for sample_idx in range(len(random_results["random_samples"])):
        # Get the parameters for this sample
        sample_params = random_results["random_samples"][sample_idx]

        # Extract parameter values (only for PARAMS_OF_INTEREST)
        for param_name in PARAMS_OF_INTEREST:
            param_config = ALL_VARIABLE_PARAMS[param_name]
            if param_config["level"] == "universe":
                value = sample_params["universe"].get(param_name)
            else:
                value = sample_params["family"].get(param_name)

            # For range parameters, use the midpoint
            if isinstance(value, tuple) and len(value) == 2:
                value = (value[0] + value[1]) / 2

            # Special handling for degree_center_method to convert to numeric values for plotting
            if param_name == "degree_center_method" and value == "random":
                value = True
            elif param_name == "degree_center_method" and value == "constant":
                value = False

            param_values[param_name].append(value)

        # Extract metric values and standard deviations (across repeats for this sample)
        for metric in SIGNAL_METRICS:
            if (
                metric in random_results["signal_metrics"]
                and sample_idx < len(random_results["signal_metrics"][metric])
                and random_results["signal_metrics"][metric][sample_idx]
            ):
                # Flatten all values across repeats and graphs
                all_values = []
                for repeat_data in random_results["signal_metrics"][metric][sample_idx]:
                    if repeat_data:
                        all_values.extend(repeat_data)
                if all_values:
                    metric_values[metric].append(np.mean(all_values))
                    metric_stds[metric].append(np.std(all_values))
                else:
                    metric_values[metric].append(np.nan)
                    metric_stds[metric].append(np.nan)
            else:
                metric_values[metric].append(np.nan)
                metric_stds[metric].append(np.nan)

        for metric in CONSISTENCY_METRICS:
            if (
                metric in random_results["consistency_metrics"]
                and sample_idx < len(random_results["consistency_metrics"][metric])
                and random_results["consistency_metrics"][metric][sample_idx]
            ):
                all_values = []
                for repeat_data in random_results["consistency_metrics"][metric][sample_idx]:
                    if repeat_data:
                        all_values.extend(repeat_data)
                if all_values:
                    metric_values[metric].append(np.mean(all_values))
                    metric_stds[metric].append(np.std(all_values))
                else:
                    metric_values[metric].append(np.nan)
                    metric_stds[metric].append(np.nan)
            else:
                metric_values[metric].append(np.nan)
                metric_stds[metric].append(np.nan)

        for metric in PROPERTY_METRICS:
            if (
                metric in random_results["property_metrics"]
                and sample_idx < len(random_results["property_metrics"][metric])
                and random_results["property_metrics"][metric][sample_idx]
            ):
                all_values = []
                for repeat_data in random_results["property_metrics"][metric][sample_idx]:
                    if repeat_data:
                        all_values.extend(repeat_data)
                if all_values:
                    metric_values[metric].append(np.mean(all_values))
                    metric_stds[metric].append(np.std(all_values))
                else:
                    metric_values[metric].append(np.nan)
                    metric_stds[metric].append(np.nan)
            else:
                metric_values[metric].append(np.nan)
                metric_stds[metric].append(np.nan)

    # Create plots for each parameter of interest
    for param_name in PARAMS_OF_INTEREST:
        # Get parameter values and check if it's categorical
        param_config = ALL_VARIABLE_PARAMS[param_name]
        # Treat degree_center_method as categorical since we convert it to True/False
        is_categorical = (
            param_config["type"] in ["categorical", "boolean"]
            or param_name == "degree_center_method"
        )

        # Create subplots for all metrics with new organization:
        # Row 1: 3 signal metrics
        # Row 2: 3 consistency metrics
        # Row 3: 3 main property metrics
        # Row 4: 2 technical metrics (centered)

        # Use simple 4 rows, 3 columns grid layout
        n_rows_metric, n_cols_metric = 4, 3

        # Create subplot for this parameter with simple grid layout
        fig_width = 4.5 * n_cols_metric
        fig_height = 4.5 * n_rows_metric
        param_fig, param_axes = plt.subplots(
            n_rows_metric,
            n_cols_metric,
            figsize=(fig_width, fig_height),
            constrained_layout=True,
        )

        # Create new ordered list of metrics with proper organization:
        # Row 1: Signal metrics (3)
        # Row 2: Consistency metrics (3)
        # Row 3: Main property metrics (3)
        # Row 4: Technical metrics (2, centered)
        ordered_metrics = (
            SIGNAL_METRICS + CONSISTENCY_METRICS + MAIN_PROPERTY_METRICS + TECHNICAL_METRICS
        )
        metric_types = (
            ["signal"] * len(SIGNAL_METRICS)
            + ["consistency"] * len(CONSISTENCY_METRICS)
            + ["main_property"] * len(MAIN_PROPERTY_METRICS)
            + ["technical"] * len(TECHNICAL_METRICS)
        )

        # Plot all metrics in simple grid order
        for metric_idx_loop, (metric, metric_type) in enumerate(
            zip(ordered_metrics, metric_types, strict=False)
        ):
            # Calculate position in simple 3x4 grid
            row = metric_idx_loop // 3
            col = metric_idx_loop % 3

            # Skip if we're beyond our grid
            if row >= n_rows_metric or col >= n_cols_metric:
                continue

            metric_ax = param_axes[row, col]

            # Get parameter and metric values
            if param_name in param_values and metric in metric_values:
                param_vals = param_values[param_name]
                metric_vals = metric_values[metric]
                metric_std_vals = metric_stds[metric]

                # Remove NaN values
                valid_indices = [
                    i
                    for i in range(len(param_vals))
                    if not (is_invalid_value(param_vals[i]) or is_invalid_value(metric_vals[i]))
                ]

                if len(valid_indices) > 1:
                    valid_params = [param_vals[i] for i in valid_indices]
                    valid_metrics = [metric_vals[i] for i in valid_indices]
                    valid_stds = [metric_std_vals[i] for i in valid_indices]

                    if is_categorical:
                        # For categorical parameters, use boxplots
                        box_data = []
                        box_positions = []
                        x_labels = []

                        # Group data by parameter values
                        unique_params = sorted(set(valid_params))
                        for i, param_val in enumerate(unique_params):
                            mask = [p == param_val for p in valid_params]
                            group_metrics = [
                                valid_metrics[j] for j in range(len(mask)) if mask[j]
                            ]
                            if group_metrics:
                                box_data.append(group_metrics)
                                box_positions.append(i)
                                x_labels.append(str(param_val))

                        # Use appropriate colors based on metric type - colorblind friendly
                        if metric_type == "signal":
                            colors = {"face": "lightblue", "median": "blue", "whiskers": "blue"}
                        elif metric_type == "consistency":
                            colors = {
                                "face": "lightgreen",
                                "median": "darkgreen",
                                "whiskers": "darkgreen",
                            }
                        elif metric_type == "main_property":
                            colors = {
                                "face": "mistyrose",
                                "median": "firebrick",
                                "whiskers": "firebrick",
                            }
                        else:  # technical
                            colors = {
                                "face": "lightgray",
                                "median": "black",
                                "whiskers": "black",
                            }
                        create_categorical_boxplot_with_tests(
                            metric_ax, box_data, box_positions, x_labels, colors, "lightgray"
                        )

                    else:
                        # For continuous parameters, use scatter plot with error bars - color by metric type
                        if metric_type == "signal":
                            point_color = "blue"
                        elif metric_type == "consistency":
                            point_color = "darkgreen"
                        elif metric_type == "main_property":
                            point_color = "firebrick"
                        else:  # technical
                            point_color = "black"
                        metric_ax.errorbar(
                            valid_params,
                            valid_metrics,
                            yerr=valid_stds,
                            fmt="o",
                            alpha=0.6,
                            markersize=4,
                            capsize=3,
                            capthick=1,
                            color=point_color,
                        )

                        # Add dark gray dotted correlation line (no confidence intervals, just simple fit)
                        if len(valid_params) > 1:
                            z = np.polyfit(valid_params, valid_metrics, 1)
                            p = np.poly1d(z)
                            metric_ax.plot(
                                valid_params,
                                p(valid_params),
                                ":",
                                alpha=0.8,
                                linewidth=3,
                                color="darkgray",
                            )

                        # Calculate correlation with significance
                        # For randomized data, each sample is its own family
                        family_ids = [f"sample_{idx}" for idx in range(len(valid_params))]
                        (
                            corr,
                            direction,
                            significance,
                            ci_lower,
                            ci_upper,
                        ) = calculate_correlation_with_significance(
                            np.array(valid_params),
                            np.array(valid_metrics),
                            param_config.get("type", "continuous"),
                            family_ids=np.array(family_ids),
                        )

                        # Add correlation info
                        corr_display = format_correlation_display(
                            corr, direction, significance, ci_lower, ci_upper
                        )
                        metric_ax.text(
                            0.05,
                            0.95,
                            corr_display,
                            transform=metric_ax.transAxes,
                            verticalalignment="top",
                            bbox={"facecolor": "white", "edgecolor": "black", "linewidth": 0.8, "alpha": 0.9, "boxstyle": "round,pad=0.3"}
                        )

                    # Use display names from HEATMAP_DISPLAY_NAMES if available
                    metric_title = display_names.get(
                        metric, metric.replace("_", " ").title()
                    )
                    metric_ax.set_title(metric_title, fontsize=16, weight="bold")
                    metric_ax.set_xlabel(get_plot_param_name(param_name), fontsize=14)
                    metric_ax.set_ylabel("Value")
                    apply_publication_style(metric_ax)
                else:
                    metric_ax.text(
                        0.5,
                        0.5,
                        "No valid data",
                        ha="center",
                        va="center",
                        transform=metric_ax.transAxes,
                    )
                    # Use display names from HEATMAP_DISPLAY_NAMES if available
                    metric_title = display_names.get(metric, metric.replace("_", " ").title())
                    metric_ax.set_title(metric_title, fontsize=16, weight="bold")

        # Hide the unused subplot (last position in 3x4 grid)
        if len(ordered_metrics) < n_rows_metric * n_cols_metric:
            param_axes[3, 2].set_visible(False)

        # Set the main title for this parameter using display names
        param_title = display_names.get(param_name, get_plot_param_name(param_name))
        param_fig.suptitle(f"{param_title}", fontsize=20, y=0.98, fontweight="bold")
        param_fig.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the title

        # Save individual parameter plot
        param_output_file = os.path.join(save_dir, f"{param_name}_random_baseline_analysis.png")
        param_fig.savefig(param_output_file, dpi=300, bbox_inches="tight")
        plt.close(param_fig)

    # Convert random baseline data to summary format and create summary heatmap
    random_baseline_summary_format = convert_random_baseline_to_summary_format(
        param_values, metric_values
    )
    heatmap_path = os.path.join(save_dir, "random_baseline_correlation_heatmap.png")
    create_summary_heatmap(
        random_baseline_summary_format, heatmap_path, display_names=display_names
    )
    print(f"Random baseline summary heatmap saved to {heatmap_path}")

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze parameter effects on signals and consistency using randomized analysis"
    )
    parser.add_argument(
        "--output-dir", default="parameter_analysis_results", help="Directory to save results"
    )
    parser.add_argument("--n-graphs", type=int, default=10, help="Number of graphs per family")
    parser.add_argument("--plot-only", action="store_true", help="Only plot existing results")
    parser.add_argument(
        "--n-random-samples",
        type=int,
        default=10,
        help="Number of random parameter samples to generate",
    )
    parser.add_argument(
        "--n-random-repeats-per-sample",
        type=int,
        default=1,
        help="Number of repeats per random sample",
    )

    args = parser.parse_args()

    # Update global settings
    GRAPHS_PER_FAMILY = args.n_graphs

    if not args.plot_only:
        # Run random baseline analysis
        random_results = run_random_baseline_analysis(
            n_samples=args.n_random_samples,
            n_repeats_per_sample=args.n_random_repeats_per_sample,
            output_dir=args.output_dir,
        )
        # Create plots for random baseline
        plot_random_baseline_results(
            random_results, os.path.join(args.output_dir, "random_baseline_plots")
        )
    else:
        # Try to load random baseline results
        try:
            with open(os.path.join(args.output_dir, "random_baseline_analysis.pkl"), "rb") as f:
                random_results = pickle.load(f)
            plot_random_baseline_results(
                random_results, os.path.join(args.output_dir, "random_baseline_plots")
            )
        except FileNotFoundError:
            print("Random baseline analysis results not found.")

    print("\nAnalysis complete! Check the output directory for results and plots.")
