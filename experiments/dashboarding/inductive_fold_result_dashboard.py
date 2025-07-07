import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
from scipy.interpolate import griddata
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import rankdata

st.set_page_config(page_title="GraphWorld Benchmarking Visualization", layout="wide")
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import curve_fit
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def propagate_uncertainty_linear_regression(x, y_mean, y_std, x_pred=None):
    """
    Fit linear regression with proper uncertainty propagation using bootstrap + analytical methods.
    
    Returns fitted line with confidence bands and statistical measures.
    """
    n_bootstrap = 1000  # Bootstrap samples
    n_points = len(x)
    
    if x_pred is None:
        x_pred = np.linspace(x.min(), x.max(), 100)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y_mean) | np.isnan(y_std))
    x_clean = x[valid_mask]
    y_mean_clean = y_mean[valid_mask]
    y_std_clean = y_std[valid_mask]
    
    if len(x_clean) < 3:
        # Not enough data points
        return None
    
    # Method 1: Bootstrap resampling of data points with uncertainty
    bootstrap_slopes = []
    bootstrap_intercepts = []
    bootstrap_predictions = []
    correlations = []
    r_squared_values = []
    
    for _ in range(n_bootstrap):
        # Bootstrap resample indices
        boot_indices = np.random.choice(len(x_clean), size=len(x_clean), replace=True)
        x_boot = x_clean[boot_indices]
        y_mean_boot = y_mean_clean[boot_indices]
        y_std_boot = y_std_clean[boot_indices]
        
        # Sample from uncertainty for each bootstrapped point
        y_boot = np.random.normal(y_mean_boot, y_std_boot)
        
        # Fit regression
        try:
            reg = LinearRegression().fit(x_boot.reshape(-1, 1), y_boot)
            slope = reg.coef_[0]
            intercept = reg.intercept_
            
            # Calculate R²
            y_pred_boot = reg.predict(x_boot.reshape(-1, 1))
            ss_res = np.sum((y_boot - y_pred_boot) ** 2)
            ss_tot = np.sum((y_boot - np.mean(y_boot)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Calculate correlation
            corr, _ = stats.pearsonr(x_boot, y_boot)
            
            if not (np.isnan(slope) or np.isnan(intercept) or np.isnan(r_squared) or np.isnan(corr)):
                bootstrap_slopes.append(slope)
                bootstrap_intercepts.append(intercept)
                r_squared_values.append(r_squared)
                correlations.append(corr)
                
                # Predictions for this bootstrap sample
                pred_boot = reg.predict(x_pred.reshape(-1, 1))
                bootstrap_predictions.append(pred_boot)
        except:
            continue
    
    if len(bootstrap_slopes) < 50:  # Need sufficient bootstrap samples
        return None
    
    bootstrap_slopes = np.array(bootstrap_slopes)
    bootstrap_intercepts = np.array(bootstrap_intercepts)
    bootstrap_predictions = np.array(bootstrap_predictions)
    correlations = np.array(correlations)
    r_squared_values = np.array(r_squared_values)
    
    # Method 2: Analytical uncertainty propagation for comparison
    # Fit main regression with mean values
    main_reg = LinearRegression().fit(x_clean.reshape(-1, 1), y_mean_clean)
    mean_slope = main_reg.coef_[0]
    mean_intercept = main_reg.intercept_
    
    # Calculate residuals and effective uncertainty
    y_pred_main = main_reg.predict(x_clean.reshape(-1, 1))
    residuals = y_mean_clean - y_pred_main
    
    # Combine measurement uncertainty with model uncertainty
    total_variance = y_std_clean**2 + np.var(residuals)
    effective_std = np.sqrt(np.mean(total_variance))
    
    # Analytical confidence intervals for predictions
    n = len(x_clean)
    x_mean = np.mean(x_clean)
    sxx = np.sum((x_clean - x_mean)**2)
    
    # Standard error of prediction at each x_pred point
    se_pred = effective_std * np.sqrt(1 + 1/n + (x_pred - x_mean)**2 / sxx)
    
    # Use bootstrap results as primary, analytical as fallback
    mean_line_boot = np.mean(bootstrap_predictions, axis=0)
    pred_lower_boot = np.percentile(bootstrap_predictions, 2.5, axis=0)
    pred_upper_boot = np.percentile(bootstrap_predictions, 97.5, axis=0)
    
    # Ensure bands are not too narrow by enforcing minimum width
    pred_std_boot = np.std(bootstrap_predictions, axis=0)
    min_band_width = 2 * effective_std  # Minimum reasonable width
    
    current_width = pred_upper_boot - pred_lower_boot
    too_narrow_mask = current_width < min_band_width
    
    if np.any(too_narrow_mask):
        # Expand narrow regions using analytical approach
        analytical_line = mean_slope * x_pred + mean_intercept
        analytical_lower = analytical_line - 1.96 * se_pred
        analytical_upper = analytical_line + 1.96 * se_pred
        
        pred_lower_boot[too_narrow_mask] = analytical_lower[too_narrow_mask]
        pred_upper_boot[too_narrow_mask] = analytical_upper[too_narrow_mask]
    
    # Statistical tests using bootstrap distributions
    slope_mean = np.mean(bootstrap_slopes)
    slope_std = np.std(bootstrap_slopes)
    
    # IMPROVED: Bootstrap-based p-value for slope ≠ 0
    # Count how many bootstrap slopes have opposite sign from mean
    # This is a more robust test that accounts for full uncertainty structure
    if slope_mean > 0:
        slope_p_value = 2 * np.mean(bootstrap_slopes <= 0)  # Two-tailed test
    else:
        slope_p_value = 2 * np.mean(bootstrap_slopes >= 0)  # Two-tailed test
    
    # Ensure p-value is not exactly 0 (add small epsilon for numerical stability)
    slope_p_value = max(slope_p_value, 1 / len(bootstrap_slopes))
    
    # Calculate t-statistic for comparison (but use bootstrap p-value as primary)
    slope_t_stat = slope_mean / (slope_std / np.sqrt(len(bootstrap_slopes))) if slope_std > 0 else 0
    slope_p_value_classical = 2 * (1 - stats.t.cdf(abs(slope_t_stat), len(bootstrap_slopes) - 1))
    
    # Correlation statistics
    mean_correlation = np.mean(correlations)
    corr_ci_lower = np.percentile(correlations, 2.5)
    corr_ci_upper = np.percentile(correlations, 97.5)
    
    # IMPROVED: Bootstrap-based p-value for correlation ≠ 0
    # More robust than classical t-test as it accounts for uncertainty structure
    if mean_correlation > 0:
        corr_p_value = 2 * np.mean(correlations <= 0)  # Two-tailed test
    else:
        corr_p_value = 2 * np.mean(correlations >= 0)  # Two-tailed test
    
    # Ensure p-value is not exactly 0
    corr_p_value = max(corr_p_value, 1 / len(correlations))
    
    # Calculate classical t-statistic for comparison
    corr_t_stat = mean_correlation * np.sqrt((n - 2) / (1 - mean_correlation**2)) if abs(mean_correlation) < 0.99 else 0
    corr_p_value_classical = 2 * (1 - stats.t.cdf(abs(corr_t_stat), n - 2))
    
    return {
        'x_pred': x_pred,
        'mean_line': mean_line_boot,
        'pred_lower': pred_lower_boot,
        'pred_upper': pred_upper_boot,
        'slope': {
            'mean': slope_mean,
            'std': slope_std,
            't_stat': slope_t_stat,
            'p_value': slope_p_value,  # Bootstrap-based p-value
            'p_value_classical': slope_p_value_classical  # Classical t-test p-value
        },
        'correlation': {
            'mean': mean_correlation,
            'ci_lower': corr_ci_lower,
            'ci_upper': corr_ci_upper,
            't_stat': corr_t_stat,
            'p_value': corr_p_value,  # Bootstrap-based p-value
            'p_value_classical': corr_p_value_classical  # Classical t-test p-value
        },
        'r_squared': {
            'mean': np.mean(r_squared_values),
            'std': np.std(r_squared_values)
        },
        'effective_uncertainty': effective_std,
        'n_bootstrap_samples': len(bootstrap_slopes),
        'test_interpretation': {
            'slope_test': 'H₀: slope = 0 (no relationship) vs H₁: slope ≠ 0 (relationship exists)',
            'correlation_test': 'H₀: correlation = 0 (no linear relationship) vs H₁: correlation ≠ 0 (linear relationship exists)',
            'rejection_criterion': 'p < 0.05 means reject H₀ (relationship IS significant)'
        }
    }

def create_enhanced_statistical_plot(df, x_param, task, metric, models_to_analyze):
    """
    Create enhanced statistical plots with uncertainty-aware trend analysis.
    """
    # Filter out runs with NaN values for any of the selected models
    relevant_columns = [x_param]
    for model in models_to_analyze:
        model_col_mean = f'{task}-{model}-{metric}_mean'
        model_col_std = f'{task}-{model}-{metric}_std'
        if model_col_mean in df.columns and model_col_std in df.columns:
            relevant_columns.extend([model_col_mean, model_col_std])
    
    df_clean = df[relevant_columns].dropna()
    
    if len(df_clean) == 0:
        st.error("❌ No complete data found for the selected models and metric!")
        return None, None
    
    # Prepare x data
    x_data = df_clean[x_param].values
    
    # Calculate statistics for all graph-based models combined
    all_graph_y_means = []
    all_graph_y_stds = []
    all_graph_x = []
    
    for model in models_to_analyze:
        model_col_mean = f'{task}-{model}-{metric}_mean'
        model_col_std = f'{task}-{model}-{metric}_std'
        if model_col_mean in df_clean.columns and model_col_std in df_clean.columns:
            all_graph_y_means.extend(df_clean[model_col_mean].values)
            all_graph_y_stds.extend(df_clean[model_col_std].values)
            all_graph_x.extend(x_data)
    
    all_graph_y_means = np.array(all_graph_y_means)
    all_graph_y_stds = np.array(all_graph_y_stds)
    all_graph_x = np.array(all_graph_x)
    
    # Global analysis (all graph-based models)
    global_stats = propagate_uncertainty_linear_regression(all_graph_x, all_graph_y_means, all_graph_y_stds)
    
    # Individual model analyses
    individual_stats = {}
    for model in models_to_analyze:
        model_col_mean = f'{task}-{model}-{metric}_mean'
        model_col_std = f'{task}-{model}-{metric}_std'
        if model_col_mean in df_clean.columns and model_col_std in df_clean.columns:
            y_means = df_clean[model_col_mean].values
            y_stds = df_clean[model_col_std].values
            individual_stats[model] = propagate_uncertainty_linear_regression(x_data, y_means, y_stds)
    
    # Create subplot figure
    n_models = len(individual_stats)
    n_cols = min(3, n_models)  # Max 3 columns
    n_rows = (n_models + n_cols - 1) // n_cols + 1  # +1 for global plot
    
    subplot_titles = ['All Graph-Based Models (Global)'] + [f'{model.upper()}' for model in individual_stats.keys()]
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Color palette - use hex colors to avoid conversion issues
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Global plot (first subplot)
    # Scatter points for all models
    for i, model in enumerate(models_to_analyze):
        model_col_mean = f'{task}-{model}-{metric}_mean'
        model_col_std = f'{task}-{model}-{metric}_std'
        if model_col_mean in df_clean.columns and model_col_std in df_clean.columns:
            fig.add_trace(go.Scatter(
                x=x_data,
                y=df_clean[model_col_mean],
                error_y=dict(type='data', array=df_clean[model_col_std], visible=True),
                mode='markers',
                name=f'{model.upper()}',
                marker=dict(color=colors[i % len(colors)], size=6, opacity=0.7),
                showlegend=True,
                legendgroup='models'
            ), row=1, col=1)
    
    # Global trend line with confidence band
    fig.add_trace(go.Scatter(
        x=global_stats['x_pred'],
        y=global_stats['pred_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=global_stats['x_pred'],
        y=global_stats['pred_lower'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(128,128,128,0.2)',
        line=dict(width=0),
        name='95% CI (Global)',
        showlegend=True,
        legendgroup='global'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=global_stats['x_pred'],
        y=global_stats['mean_line'],
        mode='lines',
        line=dict(color='black', width=3),
        name='Global Trend',
        showlegend=True,
        legendgroup='global'
    ), row=1, col=1)
    
    # Individual model plots
    for i, (model, stats) in enumerate(individual_stats.items()):
        row = (i // n_cols) + 2  # +2 because global plot takes first row
        col = (i % n_cols) + 1
        
        model_col_mean = f'{task}-{model}-{metric}_mean'
        model_col_std = f'{task}-{model}-{metric}_std'
        
        # Scatter points
        fig.add_trace(go.Scatter(
            x=x_data,
            y=df_clean[model_col_mean],
            error_y=dict(type='data', array=df_clean[model_col_std], visible=True),
            mode='markers',
            name=f'{model.upper()} Data',
            marker=dict(color=colors[i % len(colors)], size=8, opacity=0.8),
            showlegend=False
        ), row=row, col=col)
        
        # Confidence band
        fig.add_trace(go.Scatter(
            x=stats['x_pred'],
            y=stats['pred_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ), row=row, col=col)
        
        # Convert hex color to RGB for rgba
        hex_color = colors[i % len(colors)]
        rgb_values = tuple(int(hex_color[j:j+2], 16) for j in (1, 3, 5))  # Skip # and get R,G,B
        rgba_color = f'rgba({rgb_values[0]},{rgb_values[1]},{rgb_values[2]},0.2)'
        
        fig.add_trace(go.Scatter(
            x=stats['x_pred'],
            y=stats['pred_lower'],
            mode='lines',
            fill='tonexty',
            fillcolor=rgba_color,
            line=dict(width=0),
            showlegend=False
        ), row=row, col=col)
        
        # Trend line
        fig.add_trace(go.Scatter(
            x=stats['x_pred'],
            y=stats['mean_line'],
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=3),
            showlegend=False
        ), row=row, col=col)
    
    # Update layout
    fig.update_layout(
        title=f"Statistical Analysis: {metric.upper()} vs {x_param}<br>"
              f"<sub>Global: r={global_stats['correlation']['mean']:.3f} "
              f"[{global_stats['correlation']['ci_lower']:.3f}, {global_stats['correlation']['ci_upper']:.3f}], "
              f"p={global_stats['correlation']['p_value']:.3f}</sub>",
        title_x=0.5,
        height=400 * n_rows,
        showlegend=True,
        legend=dict(x=1.02, y=1, xanchor='left')
    )
    
    # Update axes labels
    for i in range(n_rows):
        for j in range(n_cols):
            if i == n_rows - 1:  # Bottom row
                fig.update_xaxes(title_text=x_param, row=i+1, col=j+1)
            if j == 0:  # Left column
                fig.update_yaxes(title_text=metric.upper(), row=i+1, col=j+1)
    
    return fig, {'global': global_stats, 'individual': individual_stats}


def create_statistical_summary_table(stats_results):
    """Create a summary table of statistical results."""
    summary_data = []
    
    # Global results
    global_stats = stats_results['global']
    summary_data.append({
        'Model': 'All Graph-Based (Global)',
        'Correlation (r)': f"{global_stats['correlation']['mean']:.3f}",
        'Correlation 95% CI': f"[{global_stats['correlation']['ci_lower']:.3f}, {global_stats['correlation']['ci_upper']:.3f}]",
        'Correlation p-value (Bootstrap)': f"{global_stats['correlation']['p_value']:.3f}",
        'Correlation p-value (Classical)': f"{global_stats['correlation'].get('p_value_classical', 'N/A'):.3f}" if global_stats['correlation'].get('p_value_classical') is not None else 'N/A',
        'Slope': f"{global_stats['slope']['mean']:.3f} ± {global_stats['slope']['std']:.3f}",
        'Slope p-value (Bootstrap)': f"{global_stats['slope']['p_value']:.3f}",
        'Slope p-value (Classical)': f"{global_stats['slope'].get('p_value_classical', 'N/A'):.3f}" if global_stats['slope'].get('p_value_classical') is not None else 'N/A',
        'R² (mean ± std)': f"{global_stats['r_squared']['mean']:.3f} ± {global_stats['r_squared']['std']:.3f}",
        'Bootstrap Samples': global_stats.get('n_bootstrap_samples', 'N/A'),
        'Significance (Bootstrap)': '***' if global_stats['correlation']['p_value'] < 0.001 else 
                                   '**' if global_stats['correlation']['p_value'] < 0.01 else 
                                   '*' if global_stats['correlation']['p_value'] < 0.05 else 'ns'
    })
    
    # Individual model results
    for model, stats in stats_results['individual'].items():
        if stats is not None:  # Check if analysis succeeded
            summary_data.append({
                'Model': model.upper(),
                'Correlation (r)': f"{stats['correlation']['mean']:.3f}",
                'Correlation 95% CI': f"[{stats['correlation']['ci_lower']:.3f}, {stats['correlation']['ci_upper']:.3f}]",
                'Correlation p-value (Bootstrap)': f"{stats['correlation']['p_value']:.3f}",
                'Correlation p-value (Classical)': f"{stats['correlation'].get('p_value_classical', 'N/A'):.3f}" if stats['correlation'].get('p_value_classical') is not None else 'N/A',
                'Slope': f"{stats['slope']['mean']:.3f} ± {stats['slope']['std']:.3f}",
                'Slope p-value (Bootstrap)': f"{stats['slope']['p_value']:.3f}",
                'Slope p-value (Classical)': f"{stats['slope'].get('p_value_classical', 'N/A'):.3f}" if stats['slope'].get('p_value_classical') is not None else 'N/A',
                'R² (mean ± std)': f"{stats['r_squared']['mean']:.3f} ± {stats['r_squared']['std']:.3f}",
                'Bootstrap Samples': stats.get('n_bootstrap_samples', 'N/A'),
                'Significance (Bootstrap)': '***' if stats['correlation']['p_value'] < 0.001 else 
                                           '**' if stats['correlation']['p_value'] < 0.01 else 
                                           '*' if stats['correlation']['p_value'] < 0.05 else 'ns'
            })
    
    return pd.DataFrame(summary_data)

# Integration into your existing dashboard
def enhanced_basic_plots_tab(df, available_tasks_metrics, param_columns):
    """Enhanced version of your basic plots tab with statistical analysis."""
    st.header("Enhanced Statistical Analysis with Uncertainty")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        x_param = st.selectbox("X-axis Parameter:", param_columns, key="enhanced_x")
        
    with col2:
        selected_task = st.selectbox("Task:", list(available_tasks_metrics.keys()), key="enhanced_task")
        
    with col3:
        if selected_task in available_tasks_metrics:
            y_metric = st.selectbox("Y-axis Metric:", available_tasks_metrics[selected_task], key="enhanced_y")
        else:
            st.error("No metrics available for selected task")
            return
    
    with col4:
        # Get all models for this task and filter out MLP
        all_models = get_models_for_task(df, selected_task)
        graph_models = [model for model in all_models if 'mlp' not in model.lower()]
        
        if not graph_models:
            st.error("No graph-based models found!")
            return
            
        selected_models = st.multiselect(
            "Graph-Based Models:", 
            graph_models, 
            default=graph_models,  # Select all by default
            key="enhanced_models"
        )
    
    if selected_models and len(selected_models) >= 1:
        # Add button to start analysis
        if st.button("🚀 Start Enhanced Statistical Analysis", type="primary"):
            with st.spinner("Performing statistical analysis with uncertainty propagation..."):
                fig, stats_results = create_enhanced_statistical_plot(
                    df, x_param, selected_task, y_metric, selected_models
                )
            
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistical summary
                st.subheader("📊 Statistical Summary")
                summary_df = create_statistical_summary_table(stats_results)
                st.dataframe(summary_df, use_container_width=True)
                
                # Interpretation guide
                st.subheader("📖 Interpretation Guide")
                st.write("""
                - **Correlation (r)**: Strength and direction of linear relationship (-1 to 1)
                - **95% CI**: Confidence interval for the correlation coefficient
                - **p-value**: Probability that the observed correlation occurred by chance
                - **Significance**: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant
                - **Slope**: Rate of change in performance per unit change in parameter
                - **R²**: Proportion of variance explained by the linear relationship
                - **Shaded area**: 95% confidence band around the fitted line
                """)
                
                # Download results
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Statistical Results",
                    data=csv,
                    file_name=f"statistical_analysis_{selected_task}_{y_metric}_{x_param}.csv",
                    mime="text/csv"
                )
            else:
                st.error("Could not create statistical analysis plot.")
        else:
            st.info("👆 Click the button above to start the enhanced statistical analysis")
    else:
        st.warning("Please select at least one graph-based model for analysis.")

# Helper function you'll need to import
def get_models_for_task(df, task):
    """Get all model names for a specific task."""
    models = set()
    
    # Find all columns that match this task
    for col in df.columns:
        if col.startswith(f'{task}-') and not col.endswith('train_time'):
            # Remove the task prefix
            remaining = col[len(task)+1:]  # +1 for the hyphen
            
            # Split the remaining part
            parts = remaining.split('-')
            
            # Handle label-specific metrics (e.g., gcn-mse-label-0)
            if 'label' in parts:
                label_idx = parts.index('label')
                # Model is everything before the metric (which is before 'label')
                model_parts = parts[:label_idx-1]
            else:
                # Regular format: model-metric
                model_parts = parts[:-1]  # Everything except the last part (metric)
            
            if model_parts:
                model = '-'.join(model_parts) if len(model_parts) > 1 else model_parts[0]
                models.add(model)
    
    return sorted(list(models))

@st.cache_data
def load_experiment_data(file_path):
    """Load and parse the experiment results JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

@st.cache_data
def process_results_to_dataframe(data):
    """Convert the nested JSON results to a flat pandas DataFrame."""
    rows = []
    
    for result in data['all_results']:
        base_row = {
            'run_id': result['run_id'],
            'method': result['method'],
            'degree_distribution': result['degree_distribution'],
            'n_graphs': result['n_graphs'],
            'total_time': result['total_time'],
            'universe_K': result['universe_K']
        }
        
        # Add sweep parameters
        for key, value in result['sweep_parameters'].items():
            base_row[f'sweep_{key}'] = value
            
        # Add random parameters
        for key, value in result['random_parameters'].items():
            base_row[f'random_{key}'] = value
            
        # Add family properties
        for key, value in result['family_properties'].items():
            if isinstance(value, (int, float)):
                base_row[f'family_{key}'] = value
                
        # Add family consistency scores
        for key, value in result['family_consistency'].items():
            if isinstance(value, dict) and 'score' in value:
                base_row[f'consistency_{key}'] = value['score']
                
        # Add community signals
        for signal_type, signal_data in result['community_signals'].items():
            if isinstance(signal_data, dict):
                for stat, val in signal_data.items():
                    if stat in ['mean', 'std', 'min', 'max'] and isinstance(val, (int, float)):
                        base_row[f'signal_{signal_type}_{stat}'] = val
        
        # Add model results - now handling mean/std for each metric
        for task_name, task_results in result['model_results'].items():
            for model_name, model_data in task_results.items():
                if model_data['success']:
                    for metric, value in model_data['test_metrics'].items():
                        if isinstance(value, dict) and 'mean' in value and 'std' in value:
                            base_row[f'{task_name}-{model_name}-{metric}_mean'] = value['mean']
                            base_row[f'{task_name}-{model_name}-{metric}_std'] = value['std']
                        else:
                            # Handle legacy format where value is just a number
                            base_row[f'{task_name}-{model_name}-{metric}_mean'] = value
                            base_row[f'{task_name}-{model_name}-{metric}_std'] = 0.0
                    base_row[f'{task_name}-{model_name}-train_time'] = model_data['train_time']
        
        rows.append(base_row)
    
    return pd.DataFrame(rows)

def get_available_tasks_and_metrics(df):
    """Extract available tasks and their metrics from the dataframe columns."""
    tasks = {}
    
    # Find all task_model_metric columns
    for col in df.columns:
        if '-' in col and not col.startswith(('sweep_', 'random_', 'family_', 'consistency_', 'signal_')):
            parts = col.split('-')
            if len(parts) >= 3:
                # Handle label-specific metrics (e.g., k_hop_community_counts-gcn-mse-label-0)
                if 'label' in parts:
                    label_idx = parts.index('label')
                    # Everything before 'label' minus the last part (which is the base metric)
                    task_model_parts = parts[:label_idx-1]
                    metric_part = parts[label_idx-1]  # The base metric name
                else:
                    # Regular format: task-model-metric
                    task_model_parts = parts[:-1]  # Everything except the last part
                    metric_part = parts[-1]  # The metric name
                
                # Remove _mean or _std suffix to get base metric name
                if metric_part.endswith('_mean') or metric_part.endswith('_std'):
                    metric_part = metric_part[:-5]  # Remove _mean or _std
                
                # Normalize metric names to remove duplicates
                metric_part = metric_part.lower()  # Convert to lowercase
                if metric_part == 'f1_macr':
                    metric_part = 'f1_macro'
                elif metric_part == 'accurac':
                    metric_part = 'accuracy'
                
                if len(task_model_parts) >= 2:
                    # The task could be multi-word (e.g., k_hop_community_counts)
                    # We need to figure out where task ends and model begins
                    task = task_model_parts[0]  # First part is always the task
                    
                    if task not in tasks:
                        tasks[task] = set()
                    tasks[task].add(metric_part)
    
    # Convert sets to sorted lists
    for task in tasks:
        tasks[task] = sorted(list(tasks[task]))
    
    return tasks

def is_higher_better_metric(metric):
    """Determine if higher values are better for a given metric."""
    # Metrics where higher is better
    higher_better = {'accuracy', 'f1_macro', 'f1_micro', 'precision', 'recall', 'r2', 'auc', 'roc_auc'}
    # Metrics where lower is better
    lower_better = {'mse', 'rmse', 'mae', 'loss', 'error'}
    
    metric_lower = metric.lower()
    
    if any(hb in metric_lower for hb in higher_better):
        return True
    elif any(lb in metric_lower for lb in lower_better):
        return False
    else:
        # Default assumption: higher is better (can be customized)
        return True

def create_2d_parameter_manifold(df, param1, param2, model, task='community', metric='macro_f1', distance_data=None, shared_colorscale_range=None):
    """Create a 2D manifold plot showing model performance across parameter space."""
    if distance_data is not None:
        distance_df, distance_type = distance_data
        
        if distance_type == 'performance':
            # Performance distance: need to consider metric direction
            z_col = f'{task}-{model}-rank'
            z_values = distance_df[z_col].values
            title = f'{model.upper()}'
            colorbar_title = 'Performance Distance from Average'
            
            # For metrics where lower is better (like MAE, MSE), flip the sign
            # so that positive distance = better performance (above average)
            if not is_higher_better_metric(metric):
                z_values = -z_values  # Flip sign for lower-is-better metrics
            
            colorscale = 'RdBu'  # Red = good (positive), Blue = bad (negative)
        elif distance_type == 'ranking':
            z_col = f'{task}-{model}-rank'
            z_values = distance_df[z_col].values  # These are the actual rankings
            title = f'{model.upper()}'
            colorbar_title = 'Average Ranking'
            colorscale = 'RdYlBu'  # Lower rank (better) = red, higher rank (worse) = blue
        else:  # result values
            z_col = f'{task}-{model}-rank'
            z_values = distance_df[z_col].values  # These are the actual metric values
            title = f'{model.upper()}'
            colorbar_title = metric.capitalize()
            colorscale = 'RdBu'  # Red = good, Blue = bad
            
            # For lower-is-better metrics, flip the values for visualization
            # if not is_higher_better_metric(metric):
            #     z_values = -z_values
    else:
        # Use raw metric values (mean)
        z_col = f'{task}-{model}-{metric}_mean'
        z_values = df[z_col].values
        title = f'{model.upper()}'
        colorbar_title = metric.capitalize()
        colorscale = 'Viridis'
    
    x_values = df[param1].values
    y_values = df[param2].values
    
    fig = go.Figure()
    
    # Use shared colorscale range if provided
    if shared_colorscale_range is not None:
        zmin, zmax = shared_colorscale_range
    else:
        zmin, zmax = z_values.min(), z_values.max()
    
    # Only try interpolation if we have enough unique points (at least 4) and they're not all collinear
    unique_points = len(set(zip(x_values, y_values)))
    x_range = x_values.max() - x_values.min()
    y_range = y_values.max() - y_values.min()
    
    if unique_points >= 4 and x_range > 0 and y_range > 0:
        try:
            # Create a grid for interpolation
            x_min, x_max = x_values.min(), x_values.max()
            y_min, y_max = y_values.min(), y_values.max()
            
            # Add some padding to avoid edge effects
            x_padding = x_range * 0.1
            y_padding = y_range * 0.1
            
            xi = np.linspace(x_min - x_padding, x_max + x_padding, 40)
            yi = np.linspace(y_min - y_padding, y_max + y_padding, 40)
            xi_grid, yi_grid = np.meshgrid(xi, yi)
            
            # Try different interpolation methods
            zi = None
            for method in ['linear', 'nearest', 'cubic']:
                try:
                    zi = griddata((x_values, y_values), z_values, (xi_grid, yi_grid), 
                                method=method, fill_value=np.nan)
                    # Check if interpolation produced valid results
                    if not np.all(np.isnan(zi)):
                        break
                except:
                    continue
            
            # Add contour plot if interpolation was successful
            if zi is not None and not np.all(np.isnan(zi)):
                fig.add_trace(go.Contour(
                    x=xi,
                    y=yi,
                    z=zi,
                    colorscale=colorscale,
                    showscale=False,  # Hide colorbar for contour
                    opacity=0.6,
                    contours=dict(
                        showlabels=True,
                        labelfont=dict(size=8, color='white')
                    ),
                    name='Surface',
                    showlegend=False,
                    zmin=zmin,
                    zmax=zmax
                ))
        except Exception as e:
            # If interpolation fails, just show scatter plot
            pass
    
    # Add scatter points for actual data (always show these)
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='markers',
        marker=dict(
            size=10,
            color=z_values,
            colorscale=colorscale,
            line=dict(width=2, color='white'),
            showscale=False,  # We'll handle the colorbar separately
            cmin=zmin,
            cmax=zmax
        ),
        name='Data Points',
        showlegend=False,
        text=[f'{param1}: {x:.3f}<br>{param2}: {y:.3f}<br>{colorbar_title}: {z:.3f}' 
              for x, y, z in zip(x_values, y_values, z_values)],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        showlegend=False,
        width=400,
        height=400,
        margin=dict(t=50, b=50, l=50, r=50),
        xaxis_title=param1,
        yaxis_title=param2
    )
    
    return fig, z_values, colorbar_title, colorscale

def calculate_sampling_rank_distributions(df, task, metric, n_samples=10000):
    """Calculate ranking distributions via sampling from model score distributions."""
    from scipy.stats import norm

    model_columns = [col for col in df.columns if f'{task}-' in col and f'-{metric}_mean' in col and 'train_time' not in col and 'label' not in col]
    std_columns = [col.replace('_mean', '_std') for col in model_columns]
    model_names = [col.replace(f'{task}-', '').replace(f'-{metric}_mean', '') for col in model_columns]

    df_clean = df[model_columns + std_columns].dropna()
    if df_clean.empty:
        st.error("❌ No complete runs with mean+std values found.")
        return None, None

    # Get numpy arrays of means and stds
    means = df_clean[model_columns].values  # shape: (n_runs, n_models)
    stds = df_clean[std_columns].values     # shape: (n_runs, n_models)

    higher_better = is_higher_better_metric(metric)
    all_ranks = []

    # Sample-based ranking
    for run_idx in range(len(means)):
        run_means = means[run_idx]
        run_stds = stds[run_idx]

        # Draw samples from each model's distribution
        sampled_scores = np.random.normal(loc=run_means[:, None], scale=run_stds[:, None], size=(len(run_means), n_samples))

        # FIXED: Use rankdata instead of argsort to get proper ranks
        if higher_better:
            # For higher-is-better metrics, we want to rank in descending order
            # rankdata with method='min' gives rank 1 to the highest value when we negate
            ranks = np.apply_along_axis(lambda x: rankdata(-x, method='min'), axis=0, arr=sampled_scores)
        else:
            # For lower-is-better metrics, rank in ascending order  
            # rankdata gives rank 1 to the lowest value
            ranks = np.apply_along_axis(lambda x: rankdata(x, method='min'), axis=0, arr=sampled_scores)

        all_ranks.append(ranks)

    # Stack ranks: shape (n_runs * n_models, n_samples)
    all_ranks = np.stack(all_ranks)  # shape: (n_runs, n_models, n_samples)
    model_ranks_all = all_ranks.reshape(-1, all_ranks.shape[-1])  # flatten across runs

    # Average rank distribution per model
    avg_ranks = model_ranks_all.mean(axis=1).reshape(len(df_clean), -1).mean(axis=0)
    rank_distributions = model_ranks_all.reshape(len(df_clean), len(model_names), n_samples)

    return model_names, rank_distributions


def calculate_model_rankings(df, task='community', metric='accuracy'):
    """Calculate model rankings for each run and overall averages."""
    # Use mean values for ranking
    model_columns = [col for col in df.columns if f'{task}-' in col and f'-{metric}_mean' in col and 'train_time' not in col and 'label' not in col]
    model_names = [col.replace(f'{task}-', '').replace(f'-{metric}_mean', '') for col in model_columns]
    
    # Filter out runs that have NaN values for any model in this task/metric
    df_clean = df[model_columns].copy()
    
    # Remove rows with any NaN values
    initial_count = len(df_clean)
    df_clean = df_clean.dropna()
    final_count = len(df_clean)
    
    if final_count < initial_count:
        st.warning(f"⚠️ Filtered out {initial_count - final_count} runs with NaN values. Using {final_count} complete runs.")
    
    if len(df_clean) == 0:
        st.error("❌ No complete runs found for the selected task/metric combination!")
        return None, None, None, None, None
    
    # Per-run rankings (1 = best)
    ranking_df = df_clean.copy()
    higher_better = is_higher_better_metric(metric)
    
    for idx, row in ranking_df.iterrows():
        if higher_better:
            # Higher metric = better = lower rank number
            ranks = rankdata(-row.values, method='min')
        else:
            # Lower metric = better = lower rank number  
            ranks = rankdata(row.values, method='min')
        ranking_df.loc[idx] = ranks
    
    # Rename columns to indicate rankings
    ranking_df.columns = [f'{task}-{name}-rank' for name in model_names]
    
    # Calculate average rankings
    avg_rankings = ranking_df.mean()
    
    # Calculate distance from average for each run - but for PERFORMANCE, not ranking
    # We want to show how much better/worse each model performed compared to average
    performance_distance_df = pd.DataFrame(index=df_clean.index)
    avg_performance_per_run = df_clean.mean(axis=1)  # Average performance across models for each run
    
    for i, model_name in enumerate(model_names):
        original_col = model_columns[i]
        model_performance = df_clean[original_col]
        
        # Calculate distance from run average
        distance = model_performance - avg_performance_per_run
        
        # For lower-is-better metrics, we need to flip the interpretation
        # A positive distance means the model performed worse than average (higher value)
        # A negative distance means the model performed better than average (lower value)
        # But we want positive to mean "better than average" for visualization
        if not higher_better:
            distance = -distance  # Flip for lower-is-better metrics
        
        performance_distance_df[f'{task}-{model_name}-rank'] = distance
    
    # Return the indices of clean runs so we can filter the main dataframe too
    clean_indices = df_clean.index
    
    return ranking_df, performance_distance_df, avg_rankings, model_names, clean_indices

def calculate_multi_task_rankings(df, task_metric_pairs, use_sampling=False, n_samples=10000):
    """Calculate rankings across multiple tasks and metrics."""
    all_rankings = {}
    all_std_rankings = {}
    all_model_names = set()
    
    for task, metric in task_metric_pairs:
        if use_sampling:
            result = calculate_sampling_rank_distributions(df, task, metric, n_samples)
            if result and result[0] is not None:
                model_names, rank_distributions = result
                avg_ranks = rank_distributions.mean(axis=-1).mean(axis=0)
                std_ranks = rank_distributions.std(axis=-1).mean(axis=0)
                all_rankings[f"{task}-{metric}"] = dict(zip(model_names, avg_ranks))
                all_std_rankings[f"{task}-{metric}"] = dict(zip(model_names, std_ranks))
                all_model_names.update(model_names)
        else:
            result = calculate_model_rankings(df, task, metric)
            if result[0] is not None:
                ranking_df, _, avg_rankings, model_names, _ = result
                all_rankings[f"{task}-{metric}"] = avg_rankings.to_dict()
                # Clean up the dictionary keys to remove the task-model-rank format
                clean_rankings = {}
                for key, value in avg_rankings.items():
                    model_name = key.replace(f'{task}-', '').replace('-rank', '')
                    clean_rankings[model_name] = value
                all_rankings[f"{task}-{metric}"] = clean_rankings
                all_std_rankings[f"{task}-{metric}"] = {model: 0.0 for model in clean_rankings.keys()}  # No std for non-sampling
                all_model_names.update(clean_rankings.keys())
    
    return all_rankings, sorted(list(all_model_names)), all_std_rankings

def create_multi_task_ranking_plot(all_rankings, all_model_names, all_std_rankings=None):
    """Create a grouped bar plot for multi-task rankings with error bars."""
    # Prepare data for plotting
    plot_data = []
    
    for task_metric, rankings in all_rankings.items():
        for model in all_model_names:
            if model in rankings:
                # Convert rank to inverse for better visualization (higher bar = better rank)
                # Since rank 1 is best, we use max_rank + 1 - rank to invert
                max_rank = max(rankings.values()) if rankings.values() else 1
                inverted_rank = max_rank + 1 - rankings[model]
                
                # Get standard deviation if available
                std_rank = 0.0
                if all_std_rankings and task_metric in all_std_rankings and model in all_std_rankings[task_metric]:
                    std_rank = all_std_rankings[task_metric][model]
                
                plot_data.append({
                    'Model': model.upper(),
                    'Task-Metric': task_metric,
                    'Rank': rankings[model],
                    'Inverted_Rank': inverted_rank,
                    'Std_Rank': std_rank
                })
    
    df_plot = pd.DataFrame(plot_data)
    
    if df_plot.empty:
        return None
    
    # Create grouped bar plot with error bars
    fig = px.bar(
        df_plot,
        x='Model',
        y='Inverted_Rank',
        color='Task-Metric',
        title='Model Rankings Across Tasks (Higher bars = Better rank)',
        labels={'Inverted_Rank': 'Rank Score', 'Model': 'Model'},
        text='Rank',  # Show actual rank numbers on bars
        barmode='group',
        error_y='Std_Rank'  # Add error bars using standard deviation
    )
    
    # Update text position and format
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    
    # Use custom y-axis to show actual rank numbers with 1 at top
    max_rank_multi = max([max(rankings.values()) for rankings in all_rankings.values()]) if all_rankings else 1
    
    fig.update_layout(
        height=500,
        xaxis_title='Model',
        yaxis_title='Rank (1 = Best)',
        yaxis=dict(
            tickvals=list(range(1, int(max_rank_multi) + 1)),
            ticktext=[str(i) for i in range(int(max_rank_multi), 0, -1)]  # Show 1 at top
        ),
        legend_title='Task-Metric'
    )
    
    return fig
       
def get_models_for_task(df, task):
    """Get all model names for a specific task."""
    models = set()
    
    # Find all columns that match this task
    for col in df.columns:
        if col.startswith(f'{task}-') and not col.endswith('train_time'):
            # Remove the task prefix
            remaining = col[len(task)+1:]  # +1 for the hyphen
            
            # Split the remaining part
            parts = remaining.split('-')
            
            # Handle label-specific metrics (e.g., gcn-mse-label-0)
            if 'label' in parts:
                label_idx = parts.index('label')
                # Model is everything before the metric (which is before 'label')
                model_parts = parts[:label_idx-1]
            else:
                # Regular format: model-metric
                model_parts = parts[:-1]  # Everything except the last part (metric)
            
            if model_parts:
                model = '-'.join(model_parts) if len(model_parts) > 1 else model_parts[0]
                models.add(model)
    
    return sorted(list(models))
        
# Main Streamlit App
def main():
    st.title("🧠 GraphUniverse Benchmarking Visualization")
    
    # Sidebar for file selection
    st.sidebar.header("📁 Data Selection")
    
    # Find available experiment files
    experiments_dir = Path("multi_results")
    if experiments_dir.exists():
        experiment_folders = [d for d in experiments_dir.iterdir() if d.is_dir()]
        experiment_names = [d.name for d in experiment_folders]
        
        if experiment_names:
            selected_experiment = st.sidebar.selectbox("Select Experiment:", experiment_names)
            
            # Try to load final_results.json first, fall back to intermediate_results.json
            json_file = experiments_dir / selected_experiment / "final_results.json"
            is_intermediate = False
            if not json_file.exists():
                json_file = experiments_dir / selected_experiment / "intermediate_results.json"
                if json_file.exists():
                    is_intermediate = True
                    st.sidebar.warning("⚠️ Using intermediate results (final results not found)")
                else:
                    st.error(f"No results file found in {selected_experiment}")
                    return
            
            data = load_experiment_data(json_file)
            df = process_results_to_dataframe(data)
            
            # Get available tasks and metrics
            available_tasks_metrics = get_available_tasks_and_metrics(df)
            
            # Display basic info
            st.sidebar.success(f"✅ Loaded {len(df)} successful runs")
            
            # Only show summary stats if they exist (final results)
            if not is_intermediate and 'summary_stats' in data:
                st.sidebar.info(f"📊 Total runs attempted: {data['summary_stats']['total_runs_attempted']}")
                st.sidebar.info(f"⏱️ Success rate: {data['summary_stats']['success_rate']:.1%}")
            else:
                st.sidebar.info("ℹ️ Using intermediate results - summary statistics not available")
            
            st.sidebar.info(f"🎯 Available tasks: {', '.join(available_tasks_metrics.keys())}")
            
        else:
            st.error("No experiment folders found in experiments/multi_results/")
            return
    else:
        st.error("experiments/multi_results/ directory not found")
        return
    
    # Get available parameters
    param_columns = [col for col in df.columns if col.startswith(('sweep_', 'random_', 'family_', 'consistency_', 'signal_'))]
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📈 Basic Plots", "🏆 Model Rankings", "🏆 Multi-Task Rankings", "📊 Summary Stats", "🔍 Data Explorer", "🎯 Hyperparameter Analysis", "🗺️ Parameter Manifolds"])
    
    with tab1:
        st.header("📈 Enhanced Statistical Analysis")
        
        # Add toggle for enhanced vs basic view
        analysis_mode = st.radio(
            "Analysis Mode:",
            ["Basic Plotting", "Statistical Analysis with Uncertainty"],
            index=1,  # Default to enhanced
            horizontal=True
        )
        
        if analysis_mode == "Statistical Analysis with Uncertainty":
            # Enhanced statistical analysis
            enhanced_basic_plots_tab(df, available_tasks_metrics, param_columns)
            
        else:
            # Original basic plotting code
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                x_param = st.selectbox("X-axis Parameter:", param_columns, key="basic_x")
                
            with col2:
                selected_task = st.selectbox("Task:", list(available_tasks_metrics.keys()), key="basic_task")
                
            with col3:
                if selected_task in available_tasks_metrics:
                    y_metric = st.selectbox("Y-axis Metric:", available_tasks_metrics[selected_task], key="basic_y")
                else:
                    st.error("No metrics available for selected task")
                    return
            with col4:
                model_names = get_models_for_task(df, selected_task)
                selected_models = st.multiselect("Models to Plot:", model_names, default=model_names[:3], key="basic_models")
            
        if selected_models:
            # Filter out runs with NaN values for the selected models
            relevant_columns = []
            for model in selected_models:
                model_col_mean = f'{selected_task}-{model}-{y_metric}_mean'
                model_col_std = f'{selected_task}-{model}-{y_metric}_std'
                if model_col_mean in df.columns and model_col_std in df.columns:
                    relevant_columns.extend([model_col_mean, model_col_std])
            
            # Only use runs that have complete data for all selected models
            df_plot = df[relevant_columns + [x_param]].dropna()
            
            if len(df_plot) < len(df):
                st.info(f"ℹ️ Using {len(df_plot)} runs with complete data (filtered out {len(df) - len(df_plot)} runs with NaN values)")
            
            if len(df_plot) == 0:
                st.error("❌ No complete data found for the selected models and metric!")
            else:
                fig = go.Figure()
                
                for model in selected_models:
                    model_col_mean = f'{selected_task}-{model}-{y_metric}_mean'
                    model_col_std = f'{selected_task}-{model}-{y_metric}_std'
                    if model_col_mean in df_plot.columns and model_col_std in df_plot.columns:
                        fig.add_trace(go.Scatter(
                            x=df_plot[x_param],
                            y=df_plot[model_col_mean],
                            error_y=dict(
                                type='data',
                                array=df_plot[model_col_std],
                                visible=True
                            ),
                            mode='markers',
                            name=model.upper(),
                            marker=dict(size=8, opacity=0.7)
                        ))
                
                fig.update_layout(
                    title=f"{selected_task} - {y_metric} vs {x_param}",
                    xaxis_title=x_param,
                    yaxis_title=y_metric,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Model Rankings Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            task = st.selectbox("Task:", list(available_tasks_metrics.keys()), key="rank_task")
        
        with col2:
            if task in available_tasks_metrics:
                metric = st.selectbox("Metric:", available_tasks_metrics[task], key="rank_metric")
            else:
                st.error("No metrics available for selected task")
                return
        
        use_sampling = st.checkbox("Use Sampling-Based Ranking (Monte Carlo)", value=False)

        # Initialize variables
        ranking_df = None
        distance_df = None
        avg_rankings = None
        model_names = None
        clean_indices = None

        if use_sampling:
            result = calculate_sampling_rank_distributions(df, task, metric)
            if result and result[0] is not None:
                model_names, rank_distributions = result

                st.subheader("Expected Rank ± Std over Sampling")

                avg_ranks = rank_distributions.mean(axis=-1).mean(axis=0)
                std_ranks = rank_distributions.std(axis=-1).mean(axis=0)

                # Use inverted ranks so higher bars = better rank (rank 1 = highest bar)
                max_rank = len(model_names)
                inverted_ranks = max_rank + 1 - avg_ranks

                fig = go.Figure(data=[
                    go.Bar(
                        x=model_names,
                        y=inverted_ranks,
                        error_y=dict(type='data', array=std_ranks, visible=True),
                        text=[f'{r:.2f}±{s:.2f}' for r, s in zip(avg_ranks, std_ranks)],
                        textposition='outside'
                    )
                ])
                fig.update_layout(
                    title="Expected Model Rankings from Monte Carlo Sampling",
                    xaxis_title="Model",
                    yaxis_title="Rank (1 = Best)",
                    yaxis=dict(
                        tickvals=list(range(1, max_rank + 1)),
                        ticktext=[str(i) for i in range(max_rank, 0, -1)]  # Show 1 at top
                    ),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            result = calculate_model_rankings(df, task, metric)
            if result[0] is not None:
                ranking_df, distance_df, avg_rankings, model_names, clean_indices = result

        if model_names is not None:
            # Average rankings bar chart (only show if we have data)
            if not use_sampling and avg_rankings is not None:
                # Use inverted ranks so higher bars = better rank (rank 1 = highest bar)
                max_rank = len(model_names)
                inverted_ranks = max_rank + 1 - avg_rankings.values

                fig_avg = go.Figure(data=[
                    go.Bar(
                        x=model_names,
                        y=inverted_ranks,
                        text=[f'{val:.2f}' for val in avg_rankings.values],
                        textposition='outside'
                    )
                ])
                fig_avg.update_layout(
                    title=f"Average Model Rankings ({metric}) - Higher bars = Better rank",
                    xaxis_title="Model",
                    yaxis_title="Rank (1 = Best)",
                    yaxis=dict(
                        tickvals=list(range(1, max_rank + 1)),
                        ticktext=[str(i) for i in range(max_rank, 0, -1)]  # Show 1 at top
                    ),
                    height=400
                )
                
                st.plotly_chart(fig_avg, use_container_width=True)
            
            # Show which direction is better for this metric
            higher_better = is_higher_better_metric(metric)
            if higher_better:
                st.info(f"ℹ️ For {metric}: Higher values are better")
            else:
                st.info(f"ℹ️ For {metric}: Lower values are better")
            
            # Rankings distribution (only for non-sampling method)
            if not use_sampling and ranking_df is not None:
                st.subheader("Ranking Distributions")
                
                cols = st.columns(len(model_names))
                for i, model in enumerate(model_names):
                    with cols[i]:
                        rank_col = f'{task}-{model}-rank'
                        fig_dist = px.histogram(
                            ranking_df, 
                            x=rank_col,
                            title=f"{model.upper()}<br>Rank Distribution",
                            nbins=int(max(ranking_df[rank_col])) if max(ranking_df[rank_col]) > 1 else 5
                        )
                        fig_dist.update_layout(height=300, showlegend=False)
                        st.plotly_chart(fig_dist, use_container_width=True)

    with tab3:
        st.header("🏆 Multi-Task Rankings Analysis")
        st.write("📊 Comprehensive ranking analysis across all available tasks using Monte Carlo simulation")
        
        # Automatically determine task-metric pairs for all available tasks
        auto_task_metric_pairs = []
        for task in available_tasks_metrics.keys():
            if task == 'community' and 'f1_macro' in available_tasks_metrics[task]:
                auto_task_metric_pairs.append((task, 'f1_macro'))
            elif 'mse' in available_tasks_metrics[task]:
                auto_task_metric_pairs.append((task, 'mse'))
            elif available_tasks_metrics[task]:  # Fallback to first available metric
                auto_task_metric_pairs.append((task, available_tasks_metrics[task][0]))
        
        if auto_task_metric_pairs:
            st.info(f"🎯 **Tasks to analyze:** {', '.join([f'{task} ({metric})' for task, metric in auto_task_metric_pairs])}")
            
            # Show progress and run analysis
            if st.button("🚀 Start Multi-Task Analysis", type="primary"):
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_tasks = len(auto_task_metric_pairs)
                all_rankings = {}
                all_std_rankings = {}
                all_model_names = set()
                
                # Process each task with progress updates
                for i, (task, metric) in enumerate(auto_task_metric_pairs):
                    status_text.text(f"🔄 Analyzing {task} ({metric})... ({i+1}/{total_tasks})")
                    progress_bar.progress((i) / total_tasks)
                    
                    result = calculate_sampling_rank_distributions(df, task, metric, n_samples=10000)
                    if result and result[0] is not None:
                        model_names, rank_distributions = result
                        avg_ranks = rank_distributions.mean(axis=-1).mean(axis=0)
                        std_ranks = rank_distributions.std(axis=-1).mean(axis=0)
                        all_rankings[f"{task}-{metric}"] = dict(zip(model_names, avg_ranks))
                        all_std_rankings[f"{task}-{metric}"] = dict(zip(model_names, std_ranks))
                        all_model_names.update(model_names)
                
                # Complete progress
                progress_bar.progress(1.0)
                status_text.text("✅ Analysis complete!")
                
                # Store results in session state to persist them
                st.session_state['multi_task_results'] = {
                    'all_rankings': all_rankings,
                    'all_std_rankings': all_std_rankings,
                    'all_model_names': sorted(list(all_model_names)),
                    'task_metric_pairs': auto_task_metric_pairs
                }
            
            # Display results if available
            if 'multi_task_results' in st.session_state:
                results = st.session_state['multi_task_results']
                all_rankings = results['all_rankings']
                all_std_rankings = results['all_std_rankings']
                all_model_names = results['all_model_names']
                
                if all_rankings and all_model_names:
                    st.success("🎉 Multi-task analysis completed!")
                    
                    # Create multi-task ranking plot with error bars
                    multi_fig = create_multi_task_ranking_plot(all_rankings, all_model_names, all_std_rankings)
                    
                    if multi_fig is not None:
                        st.plotly_chart(multi_fig, use_container_width=True)
                        
                        # Show summary table
                        st.subheader("📋 Detailed Ranking Summary")
                        
                        # Create summary DataFrame with mean ± std
                        summary_data = []
                        for model in all_model_names:
                            row = {'Model': model.upper()}
                            for task_metric in results['task_metric_pairs']:
                                task_metric_key = f"{task_metric[0]}-{task_metric[1]}"
                                if model in all_rankings.get(task_metric_key, {}):
                                    mean_rank = all_rankings[task_metric_key][model]
                                    std_rank = all_std_rankings[task_metric_key][model] if task_metric_key in all_std_rankings and model in all_std_rankings[task_metric_key] else 0
                                    row[task_metric_key] = f"{mean_rank:.2f}±{std_rank:.2f}"
                                else:
                                    row[task_metric_key] = "N/A"
                            summary_data.append(row)
                        
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True)
                        
                        # Download button for results
                        csv = summary_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Multi-Task Results as CSV",
                            data=csv,
                            file_name=f"{selected_experiment}_multi_task_rankings.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("Could not create multi-task ranking plot - insufficient data.")
                else:
                    st.warning("No ranking data available.")
            else:
                st.write("👆 Click the button above to start the analysis")
                
        else:
            st.warning("❌ No suitable task-metric combinations found automatically.")

    with tab4:
        st.header("Summary Statistics")
        
        col1, col2 = st.columns([1, 1])  # Explicitly specify column widths
        
        with col1:
            st.subheader("Experiment Overview")
            if 'summary_stats' in data:
                st.json({
                    "Total Runs": data['summary_stats']['total_runs_attempted'],
                    "Successful Runs": data['summary_stats']['successful_runs'],
                    "Failed Runs": data['summary_stats']['failed_runs'],
                    "Success Rate": f"{data['summary_stats']['success_rate']:.1%}",
                    "Total Time": f"{data['summary_stats']['total_time']:.1f}s",
                    "Avg Time per Run": f"{data['summary_stats']['avg_time_per_run']:.1f}s"
                })
            else:
                # Calculate basic stats from the dataframe
                total_runs = len(df)
                successful_runs = len(df[df['total_time'] > 0])  # Assuming runs with time > 0 are successful
                failed_runs = total_runs - successful_runs
                success_rate = successful_runs / total_runs if total_runs > 0 else 0
                total_time = df['total_time'].sum()
                avg_time = total_time / successful_runs if successful_runs > 0 else 0
                
                st.json({
                    "Total Runs": total_runs,
                    "Successful Runs": successful_runs,
                    "Failed Runs": failed_runs,
                    "Success Rate": f"{success_rate:.1%}",
                    "Total Time": f"{total_time:.1f}s",
                    "Avg Time per Run": f"{avg_time:.1f}s"
                })
        
        with col2:
            st.subheader("Parameter Ranges")
            param_stats = df[param_columns].describe()
            st.dataframe(param_stats)
        
        # Correlation heatmap
        st.subheader("Parameter-Metric Correlations")
        
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                title="Correlation Heatmap",
                color_continuous_scale='RdBu',
                aspect="auto"
            )
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab5:
        st.header("Data Explorer")
        
        # Filtering options
        st.subheader("Filter Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            method_filter = st.multiselect("Method:", df['method'].unique(), default=df['method'].unique())
        with col2:
            degree_dist_filter = st.multiselect("Degree Distribution:", df['degree_distribution'].unique(), 
                                               default=df['degree_distribution'].unique())
        with col3:
            min_graphs = st.slider("Min Graphs per Family:", 
                                 int(df['n_graphs'].min()), 
                                 max(int(df['n_graphs'].max()), int(df['n_graphs'].min()) + 1),
                                 int(df['n_graphs'].min()))
        
        # Apply filters
        filtered_df = df[
            (df['method'].isin(method_filter)) &
            (df['degree_distribution'].isin(degree_dist_filter)) &
            (df['n_graphs'] >= min_graphs)
        ]
        
        st.subheader(f"Filtered Data ({len(filtered_df)} runs)")
        
        # Show data table
        all_metric_columns = []
        for task, metrics in available_tasks_metrics.items():
            for metric in metrics:
                task_model_columns = [col for col in df.columns if f'{task}-' in col and f'-{metric}' in col and 'train_time' not in col and 'label' not in col]
                all_metric_columns.extend(task_model_columns)
        
        display_columns = st.multiselect(
            "Select Columns to Display:",
            df.columns.tolist(),
            default=['run_id', 'method', 'degree_distribution', 'n_graphs'] + all_metric_columns[:5]
        )
        
        if display_columns:
            st.dataframe(filtered_df[display_columns])
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name=f"{selected_experiment}_filtered_results.csv",
            mime="text/csv"
        )

    with tab6:
        st.header("Hyperparameter Analysis")
        
        # Get all models that have hyperparameter optimization results
        models_with_hyperopt = []
        for result in data['all_results']:
            for task_name, task_results in result['model_results'].items():
                for model_name, model_data in task_results.items():
                    if model_data.get('success') and model_data.get('optimal_hyperparams'):
                        if (task_name, model_name) not in models_with_hyperopt:
                            models_with_hyperopt.append((task_name, model_name))
        
        if not models_with_hyperopt:
            st.warning("No hyperparameter optimization results found in the data.")
        else:
            # Model selection
            col1, col2 = st.columns(2)
            with col1:
                available_tasks_hyperopt = list(set([m[0] for m in models_with_hyperopt]))
                selected_task = st.selectbox("Task:", available_tasks_hyperopt)
            with col2:
                available_models_for_task = [m[1] for m in models_with_hyperopt if m[0] == selected_task]
                selected_model = st.selectbox("Model:", available_models_for_task)
            
            # Get all hyperparameters for the selected model
            hyperparams = set()
            available_metrics = set()
            for result in data['all_results']:
                if selected_task in result['model_results'] and selected_model in result['model_results'][selected_task]:
                    model_data = result['model_results'][selected_task][selected_model]
                    if model_data.get('success') and model_data.get('optimal_hyperparams'):
                        hyperparams.update(model_data['optimal_hyperparams'].keys())
                        if 'test_metrics' in model_data:
                            available_metrics.update(model_data['test_metrics'].keys())
            
            if not hyperparams:
                st.warning(f"No hyperparameter optimization results found for {selected_model} on {selected_task} task.")
            else:
                # Create two columns for selection
                col1, col2 = st.columns(2)
                
                with col1:
                    # Hyperparameter selection
                    selected_hyperparam = st.selectbox("Hyperparameter:", sorted(list(hyperparams)))
                
                with col2:
                    # Performance metric selection
                    if available_metrics:
                        performance_metric = st.selectbox("Performance Metric:", sorted(list(available_metrics)))
                    else:
                        st.warning("No performance metrics available for this model/task combination.")
                        return
                
                # Collect hyperparameter values and corresponding performance
                hyperparam_values = []
                performance_values = []
                sweep_params = []
                
                for result in data['all_results']:
                    if selected_task in result['model_results'] and selected_model in result['model_results'][selected_task]:
                        model_data = result['model_results'][selected_task][selected_model]
                        if model_data.get('success') and model_data.get('optimal_hyperparams'):
                            if selected_hyperparam in model_data['optimal_hyperparams']:
                                hyperparam_values.append(model_data['optimal_hyperparams'][selected_hyperparam])
                                # Use the selected performance metric
                                if performance_metric in model_data['test_metrics']:
                                    metric_value = model_data['test_metrics'][performance_metric]
                                    if isinstance(metric_value, dict) and 'mean' in metric_value:
                                        performance_values.append(metric_value['mean'])
                                    else:
                                        performance_values.append(metric_value)
                                else:
                                    # If the performance metric isn't available, skip this run
                                    continue
                                # Store sweep parameters for correlation analysis
                                sweep_params.append(result['sweep_parameters'])
                
                if not hyperparam_values:
                    st.warning(f"No values found for {selected_hyperparam}.")
                else:
                    # Create two columns for plots
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Hyperparameter Distribution")
                        
                        # Check if hyperparameter is numerical or categorical
                        is_numerical = all(isinstance(x, (int, float)) for x in hyperparam_values)
                        
                        if is_numerical:
                            # Create histogram for numerical values
                            fig = px.histogram(
                                x=hyperparam_values,
                                title=f"Distribution of {selected_hyperparam}",
                                labels={'x': selected_hyperparam, 'y': 'Count'},
                                nbins=20
                            )
                        else:
                            # Create bar plot for categorical values
                            value_counts = pd.Series(hyperparam_values).value_counts()
                            fig = px.bar(
                                x=value_counts.index,
                                y=value_counts.values,
                                title=f"Distribution of {selected_hyperparam}",
                                labels={'x': selected_hyperparam, 'y': 'Count'}
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader(f"Performance vs Hyperparameter")
                        
                        if is_numerical:
                            # Scatter plot for numerical values
                            fig = px.scatter(
                                x=hyperparam_values,
                                y=performance_values,
                                title=f"{performance_metric.upper()} vs {selected_hyperparam}",
                                labels={'x': selected_hyperparam, 'y': performance_metric.upper()},
                                trendline="ols"
                            )
                        else:
                            # Box plot for categorical values
                            df_plot = pd.DataFrame({
                                selected_hyperparam: hyperparam_values,
                                performance_metric.upper(): performance_values
                            })
                            fig = px.box(
                                df_plot,
                                x=selected_hyperparam,
                                y=performance_metric.upper(),
                                title=f"{performance_metric.upper()} by {selected_hyperparam}"
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Parameter correlation analysis
                    st.subheader("Parameter Correlation Analysis")
                    
                    # Select a sweep parameter to analyze correlation with
                    if sweep_params:
                        sweep_param_cols = st.multiselect(
                            "Select sweep parameters to analyze:",
                            list(sweep_params[0].keys()),
                            default=list(sweep_params[0].keys())[:2] if sweep_params else []
                        )
                        
                        if sweep_param_cols:
                            # Create correlation plots
                            cols = st.columns(len(sweep_param_cols))
                            
                            for i, param in enumerate(sweep_param_cols):
                                with cols[i]:
                                    param_values = [p[param] for p in sweep_params]
                                    
                                    if is_numerical:
                                        # Scatter plot for numerical values
                                        fig = px.scatter(
                                            x=param_values,
                                            y=hyperparam_values,
                                            title=f"{selected_hyperparam} vs {param}",
                                            labels={'x': param, 'y': selected_hyperparam},
                                            trendline="ols"
                                        )
                                    else:
                                        # Box plot for categorical values
                                        df_plot = pd.DataFrame({
                                            param: param_values,
                                            selected_hyperparam: hyperparam_values
                                        })
                                        fig = px.box(
                                            df_plot,
                                            x=param,
                                            y=selected_hyperparam,
                                            title=f"{selected_hyperparam} by {param}"
                                        )
                                    
                                    st.plotly_chart(fig, use_container_width=True)

    with tab7:
        st.header("Parameter Space Manifolds")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            param1 = st.selectbox("Parameter 1 (X-axis):", param_columns, key="manifold_x")
        with col2:
            param2 = st.selectbox("Parameter 2 (Y-axis):", param_columns, key="manifold_y", 
                                index=min(1, len(param_columns)-1))
        with col3:
            viz_type = st.selectbox("Visualization Type:", 
                                ["Performance Distance from Average", "Average Ranking", "Result Values"], 
                                key="viz_type")
            manifold_task = st.selectbox("Task:", list(available_tasks_metrics.keys()), key="manifold_task")
            
            if manifold_task in available_tasks_metrics:
                manifold_metric = st.selectbox("Metric:", available_tasks_metrics[manifold_task], key="manifold_metric")
            else:
                st.error("No metrics available for selected task")
                return
        
        if param1 != param2:
            result = calculate_model_rankings(df, manifold_task, manifold_metric)
            if result[0] is None:  # No complete runs found
                st.error("Cannot create manifolds - no complete data available for the selected task/metric.")
            else:
                ranking_df, performance_distance_df, avg_rankings, model_names, clean_indices = result
                
                # Filter the main dataframe to only include clean runs
                df_clean = df.loc[clean_indices]
                
                # Calculate the appropriate distance data
                if viz_type == "Performance Distance from Average":
                    distance_data = (performance_distance_df, 'performance')
                elif viz_type == "Average Ranking":
                    distance_data = (ranking_df, 'ranking')
                else:  # Result Values
                    # Create a new DataFrame with just the metric values
                    result_values_df = pd.DataFrame(index=df_clean.index)
                    for model in model_names:
                        metric_col = f'{manifold_task}-{model}-{manifold_metric}_mean'
                        result_values_df[f'{manifold_task}-{model}-rank'] = df_clean[metric_col]
                    distance_data = (result_values_df, 'result')
                
                # Calculate shared colorscale range across all models
                all_z_values = []
                for model in model_names:
                    if viz_type == "Performance Distance from Average":
                        z_col = f'{manifold_task}-{model}-rank'
                        z_values = performance_distance_df[z_col].values
                        if not is_higher_better_metric(manifold_metric):
                            z_values = -z_values  # Apply the same flip as in the plotting function
                    elif viz_type == "Average Ranking":
                        z_col = f'{manifold_task}-{model}-rank'
                        z_values = ranking_df[z_col].values
                    else:  # Result Values
                        # Use actual metric values for coloring
                        metric_col = f'{manifold_task}-{model}-{manifold_metric}_mean'
                        z_values = df_clean[metric_col].values
                        # For lower-is-better metrics, we want to flip the values for visualization
                        # so that red = better (lower value) and blue = worse (higher value)
                        # if not is_higher_better_metric(manifold_metric):
                        #     z_values = -z_values
                    all_z_values.extend(z_values)
                
                shared_colorscale_range = (max(all_z_values), min(all_z_values))
                
                # Create subplot grid
                n_models = len(model_names)
                n_cols = int(np.ceil(np.sqrt(n_models)))
                n_rows = int(np.ceil(n_models / n_cols))
                
                subplot_titles = [f"{model.upper()}" for model in model_names]
                
                fig = make_subplots(
                    rows=n_rows,
                    cols=n_cols,
                    subplot_titles=subplot_titles,
                    vertical_spacing=0.15,
                    horizontal_spacing=0.1,
                    shared_xaxes=True,
                    shared_yaxes=True
                )
                
                # Store colorbar info from first plot
                colorbar_title = None
                colorscale = None
                
                for i, model in enumerate(model_names):
                    row = (i // n_cols) + 1
                    col = (i % n_cols) + 1
                    
                    # Create individual manifold
                    individual_fig, z_values, cb_title, cs = create_2d_parameter_manifold(
                        df_clean, param1, param2, model, manifold_task, manifold_metric,
                        distance_data, shared_colorscale_range
                    )
                    
                    # Store colorbar info from first plot
                    if i == 0:
                        colorbar_title = cb_title
                        colorscale = cs
                    
                    # Add traces to subplot
                    for trace in individual_fig.data:
                        trace.showlegend = False
                        fig.add_trace(trace, row=row, col=col)
                
                # Add a single shared colorbar
                # Create a dummy trace for the colorbar
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(
                        colorscale=colorscale,
                        showscale=True,
                        cmin=shared_colorscale_range[0],
                        cmax=shared_colorscale_range[1],
                        colorbar=dict(
                            title=colorbar_title,
                            titleside='right',
                            x=1.02,
                            len=0.8
                        )
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Update layout
                fig.update_layout(
                    title=viz_type,
                    title_x=0.5,  # Center the title
                    height=400 * n_rows,
                    showlegend=False
                )
                
                # Add shared axis titles only once in the middle
                # X-axis title only on the bottom row, middle column
                middle_col = (n_cols + 1) // 2
                fig.update_xaxes(title_text=param1, row=n_rows, col=middle_col)
                
                # Y-axis title only on the leftmost column, middle row
                middle_row = (n_rows + 1) // 2
                fig.update_yaxes(title_text=param2, row=middle_row, col=1)
                
                # Show metric direction info
                higher_better = is_higher_better_metric(manifold_metric)
                if viz_type == "Performance Distance from Average":
                    if higher_better:
                        st.info(f"ℹ️ For {manifold_metric}: Red = above average (better), Blue = below average (worse)")
                    else:
                        st.info(f"ℹ️ For {manifold_metric}: Red = below average (better), Blue = above average (worse)")
                elif viz_type == "Average Ranking":
                    st.info("ℹ️ Rankings: Red = better rank (lower number), Blue = worse rank (higher number)")
                else:  # Result Values
                    if higher_better:
                        st.info(f"ℹ️ For {manifold_metric}: Red = higher values (better), Blue = lower values (worse)")
                    else:
                        st.info(f"ℹ️ For {manifold_metric}: Red = lower values (better), Blue = higher values (worse)")
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select different parameters for X and Y axes.")

if __name__ == "__main__":
    main()