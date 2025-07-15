import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="Hyperparameter Analysis",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 Hyperparameter Analysis")
st.markdown("---")

class HyperparameterDataLoader:
    """Handles loading and processing hyperparameter data."""
    
    @staticmethod
    @st.cache_data
    def load_experiment_data(multi_inductive_experiment_dir):
        """Load and process experiment data."""
        final_results_path = os.path.join(multi_inductive_experiment_dir, "final_results.json")
        
        if not os.path.exists(final_results_path):
            return None, "final_results.json not found in the specified directory"
        
        try:
            with open(final_results_path, 'r') as f:
                data = json.load(f)
            return data, None
        except Exception as e:
            return None, str(e)

class HyperparameterAnalyzer:
    """Handles hyperparameter analysis and visualization."""
    
    @staticmethod
    def get_models_with_hyperopt(data):
        """Get all models that have hyperparameter optimization results."""
        models_with_hyperopt = []
        for result in data['all_results']:
            if 'model_results' in result:
                for task_name, task_results in result['model_results'].items():
                    for model_name, model_data in task_results.items():
                        # Check if the model has optimal_hyperparams field
                        if isinstance(model_data, dict) and 'optimal_hyperparams' in model_data:
                            if (task_name, model_name) not in models_with_hyperopt:
                                models_with_hyperopt.append((task_name, model_name))
        return models_with_hyperopt
    
    @staticmethod
    def get_hyperparameters_for_model(data, selected_task, selected_model):
        """Get all hyperparameters for the selected model."""
        hyperparams = set()
        available_metrics = set()
        
        for result in data['all_results']:
            if 'model_results' in result:
                if selected_task in result['model_results'] and selected_model in result['model_results'][selected_task]:
                    model_data = result['model_results'][selected_task][selected_model]
                    if isinstance(model_data, dict) and 'optimal_hyperparams' in model_data:
                        hyperparams.update(model_data['optimal_hyperparams'].keys())
                        # Check for test_metrics in different possible locations
                        if 'test_metrics' in model_data:
                            available_metrics.update(model_data['test_metrics'].keys())
                        elif 'fold_test_metrics' in model_data:
                            # If we have fold metrics, get the first fold to extract metric names
                            fold_metrics = model_data['fold_test_metrics']
                            if fold_metrics:
                                first_fold = next(iter(fold_metrics.values()))
                                if isinstance(first_fold, dict):
                                    available_metrics.update(first_fold.keys())
                        # Also check if metrics are directly in model_data
                        for key in model_data.keys():
                            if key not in ['optimal_hyperparams', 'test_metrics', 'fold_test_metrics'] and isinstance(model_data[key], (int, float)):
                                available_metrics.add(key)
        
        return list(hyperparams), list(available_metrics)
    
    @staticmethod
    def collect_hyperparameter_data(data, selected_task, selected_model, selected_hyperparam, performance_metric):
        """Collect hyperparameter values and corresponding performance."""
        hyperparam_values = []
        performance_values = []
        sweep_params = []
        run_ids = []
        
        for result in data['all_results']:
            if 'model_results' in result:
                if selected_task in result['model_results'] and selected_model in result['model_results'][selected_task]:
                    model_data = result['model_results'][selected_task][selected_model]
                    if isinstance(model_data, dict) and 'optimal_hyperparams' in model_data:
                        if selected_hyperparam in model_data['optimal_hyperparams']:
                            hyperparam_values.append(model_data['optimal_hyperparams'][selected_hyperparam])
                            
                            # Use the selected performance metric
                            metric_found = False
                            
                            # Check test_metrics first
                            if 'test_metrics' in model_data and performance_metric in model_data['test_metrics']:
                                metric_value = model_data['test_metrics'][performance_metric]
                                if isinstance(metric_value, dict) and 'mean' in metric_value:
                                    performance_values.append(metric_value['mean'])
                                else:
                                    performance_values.append(metric_value)
                                metric_found = True
                            
                            # Check fold_test_metrics if not found in test_metrics
                            elif 'fold_test_metrics' in model_data:
                                fold_metrics = model_data['fold_test_metrics']
                                if fold_metrics:
                                    # Calculate mean across folds
                                    fold_values = []
                                    for fold_name, fold_data in fold_metrics.items():
                                        if isinstance(fold_data, dict) and performance_metric in fold_data:
                                            fold_values.append(fold_data[performance_metric])
                                    
                                    if fold_values:
                                        performance_values.append(np.mean(fold_values))
                                        metric_found = True
                            
                            # Check if metric is directly in model_data
                            elif performance_metric in model_data and isinstance(model_data[performance_metric], (int, float)):
                                performance_values.append(model_data[performance_metric])
                                metric_found = True
                            
                            if not metric_found:
                                # If the performance metric isn't available, skip this run
                                continue
                            
                            # Store sweep parameters for correlation analysis
                            sweep_params.append(result.get('sweep_parameters', {}))
                            run_ids.append(result.get('run_id', 'N/A'))
        
        return hyperparam_values, performance_values, sweep_params, run_ids
    
    @staticmethod
    def create_hyperparameter_distribution_plot(hyperparam_values, selected_hyperparam):
        """Create distribution plot for hyperparameter values."""
        is_numerical = all(isinstance(x, (int, float)) for x in hyperparam_values)
        
        if is_numerical:
            # Create histogram for numerical values
            fig = px.histogram(
                x=hyperparam_values,
                title=f"Distribution of {selected_hyperparam}",
                labels={'x': selected_hyperparam, 'y': 'Count'},
                nbins=min(20, len(set(hyperparam_values))),
                color_discrete_sequence=['#1f77b4']
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
            )
        else:
            # Create bar plot for categorical values
            value_counts = pd.Series(hyperparam_values).value_counts()
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Distribution of {selected_hyperparam}",
                labels={'x': selected_hyperparam, 'y': 'Count'},
                color_discrete_sequence=['#1f77b4']
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
            )
        
        return fig
    
    @staticmethod
    def create_performance_vs_hyperparameter_plot(hyperparam_values, performance_values, 
                                                 selected_hyperparam, performance_metric):
        """Create performance vs hyperparameter plot."""
        is_numerical = all(isinstance(x, (int, float)) for x in hyperparam_values)
        
        if is_numerical:
            # Scatter plot for numerical values
            fig = px.scatter(
                x=hyperparam_values,
                y=performance_values,
                title=f"{performance_metric.upper()} vs {selected_hyperparam}",
                labels={'x': selected_hyperparam, 'y': performance_metric.upper()},
                trendline="ols",
                color_discrete_sequence=['#1f77b4']
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
            )
        else:
            # Enhanced box plot with individual data points for categorical values
            df_plot = pd.DataFrame({
                selected_hyperparam: hyperparam_values,
                performance_metric.upper(): performance_values
            })
            
            # Create figure with subplots for better control
            fig = go.Figure()
            
            # Get unique categories and their positions
            unique_categories = sorted(df_plot[selected_hyperparam].unique())
            category_positions = {cat: i for i, cat in enumerate(unique_categories)}
            
            # Add box plot
            for i, category in enumerate(unique_categories):
                category_data = df_plot[df_plot[selected_hyperparam] == category][performance_metric.upper()]
                
                if len(category_data) > 0:
                    # Calculate box plot statistics
                    q1 = category_data.quantile(0.25)
                    q3 = category_data.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    # Filter outliers
                    outliers = category_data[(category_data < lower_bound) | (category_data > upper_bound)]
                    non_outliers = category_data[(category_data >= lower_bound) & (category_data <= upper_bound)]
                    
                    # Add box plot
                    fig.add_trace(go.Box(
                        y=category_data,
                        name=category,
                        boxpoints=False,  # Don't show points in box
                        jitter=0.3,
                        pointpos=-1.8,
                        marker_color='#1f77b4',
                        marker_size=4,
                        marker_line_width=1,
                        marker_line_color='white',
                        line_color='#1f77b4',
                        line_width=2,
                        fillcolor='rgba(31, 119, 180, 0.1)',
                        showlegend=False
                    ))
                    
                    # Add individual data points with jitter
                    if len(category_data) > 0:
                        # Create jittered x positions
                        jitter_amount = 0.2
                        jittered_x = np.random.uniform(
                            i - jitter_amount, 
                            i + jitter_amount, 
                            len(category_data)
                        )
                        
                        fig.add_trace(go.Scatter(
                            x=jittered_x,
                            y=category_data,
                            mode='markers',
                            name=f'{category} (data points)',
                            marker=dict(
                                color='#1f77b4',
                                size=6,
                                opacity=0.7,
                                line=dict(color='white', width=1)
                            ),
                            showlegend=False,
                            hovertemplate=f'<b>{category}</b><br>{performance_metric.upper()}: %{{y}}<br><extra></extra>'
                        ))
            
            # Update layout for scientific publication style
            fig.update_layout(
                title=dict(
                    text=f"{performance_metric.upper()} by {selected_hyperparam}",
                    x=0.5,
                    xanchor='center',
                    font=dict(size=16, color='black')
                ),
                xaxis=dict(
                    title=dict(text=selected_hyperparam, font=dict(size=14, color='black')),
                    ticktext=unique_categories,
                    tickvals=list(range(len(unique_categories))),
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray',
                    zeroline=False,
                    showline=True,
                    linecolor='black',
                    linewidth=1
                ),
                yaxis=dict(
                    title=dict(text=performance_metric.upper(), font=dict(size=14, color='black')),
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray',
                    zeroline=False,
                    showline=True,
                    linecolor='black',
                    linewidth=1
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=60, r=30, t=80, b=60),
                height=500,
                showlegend=False
            )
        
        return fig
    
    @staticmethod
    def create_correlation_plots(hyperparam_values, sweep_params, selected_hyperparam, sweep_param_cols):
        """Create correlation plots between hyperparameter and sweep parameters."""
        is_numerical = all(isinstance(x, (int, float)) for x in hyperparam_values)
        
        if not sweep_params or not sweep_param_cols:
            return []
        
        plots = []
        for param in sweep_param_cols:
            param_values = [p.get(param, None) for p in sweep_params]
            
            # Filter out None values
            valid_indices = [i for i, v in enumerate(param_values) if v is not None]
            if not valid_indices:
                continue
                
            valid_param_values = [param_values[i] for i in valid_indices]
            valid_hyperparam_values = [hyperparam_values[i] for i in valid_indices]
            
            if is_numerical:
                # Scatter plot for numerical values
                fig = px.scatter(
                    x=valid_param_values,
                    y=valid_hyperparam_values,
                    title=f"{selected_hyperparam} vs {param}",
                    labels={'x': param, 'y': selected_hyperparam},
                    trendline="ols",
                    color_discrete_sequence=['#ff7f0e']
                )
            else:
                # Box plot for categorical values
                df_plot = pd.DataFrame({
                    param: valid_param_values,
                    selected_hyperparam: valid_hyperparam_values
                })
                fig = px.box(
                    df_plot,
                    x=param,
                    y=selected_hyperparam,
                    title=f"{selected_hyperparam} by {param}",
                    color_discrete_sequence=['#ff7f0e']
                )
            
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
            )
            
            plots.append((param, fig))
        
        return plots
    
    @staticmethod
    def create_correlation_plots_seaborn(hyperparam_values, sweep_params, selected_hyperparam, sweep_param_cols):
        """Create correlation plots using seaborn + matplotlib."""
        is_numerical = all(isinstance(x, (int, float)) for x in hyperparam_values)
        
        if not sweep_params or not sweep_param_cols:
            return []
        
        plots = []
        for param in sweep_param_cols:
            param_values = [p.get(param, None) for p in sweep_params]
            
            # Filter out None values
            valid_indices = [i for i, v in enumerate(param_values) if v is not None]
            if not valid_indices:
                continue
                
            valid_param_values = [param_values[i] for i in valid_indices]
            valid_hyperparam_values = [hyperparam_values[i] for i in valid_indices]
            
            # Set seaborn style
            sns.set_style("whitegrid")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))
            
            if is_numerical:
                # Scatter plot for numerical values
                sns.scatterplot(
                    x=valid_param_values,
                    y=valid_hyperparam_values,
                    ax=ax,
                    color='#ff7f0e',
                    alpha=0.7,
                    s=60
                )
                
                # Add trend line
                z = np.polyfit(valid_param_values, valid_hyperparam_values, 1)
                p = np.poly1d(z)
                ax.plot(valid_param_values, p(valid_param_values), "--", color='#ff7f0e', alpha=0.8, linewidth=2)
                
            else:
                # Box plot with individual points for categorical values
                df_plot = pd.DataFrame({
                    param: valid_param_values,
                    selected_hyperparam: valid_hyperparam_values
                })
                
                # Create box plot
                sns.boxplot(
                    data=df_plot,
                    x=param,
                    y=selected_hyperparam,
                    ax=ax,
                    showfliers=False,
                    color='lightcoral',
                    width=0.7,
                    linewidth=1.5
                )
                
                # Overlay individual data points
                sns.stripplot(
                    data=df_plot,
                    x=param,
                    y=selected_hyperparam,
                    ax=ax,
                    color='#ff7f0e',
                    size=4,
                    alpha=0.7,
                    jitter=0.2,
                    linewidth=0.5,
                    edgecolor='white'
                )
            
            # Customize the plot
            ax.set_title(f"{selected_hyperparam} vs {param}", fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel(param, fontsize=12, fontweight='normal')
            ax.set_ylabel(selected_hyperparam, fontsize=12, fontweight='normal')
            
            # Customize grid and styling
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            
            # Rotate x-axis labels if they're long
            if len(set(valid_param_values)) > 3:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            plots.append((param, fig))
        
        return plots
    
    @staticmethod
    def calculate_statistics(hyperparam_values, performance_values):
        """Calculate basic statistics for the hyperparameter analysis."""
        if not hyperparam_values or not performance_values:
            return {}
        
        is_numerical = all(isinstance(x, (int, float)) for x in hyperparam_values)
        
        stats = {
            'n_samples': len(hyperparam_values),
            'hyperparam_unique_values': len(set(hyperparam_values)),
            'performance_mean': np.mean(performance_values),
            'performance_std': np.std(performance_values),
            'performance_min': np.min(performance_values),
            'performance_max': np.max(performance_values)
        }
        
        if is_numerical:
            stats.update({
                'hyperparam_mean': np.mean(hyperparam_values),
                'hyperparam_std': np.std(hyperparam_values),
                'hyperparam_min': np.min(hyperparam_values),
                'hyperparam_max': np.max(hyperparam_values),
                'correlation': np.corrcoef(hyperparam_values, performance_values)[0, 1] if len(hyperparam_values) > 1 else 0
            })
        
        return stats
    
    @staticmethod
    def get_common_hyperparameters_across_models(data):
        """Find hyperparameters that are common across all models."""
        all_hyperparams = {}
        
        for result in data['all_results']:
            if 'model_results' in result:
                for task_name, task_results in result['model_results'].items():
                    for model_name, model_data in task_results.items():
                        if isinstance(model_data, dict) and 'optimal_hyperparams' in model_data:
                            model_key = f"{task_name}_{model_name}"
                            if model_key not in all_hyperparams:
                                all_hyperparams[model_key] = set()
                            all_hyperparams[model_key].update(model_data['optimal_hyperparams'].keys())
        
        if not all_hyperparams:
            return set()
        
        # Find intersection of all hyperparameter sets
        common_hyperparams = set.intersection(*all_hyperparams.values())
        return common_hyperparams
    
    @staticmethod
    def get_available_sweep_parameters(data):
        """Get all available sweep parameters across all results."""
        sweep_params = set()
        all_params = set()
        family_properties = set()
        
        for result in data['all_results']:
            # Get sweep parameters
            if 'sweep_parameters' in result:
                sweep_params.update(result['sweep_parameters'].keys())
            
            # Get all parameters
            if 'all_parameters' in result:
                all_params.update(result['all_parameters'].keys())
            
            # Get family properties
            if 'family_properties' in result:
                fp = result['family_properties']
                for key in fp.keys():
                    if key.endswith(('_mean', '_std', '_min', '_max')):
                        family_properties.add(f"family_{key}")
        
        # Combine all parameter types
        all_available_params = []
        all_available_params.extend([f"sweep_{param}" for param in sorted(sweep_params)])
        all_available_params.extend([f"param_{param}" for param in sorted(all_params)])
        all_available_params.extend(sorted(family_properties))
        
        return all_available_params
    
    @staticmethod
    def collect_cross_model_hyperparameter_data(data, selected_hyperparam, x_axis_param):
        """Collect hyperparameter data across all models."""
        cross_model_data = []
        
        for result in data['all_results']:
            if 'model_results' in result:
                # Extract x_value based on parameter type
                x_value = None
                if x_axis_param.startswith('sweep_'):
                    param_name = x_axis_param[6:]  # Remove 'sweep_' prefix
                    x_value = result.get('sweep_parameters', {}).get(param_name, None)
                elif x_axis_param.startswith('param_'):
                    param_name = x_axis_param[6:]  # Remove 'param_' prefix
                    x_value = result.get('all_parameters', {}).get(param_name, None)
                elif x_axis_param.startswith('family_'):
                    param_name = x_axis_param[7:]  # Remove 'family_' prefix
                    x_value = result.get('family_properties', {}).get(param_name, None)
                else:
                    # Fallback to sweep parameters
                    x_value = result.get('sweep_parameters', {}).get(x_axis_param, None)
                
                for task_name, task_results in result['model_results'].items():
                    for model_name, model_data in task_results.items():
                        if isinstance(model_data, dict) and 'optimal_hyperparams' in model_data:
                            if selected_hyperparam in model_data['optimal_hyperparams']:
                                cross_model_data.append({
                                    'run_id': result.get('run_id', 'N/A'),
                                    'task': task_name,
                                    'model': model_name,
                                    'hyperparam_value': model_data['optimal_hyperparams'][selected_hyperparam],
                                    'x_axis_value': x_value,
                                    'model_key': f"{task_name}_{model_name}"
                                })
        
        return cross_model_data
    
    @staticmethod
    def calculate_cross_model_statistics(cross_model_data):
        """Calculate statistics for cross-model hyperparameter analysis."""
        if not cross_model_data:
            return {}
        
        hyperparam_values = [d['hyperparam_value'] for d in cross_model_data]
        x_values = [d['x_axis_value'] for d in cross_model_data if d['x_axis_value'] is not None]
        unique_models = len(set(d['model_key'] for d in cross_model_data))
        
        is_numerical = all(isinstance(x, (int, float)) for x in hyperparam_values)
        is_x_numerical = all(isinstance(x, (int, float)) for x in x_values if x is not None)
        
        stats = {
            'total_samples': len(cross_model_data),
            'unique_models': unique_models,
            'unique_values': len(set(hyperparam_values)),
            'hyperparam_mean': np.mean(hyperparam_values) if is_numerical else None,
            'hyperparam_std': np.std(hyperparam_values) if is_numerical else None
        }
        
        # Calculate correlation if both are numerical
        if is_numerical and is_x_numerical and len(x_values) > 1:
            valid_pairs = [(h, x) for h, x in zip(hyperparam_values, x_values) if x is not None]
            if len(valid_pairs) > 1:
                h_vals, x_vals = zip(*valid_pairs)
                stats['correlation'] = np.corrcoef(h_vals, x_vals)[0, 1]
        
        return stats
    
    @staticmethod
    def create_cross_model_distribution_plot(cross_model_data, selected_hyperparam):
        """Create distribution plot for cross-model hyperparameter values."""
        hyperparam_values = [d['hyperparam_value'] for d in cross_model_data]
        is_numerical = all(isinstance(x, (int, float)) for x in hyperparam_values)
        
        if is_numerical:
            fig = px.histogram(
                x=hyperparam_values,
                title=f"Cross-Model Distribution of {selected_hyperparam}",
                labels={'x': selected_hyperparam, 'y': 'Count'},
                nbins=min(20, len(set(hyperparam_values))),
                color_discrete_sequence=['#2ca02c']
            )
        else:
            value_counts = pd.Series(hyperparam_values).value_counts()
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Cross-Model Distribution of {selected_hyperparam}",
                labels={'x': selected_hyperparam, 'y': 'Count'},
                color_discrete_sequence=['#2ca02c']
            )
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
        )
        
        return fig
    
    @staticmethod
    def create_cross_model_distribution_plot_seaborn(cross_model_data, selected_hyperparam):
        """Create distribution plot for cross-model hyperparameter values using seaborn."""
        hyperparam_values = [d['hyperparam_value'] for d in cross_model_data]
        is_numerical = all(isinstance(x, (int, float)) for x in hyperparam_values)
        
        # Set seaborn style
        sns.set_style("whitegrid")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if is_numerical:
            # Histogram for numerical values
            sns.histplot(
                data=hyperparam_values,
                ax=ax,
                bins=min(20, len(set(hyperparam_values))),
                color='#2ca02c',
                alpha=0.7,
                edgecolor='white',
                linewidth=1
            )
        else:
            # Bar plot for categorical values
            value_counts = pd.Series(hyperparam_values).value_counts()
            sns.barplot(
                x=value_counts.index,
                y=value_counts.values,
                ax=ax,
                color='#2ca02c',
                alpha=0.7,
                edgecolor='white',
                linewidth=1
            )
        
        # Customize the plot
        ax.set_title(f"Cross-Model Distribution of {selected_hyperparam}", 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(selected_hyperparam, fontsize=14, fontweight='normal')
        ax.set_ylabel('Count', fontsize=14, fontweight='normal')
        
        # Customize grid and styling
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Rotate x-axis labels if they're long
        if len(set(hyperparam_values)) > 5:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_cross_model_correlation_plot(cross_model_data, selected_hyperparam, x_axis_param):
        """Create correlation plot for cross-model hyperparameter vs x-axis parameter."""
        # Filter out None values for x-axis
        valid_data = [d for d in cross_model_data if d['x_axis_value'] is not None]
        
        if not valid_data:
            return go.Figure().add_annotation(text="No valid x-axis data", xref="paper", yref="paper", x=0.5, y=0.5)
        
        hyperparam_values = [d['hyperparam_value'] for d in valid_data]
        x_values = [d['x_axis_value'] for d in valid_data]
        
        is_numerical = all(isinstance(x, (int, float)) for x in hyperparam_values)
        is_x_numerical = all(isinstance(x, (int, float)) for x in x_values)
        
        if is_numerical and is_x_numerical:
            # Scatter plot for numerical values
            fig = px.scatter(
                x=x_values,
                y=hyperparam_values,
                title=f"{selected_hyperparam} vs {x_axis_param} (All Models)",
                labels={'x': x_axis_param, 'y': selected_hyperparam},
                trendline="ols",
                color_discrete_sequence=['#d62728']
            )
        else:
            # Box plot for categorical values
            df_plot = pd.DataFrame({
                x_axis_param: x_values,
                selected_hyperparam: hyperparam_values
            })
            fig = px.box(
                df_plot,
                x=x_axis_param,
                y=selected_hyperparam,
                title=f"{selected_hyperparam} by {x_axis_param} (All Models)",
                color_discrete_sequence=['#d62728']
            )
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
        )
        
        return fig
    
    @staticmethod
    def create_cross_model_correlation_plot_seaborn(cross_model_data, selected_hyperparam, x_axis_param):
        """Create correlation plot for cross-model hyperparameter vs x-axis parameter using seaborn."""
        # Filter out None values for x-axis
        valid_data = [d for d in cross_model_data if d['x_axis_value'] is not None]
        
        if not valid_data:
            # Create empty figure with message
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No valid x-axis data", ha='center', va='center', transform=ax.transAxes)
            return fig
        
        hyperparam_values = [d['hyperparam_value'] for d in valid_data]
        x_values = [d['x_axis_value'] for d in valid_data]
        
        is_numerical = all(isinstance(x, (int, float)) for x in hyperparam_values)
        is_x_numerical = all(isinstance(x, (int, float)) for x in x_values)
        
        # Set seaborn style
        sns.set_style("whitegrid")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if is_numerical and is_x_numerical:
            # Scatter plot for numerical values
            sns.scatterplot(
                x=x_values,
                y=hyperparam_values,
                ax=ax,
                color='#d62728',
                alpha=0.7,
                s=60
            )
            
            # Add trend line
            z = np.polyfit(x_values, hyperparam_values, 1)
            p = np.poly1d(z)
            ax.plot(x_values, p(x_values), "--", color='#d62728', alpha=0.8, linewidth=2)
            
        else:
            # Box plot with individual points for categorical values
            df_plot = pd.DataFrame({
                x_axis_param: x_values,
                selected_hyperparam: hyperparam_values
            })
            
            # Create box plot
            sns.boxplot(
                data=df_plot,
                x=x_axis_param,
                y=selected_hyperparam,
                ax=ax,
                showfliers=False,
                color='lightcoral',
                width=0.7,
                linewidth=1.5
            )
            
            # Overlay individual data points
            sns.stripplot(
                data=df_plot,
                x=x_axis_param,
                y=selected_hyperparam,
                ax=ax,
                color='#d62728',
                size=4,
                alpha=0.7,
                jitter=0.2,
                linewidth=0.5,
                edgecolor='white'
            )
        
        # Customize the plot
        ax.set_title(f"{selected_hyperparam} vs {x_axis_param} (All Models)", 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(x_axis_param, fontsize=14, fontweight='normal')
        ax.set_ylabel(selected_hyperparam, fontsize=14, fontweight='normal')
        
        # Customize grid and styling
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Rotate x-axis labels if they're long
        if len(set(x_values)) > 3:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_performance_vs_hyperparameter_plot_seaborn(hyperparam_values, performance_values, 
                                                         selected_hyperparam, performance_metric):
        """Create performance vs hyperparameter plot using seaborn + matplotlib for categorical data."""
        # Set seaborn style for clean, scientific appearance
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create DataFrame for plotting
        df_plot = pd.DataFrame({
            selected_hyperparam: hyperparam_values,
            performance_metric.upper(): performance_values
        })
        
        # Create box plot with seaborn
        sns.boxplot(
            data=df_plot,
            x=selected_hyperparam,
            y=performance_metric.upper(),
            ax=ax,
            showfliers=False,  # Don't show outliers as they'll be shown as individual points
            color='lightblue',
            width=0.7,
            linewidth=1.5
        )
        
        # Overlay individual data points with stripplot
        sns.stripplot(
            data=df_plot,
            x=selected_hyperparam,
            y=performance_metric.upper(),
            ax=ax,
            color='#1f77b4',
            size=4,
            alpha=0.7,
            jitter=0.2,
            linewidth=0.5,
            edgecolor='white'
        )
        
        # Customize the plot
        ax.set_title(f"{performance_metric.upper()} by {selected_hyperparam}", 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(selected_hyperparam, fontsize=14, fontweight='normal')
        ax.set_ylabel(performance_metric.upper(), fontsize=14, fontweight='normal')
        
        # Customize grid and styling
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Rotate x-axis labels if they're long
        if len(df_plot[selected_hyperparam].unique()) > 3:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def create_grouped_bar_plot_discrete(hyperparam_values, x_values, selected_hyperparam, x_axis_param):
        """Create grouped bar plot for discrete values showing counts."""
        # Create DataFrame for analysis
        df_plot = pd.DataFrame({
            'x_axis': x_values,
            'hyperparam': hyperparam_values
        })
        
        # Count combinations
        count_data = df_plot.groupby(['x_axis', 'hyperparam']).size().reset_index(name='count')
        
        # Create figure
        fig = go.Figure()
        
        # Get unique values for colors
        unique_hyperparams = sorted(count_data['hyperparam'].unique())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, hyperparam_val in enumerate(unique_hyperparams):
            hyperparam_data = count_data[count_data['hyperparam'] == hyperparam_val]
            
            fig.add_trace(go.Bar(
                x=hyperparam_data['x_axis'],
                y=hyperparam_data['count'],
                name=f'{selected_hyperparam}={hyperparam_val}',
                marker_color=colors[i % len(colors)],
                hovertemplate=f'<b>{selected_hyperparam}={hyperparam_val}</b><br>{x_axis_param}: %{{x}}<br>Count: %{{y}}<br><extra></extra>'
            ))
        
        fig.update_layout(
            title=f"Distribution of {selected_hyperparam} by {x_axis_param}",
            xaxis_title=x_axis_param,
            yaxis_title="Count",
            barmode='group',
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
        )
        
        return fig
    
    @staticmethod
    def create_grouped_bar_plot_discrete_seaborn(hyperparam_values, x_values, selected_hyperparam, x_axis_param):
        """Create grouped bar plot for discrete values using seaborn."""
        # Create DataFrame for analysis
        df_plot = pd.DataFrame({
            'x_axis': x_values,
            'hyperparam': hyperparam_values
        })
        
        # Count combinations
        count_data = df_plot.groupby(['x_axis', 'hyperparam']).size().reset_index(name='count')
        
        # Set seaborn style
        sns.set_style("whitegrid")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create grouped bar plot
        sns.barplot(
            data=count_data,
            x='x_axis',
            y='count',
            hue='hyperparam',
            ax=ax,
            palette='husl'
        )
        
        # Customize the plot
        ax.set_title(f"Distribution of {selected_hyperparam} by {x_axis_param}", 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(x_axis_param, fontsize=14, fontweight='normal')
        ax.set_ylabel("Count", fontsize=14, fontweight='normal')
        
        # Customize grid and styling
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Rotate x-axis labels if they're long
        if len(count_data['x_axis'].unique()) > 5:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_model_breakdown_plot(cross_model_data, selected_hyperparam):
        """Create plot showing hyperparameter distribution by model."""
        # Group by model and calculate statistics
        model_stats = {}
        for data_point in cross_model_data:
            model_key = data_point['model_key']
            if model_key not in model_stats:
                model_stats[model_key] = []
            model_stats[model_key].append(data_point['hyperparam_value'])
        
        # Calculate mean and std for each model
        model_names = []
        model_means = []
        model_stds = []
        
        for model_key, values in model_stats.items():
            model_names.append(model_key)
            model_means.append(np.mean(values))
            model_stds.append(np.std(values))
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=model_names,
            y=model_means,
            error_y=dict(type='data', array=model_stds, visible=True),
            name='Mean Hyperparameter Value',
            marker_color='#9467bd'
        ))
        
        fig.update_layout(
            title=f"{selected_hyperparam} Distribution by Model",
            xaxis_title="Model",
            yaxis_title=selected_hyperparam,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
        )
        
        return fig

class HyperparameterDashboard:
    """Main hyperparameter analysis dashboard."""
    
    def __init__(self):
        self.analyzer = HyperparameterAnalyzer()
    
    def run(self):
        """Run the main hyperparameter analysis dashboard."""
        # Sidebar for directory selection
        st.sidebar.header("📁 Directory Selection")
        
        main_dir = st.sidebar.text_input(
            "Main Directory Path",
            value=os.path.dirname(os.getcwd()),
            help="Path to the main directory containing your experiments"
        )
        
        multi_inductive_experiment_dir = st.sidebar.selectbox(
            "Experiment Directory",
            options=[
                "multi_results/community_detect",
                "multi_results/final_k1_hop"
            ],
            index=1,
            help="Select the experiment directory to analyze"
        )
        
        # Load data button
        if st.sidebar.button("🔄 Load Experiment Data"):
            with st.spinner("Loading experiment data..."):
                data, error = HyperparameterDataLoader.load_experiment_data(multi_inductive_experiment_dir)
                
                if error:
                    st.error(f"Error loading data: {error}")
                else:
                    st.session_state.data = data
                    st.session_state.experiment_dir = multi_inductive_experiment_dir
                    st.success(f"Successfully loaded experiment data!")
        
        # Main content
        if 'data' in st.session_state and st.session_state.data is not None:
            self._show_main_content()
        else:
            self._show_initial_content()
    
    def _show_main_content(self):
        """Show main dashboard content when data is loaded."""
        data = st.session_state.data
        
        # Display basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Runs", len(data['all_results']))
        with col2:
            total_models = sum(len(result['model_results']) for result in data['all_results'])
            st.metric("Total Model Results", total_models)
        with col3:
            models_with_hyperopt = self.analyzer.get_models_with_hyperopt(data)
            st.metric("Models with Hyperopt", len(models_with_hyperopt))
        
        st.markdown("---")
        
        # Hyperparameter Analysis Section
        st.header("🔧 Hyperparameter Analysis")
        
        # Get models with hyperparameter optimization
        models_with_hyperopt = self.analyzer.get_models_with_hyperopt(data)
        
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
            
            # Get hyperparameters and metrics for the selected model
            hyperparams, available_metrics = self.analyzer.get_hyperparameters_for_model(
                data, selected_task, selected_model
            )
            
            # Debug information
            st.info(f"Debug: Found {len(hyperparams)} hyperparameters and {len(available_metrics)} metrics")
            if available_metrics:
                st.info(f"Available metrics: {sorted(available_metrics)}")
            
            if not hyperparams:
                st.warning(f"No hyperparameter optimization results found for {selected_model} on {selected_task} task.")
            else:
                # Create two columns for selection
                col1, col2 = st.columns(2)
                
                with col1:
                    # Hyperparameter selection
                    selected_hyperparam = st.selectbox("Hyperparameter:", sorted(hyperparams))
                
                with col2:
                    # Performance metric selection
                    if available_metrics:
                        performance_metric = st.selectbox("Performance Metric:", sorted(available_metrics))
                    else:
                        st.warning("No performance metrics available for this model/task combination.")
                        # Show debug info about the model data structure
                        st.subheader("Debug: Model Data Structure")
                        for result in data['all_results']:
                            if 'model_results' in result:
                                if selected_task in result['model_results'] and selected_model in result['model_results'][selected_task]:
                                    model_data = result['model_results'][selected_task][selected_model]
                                    st.json(model_data)
                                    break
                        return
                
                # Collect data
                hyperparam_values, performance_values, sweep_params, run_ids = self.analyzer.collect_hyperparameter_data(
                    data, selected_task, selected_model, selected_hyperparam, performance_metric
                )
                
                if not hyperparam_values:
                    st.warning(f"No values found for {selected_hyperparam}.")
                else:
                    # Display statistics
                    stats = self.analyzer.calculate_statistics(hyperparam_values, performance_values)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Samples", stats.get('n_samples', 0))
                    with col2:
                        st.metric("Unique Values", stats.get('hyperparam_unique_values', 0))
                    with col3:
                        st.metric("Performance Mean", f"{stats.get('performance_mean', 0):.4f}")
                    with col4:
                        if 'correlation' in stats:
                            st.metric("Correlation", f"{stats.get('correlation', 0):.4f}")
                    
                    st.markdown("---")
                    
                    # Create two columns for plots
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Hyperparameter Distribution")
                        fig = self.analyzer.create_hyperparameter_distribution_plot(
                            hyperparam_values, selected_hyperparam
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader(f"Performance vs Hyperparameter")
                        
                        # Add toggle for plot type
                        use_seaborn = st.checkbox("Use Seaborn/Matplotlib (for categorical data)", value=False)
                        
                        if use_seaborn and not all(isinstance(x, (int, float)) for x in hyperparam_values):
                            # Use seaborn for categorical data
                            fig = self.analyzer.create_performance_vs_hyperparameter_plot_seaborn(
                                hyperparam_values, performance_values, selected_hyperparam, performance_metric
                            )
                            st.pyplot(fig, use_container_width=True)
                        else:
                            # Use plotly
                            fig = self.analyzer.create_performance_vs_hyperparameter_plot(
                                hyperparam_values, performance_values, selected_hyperparam, performance_metric
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Parameter correlation analysis
                    st.subheader("Parameter Correlation Analysis")
                    
                    # Get all available parameters for analysis
                    all_available_params = self.analyzer.get_available_sweep_parameters(data)
                    
                    if all_available_params:
                        sweep_param_cols = st.multiselect(
                            "Select parameters to analyze:",
                            all_available_params,
                            default=all_available_params[:2] if len(all_available_params) >= 2 else all_available_params
                        )
                        
                        if sweep_param_cols:
                            # Add plot type selection
                            plot_type = st.radio("Plot Type:", ["Correlation", "Grouped Bar (for discrete values)"], horizontal=True)
                            
                            # Add toggle for seaborn option
                            use_seaborn_correlation = st.checkbox("Use Seaborn/Matplotlib for plots", value=False)
                            
                            if plot_type == "Correlation":
                                if use_seaborn_correlation:
                                    # Create seaborn correlation plots
                                    plots = self.analyzer.create_correlation_plots_seaborn(
                                        hyperparam_values, sweep_params, selected_hyperparam, sweep_param_cols
                                    )
                                    
                                    if plots:
                                        cols = st.columns(len(plots))
                                        
                                        for i, (param, fig) in enumerate(plots):
                                            with cols[i]:
                                                st.pyplot(fig, use_container_width=True)
                                    else:
                                        st.warning("No valid correlation data found for the selected parameters.")
                                else:
                                    # Create plotly correlation plots
                                    plots = self.analyzer.create_correlation_plots(
                                        hyperparam_values, sweep_params, selected_hyperparam, sweep_param_cols
                                    )
                                    
                                    if plots:
                                        cols = st.columns(len(plots))
                                        
                                        for i, (param, fig) in enumerate(plots):
                                            with cols[i]:
                                                st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning("No valid correlation data found for the selected parameters.")
                            
                            else:  # Grouped Bar plot
                                # Check if we have discrete values (less than 5 unique values for both)
                                for param in sweep_param_cols:
                                    # Extract parameter values
                                    param_values = []
                                    for result in data['all_results']:
                                        if param.startswith('sweep_'):
                                            param_name = param[6:]
                                            param_values.append(result.get('sweep_parameters', {}).get(param_name, None))
                                        elif param.startswith('param_'):
                                            param_name = param[6:]
                                            param_values.append(result.get('all_parameters', {}).get(param_name, None))
                                        elif param.startswith('family_'):
                                            param_name = param[7:]
                                            param_values.append(result.get('family_properties', {}).get(param_name, None))
                                    
                                    # Filter out None values
                                    valid_param_values = [v for v in param_values if v is not None]
                                    valid_hyperparam_values = [h for h, p in zip(hyperparam_values, param_values) if p is not None]
                                    
                                    if len(set(valid_param_values)) <= 5 and len(set(valid_hyperparam_values)) <= 5:
                                        st.subheader(f"Grouped Bar Plot: {selected_hyperparam} vs {param}")
                                        
                                        if use_seaborn_correlation:
                                            fig = self.analyzer.create_grouped_bar_plot_discrete_seaborn(
                                                valid_hyperparam_values, valid_param_values, selected_hyperparam, param
                                            )
                                            st.pyplot(fig, use_container_width=True)
                                        else:
                                            fig = self.analyzer.create_grouped_bar_plot_discrete(
                                                valid_hyperparam_values, valid_param_values, selected_hyperparam, param
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning(f"Grouped bar plot not suitable for {param}: too many unique values (x-axis: {len(set(valid_param_values))}, hyperparam: {len(set(valid_hyperparam_values))})")
                    
                    # Data table
                    st.subheader("📋 Hyperparameter Data")
                    
                    # Create dataframe for display
                    df_display = pd.DataFrame({
                        'Run ID': run_ids,
                        selected_hyperparam: hyperparam_values,
                        performance_metric.upper(): performance_values
                    })
                    
                    # Add sweep parameters if available
                    if sweep_params and sweep_params[0]:
                        for param in sweep_params[0].keys():
                            param_values = [p.get(param, None) for p in sweep_params]
                            df_display[f'Sweep_{param}'] = param_values
                    
                    st.dataframe(df_display, use_container_width=True)
                    
                    # Export functionality
                    st.subheader("💾 Export Data")
                    if st.button("📥 Download as CSV"):
                        csv = df_display.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"hyperparameter_analysis_{selected_model}_{selected_hyperparam}.csv",
                            mime="text/csv"
                        )
                
                # Cross-model hyperparameter analysis
                st.markdown("---")
                st.header("🔍 Cross-Model Hyperparameter Analysis")
                
                # Find common hyperparameters across all models
                common_hyperparams = self.analyzer.get_common_hyperparameters_across_models(data)
                
                if common_hyperparams:
                    st.success(f"Found {len(common_hyperparams)} hyperparameters common across all models: {', '.join(sorted(common_hyperparams))}")
                    
                    # Selection for cross-model analysis
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_common_hyperparam = st.selectbox("Common Hyperparameter:", sorted(common_hyperparams))
                    with col2:
                        # Get available sweep parameters for x-axis
                        sweep_params_for_x = self.analyzer.get_available_sweep_parameters(data)
                        x_axis_param = st.selectbox("X-axis Parameter:", ['None'] + sweep_params_for_x)
                    
                    if selected_common_hyperparam and x_axis_param != 'None':
                        # Collect cross-model data
                        cross_model_data = self.analyzer.collect_cross_model_hyperparameter_data(
                            data, selected_common_hyperparam, x_axis_param
                        )
                        
                        if cross_model_data:
                            # Display statistics
                            stats = self.analyzer.calculate_cross_model_statistics(cross_model_data)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Samples", stats.get('total_samples', 0))
                            with col2:
                                st.metric("Unique Models", stats.get('unique_models', 0))
                            with col3:
                                st.metric("Unique Values", stats.get('unique_values', 0))
                            with col4:
                                if 'correlation' in stats:
                                    st.metric("Correlation", f"{stats.get('correlation', 0):.4f}")
                            
                            st.markdown("---")
                            
                            # Add plot type selection for cross-model analysis
                            plot_type_cross_model = st.radio("Cross-Model Plot Type:", ["Correlation", "Grouped Bar (for discrete values)"], horizontal=True)
                            
                            # Add toggle for seaborn option in cross-model analysis
                            use_seaborn_cross_model = st.checkbox("Use Seaborn/Matplotlib for cross-model plots", value=False)
                            
                            if plot_type_cross_model == "Correlation":
                                # Create plots
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.subheader("Cross-Model Distribution")
                                    if use_seaborn_cross_model:
                                        fig = self.analyzer.create_cross_model_distribution_plot_seaborn(
                                            cross_model_data, selected_common_hyperparam
                                        )
                                        st.pyplot(fig, use_container_width=True)
                                    else:
                                        fig = self.analyzer.create_cross_model_distribution_plot(
                                            cross_model_data, selected_common_hyperparam
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    st.subheader(f"Hyperparameter vs {x_axis_param}")
                                    if use_seaborn_cross_model:
                                        fig = self.analyzer.create_cross_model_correlation_plot_seaborn(
                                            cross_model_data, selected_common_hyperparam, x_axis_param
                                        )
                                        st.pyplot(fig, use_container_width=True)
                                    else:
                                        fig = self.analyzer.create_cross_model_correlation_plot(
                                            cross_model_data, selected_common_hyperparam, x_axis_param
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                            
                            else:  # Grouped Bar plot for cross-model
                                # Extract x-axis and hyperparameter values
                                x_values = [d['x_axis_value'] for d in cross_model_data if d['x_axis_value'] is not None]
                                hyperparam_values = [d['hyperparam_value'] for d in cross_model_data if d['x_axis_value'] is not None]
                                
                                if len(set(x_values)) <= 5 and len(set(hyperparam_values)) <= 5:
                                    st.subheader(f"Cross-Model Grouped Bar: {selected_common_hyperparam} vs {x_axis_param}")
                                    
                                    if use_seaborn_cross_model:
                                        fig = self.analyzer.create_grouped_bar_plot_discrete_seaborn(
                                            hyperparam_values, x_values, selected_common_hyperparam, x_axis_param
                                        )
                                        st.pyplot(fig, use_container_width=True)
                                    else:
                                        fig = self.analyzer.create_grouped_bar_plot_discrete(
                                            hyperparam_values, x_values, selected_common_hyperparam, x_axis_param
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning(f"Grouped bar plot not suitable for cross-model analysis: too many unique values (x-axis: {len(set(x_values))}, hyperparam: {len(set(hyperparam_values))})")
                                    st.info("Switch to 'Correlation' plot type for continuous data analysis.")
                            
                            # Model breakdown
                            st.subheader("Model Breakdown")
                            fig = self.analyzer.create_model_breakdown_plot(
                                cross_model_data, selected_common_hyperparam
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Data table
                            st.subheader("📋 Cross-Model Data")
                            df_cross_model = pd.DataFrame(cross_model_data)
                            st.dataframe(df_cross_model, use_container_width=True)
                            
                            # Export cross-model data
                            st.subheader("💾 Export Cross-Model Data")
                            if st.button("📥 Download Cross-Model CSV"):
                                csv = df_cross_model.to_csv(index=False)
                                st.download_button(
                                    label="Download Cross-Model CSV",
                                    data=csv,
                                    file_name=f"cross_model_hyperparameter_{selected_common_hyperparam}.csv",
                                    mime="text/csv"
                                )
                        else:
                            st.warning(f"No data found for {selected_common_hyperparam} across models.")
                else:
                    st.warning("No common hyperparameters found across all models.")
    
    def _show_initial_content(self):
        """Show initial content when no data is loaded."""
        st.info("👆 Please select your experiment directory and click 'Load Experiment Data' to begin hyperparameter analysis.")
        
        st.header("📖 Expected Directory Structure")
        st.code("""
main_dir/
├── multi_results/
│   └── final_sweep_community/
│       ├── final_results.json
│       └── run_folders/
│           ├── run_1/
│           │   └── some_experiment_dir/
│           │       └── results.json
│           └── run_2/
│               └── some_experiment_dir/
│                   └── results.json
        """, language="text")
        
        st.header("🔧 Hyperparameter Analysis Features")
        st.markdown("""
        This dashboard provides comprehensive hyperparameter analysis including:
        
        - **Distribution Analysis**: Visualize the distribution of optimal hyperparameter values
        - **Performance Correlation**: Analyze how hyperparameters relate to model performance
        - **Parameter Interactions**: Explore correlations between hyperparameters and sweep parameters
        - **Statistical Insights**: Get detailed statistics about hyperparameter optimization results
        
        The analysis supports both numerical and categorical hyperparameters with appropriate visualizations.
        """)

# Main execution
if __name__ == "__main__":
    dashboard = HyperparameterDashboard()
    dashboard.run() 