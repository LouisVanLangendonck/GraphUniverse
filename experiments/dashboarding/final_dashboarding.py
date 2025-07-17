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

# Page config
st.set_page_config(
    page_title="GNN Experiment Results Analyzer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 GNN Experiment Results Analyzer")
st.markdown("---")

class DataLoader:
    """Handles loading and processing experiment data."""
    
    @staticmethod
    @st.cache_data
    def get_result_family_properties(multi_inductive_experiment_dir, run_sample):
        """Extract family properties from results.json for a given run."""
        run = run_sample['data_files']['inductive_data'].split("/")[0]
        run_path = os.path.join(multi_inductive_experiment_dir, run)
        
        for dir_name in os.listdir(run_path):
            dir_path = os.path.join(run_path, dir_name)
            if os.path.isdir(dir_path) and dir_name != "" and dir_name != "data_analysis_report.txt":
                results_file = os.path.join(dir_path, "results.json")
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    return results['family_properties']
        return None

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
            
            results_list_with_family_properties = []
            for run_sample in data['all_results']:
                family_properties = DataLoader.get_result_family_properties(
                    multi_inductive_experiment_dir, run_sample
                )
                if family_properties:
                    run_sample['family_properties'] = family_properties
                results_list_with_family_properties.append(run_sample)
            
            return results_list_with_family_properties, None
        except Exception as e:
            return None, str(e)

class DataProcessor:
    """Handles data processing and metric extraction."""
    
    @staticmethod
    def is_regression_task(experiment_dir):
        """Check if the experiment is a regression task (not community detection)."""
        return experiment_dir != "multi_results/community_detect"
    
    @staticmethod
    def get_task_name_from_dataframe(df):
        """Extract task name from dataframe columns."""
        task_names = set()
        for col in df.columns:
            if col.startswith('model.') and col.endswith('.mean'):
                parts = col.split('.')
                if len(parts) >= 4:
                    task_names.add(parts[1])
        return list(task_names)[0] if task_names else "community"
    
    @staticmethod
    def extract_model_metrics(model_results, experiment_dir=None, family_properties=None):
        """Extract model performance metrics with fold averaging."""
        metrics = {}
        if not model_results:
            return metrics
        
        is_regression = DataProcessor.is_regression_task(experiment_dir) if experiment_dir else False
        mean_density = None
        if is_regression and family_properties:
            mean_density = family_properties.get('densities_mean')

        for task_name, task_results in model_results.items():
            if isinstance(task_results, dict):
                for model_name, model_data in task_results.items():
                    if isinstance(model_data, dict):
                        # Check if we have fold_test_metrics structure
                        if 'fold_test_metrics' in model_data:
                            fold_metrics = model_data['fold_test_metrics']
                            fold_values = {}
                            
                            # Collect all fold values for each metric
                            for fold_name, fold_data in fold_metrics.items():
                                if isinstance(fold_data, dict):
                                    for metric_name, metric_value in fold_data.items():
                                        if isinstance(metric_value, (int, float)):
                                            if metric_name not in fold_values:
                                                fold_values[metric_name] = []
                                            fold_values[metric_name].append(metric_value)
                            
                            # Calculate mean and 95% confidence interval for each metric
                            for metric_name, values in fold_values.items():
                                if values:
                                    mean_val = np.mean(values)
                                    std_val = np.std(values)
                                    n_folds = len(values)
                                    ci_95 = 1.96 * std_val / np.sqrt(n_folds)
                                    
                                    # For regression tasks, negate MSE values to make maximum = best
                                    if is_regression and metric_name.lower() in ['mse', 'rmse', 'mae']:
                                        mean_val = -mean_val
                                    
                                    key_mean = f"model.{task_name}.{model_name}.{metric_name}.mean"
                                    key_ci = f"model.{task_name}.{model_name}.{metric_name}.ci95"
                                    metrics[key_mean] = mean_val
                                    metrics[key_ci] = ci_95
                                    
                                    # Add normalized version for regression tasks
                                    if is_regression and mean_density is not None and mean_density > 0:
                                        normalized_mean = mean_val / mean_density
                                        normalized_ci = ci_95 / mean_density
                                        
                                        key_normalized_mean = f"model.{task_name}.{model_name}.{metric_name}_normalized.mean"
                                        key_normalized_ci = f"model.{task_name}.{model_name}.{metric_name}_normalized.ci95"
                                        metrics[key_normalized_mean] = normalized_mean
                                        metrics[key_normalized_ci] = normalized_ci
                        elif 'repetition_test_metrics' in model_data:
                            # New repetition-based structure
                            repetition_metrics = model_data['repetition_test_metrics']
                            repetition_values = {}
                            
                            # Collect all repetition values for each metric
                            for repetition_name, repetition_data in repetition_metrics.items():
                                if isinstance(repetition_data, dict):
                                    for metric_name, metric_value in repetition_data.items():
                                        if isinstance(metric_value, (int, float)):
                                            if metric_name not in repetition_values:
                                                repetition_values[metric_name] = []
                                            repetition_values[metric_name].append(metric_value)
                            
                            # Calculate mean and 95% confidence interval for each metric
                            for metric_name, values in repetition_values.items():
                                if values:
                                    mean_val = np.mean(values)
                                    std_val = np.std(values)
                                    n_repetitions = len(values)
                                    ci_95 = 1.96 * std_val / np.sqrt(n_repetitions)
                                    
                                    # For regression tasks, negate MSE values to make maximum = best
                                    if is_regression and metric_name.lower() in ['mse', 'rmse', 'mae']:
                                        mean_val = -mean_val
                                    
                                    key_mean = f"model.{task_name}.{model_name}.{metric_name}.mean"
                                    key_ci = f"model.{task_name}.{model_name}.{metric_name}.ci95"
                                    metrics[key_mean] = mean_val
                                    metrics[key_ci] = ci_95
                                    
                                    # Add normalized version for regression tasks
                                    if is_regression and mean_density is not None and mean_density > 0:
                                        normalized_mean = mean_val / mean_density
                                        normalized_ci = ci_95 / mean_density
                                        
                                        key_normalized_mean = f"model.{task_name}.{model_name}.{metric_name}_normalized.mean"
                                        key_normalized_ci = f"model.{task_name}.{model_name}.{metric_name}_normalized.ci95"
                                        metrics[key_normalized_mean] = normalized_mean
                                        metrics[key_normalized_ci] = normalized_ci
                        else:
                            # Fallback for non-fold structure
                            for metric_name, metric_value in model_data.items():
                                if isinstance(metric_value, (int, float)):
                                    if is_regression and metric_name.lower() in ['mse', 'rmse', 'mae']:
                                        metric_value = -metric_value
                                    
                                    key = f"model.{task_name}.{model_name}.{metric_name}"
                                    metrics[key] = metric_value
                                    
                                    if is_regression and mean_density is not None and mean_density > 0:
                                        normalized_value = metric_value / mean_density
                                        key_normalized = f"model.{task_name}.{model_name}.{metric_name}_normalized"
                                        metrics[key_normalized] = normalized_value
        return metrics

    @staticmethod
    def create_dataframe_from_results(results_list, experiment_dir=None):
        """Convert results list to pandas DataFrame."""
        df_data = []
        
        for run_sample in results_list:
            row = {}
            
            # Add basic run information
            row['run_id'] = run_sample.get('run_id', 'N/A')
            row['timestamp'] = run_sample.get('timestamp', 'N/A')
            row['method'] = run_sample.get('method', 'N/A')
            row['total_time'] = run_sample.get('total_time', 0)
            
            # Add sweep parameters
            if 'sweep_parameters' in run_sample:
                for key, value in run_sample['sweep_parameters'].items():
                    row[f"sweep_{key}"] = value
            
            # Add all parameters
            if 'all_parameters' in run_sample:
                for key, value in run_sample['all_parameters'].items():
                    row[f"param_{key}"] = value
            
            # Add family properties
            if 'family_properties' in run_sample:
                fp = run_sample['family_properties']
                for key, value in fp.items():
                    if key.endswith(('_mean', '_std', '_min', '_max')):
                        row[f"family_{key}"] = value
            
            # Add model results
            if 'model_results' in run_sample:
                family_properties = run_sample.get('family_properties', {})
                model_metrics = DataProcessor.extract_model_metrics(
                    run_sample['model_results'], experiment_dir, family_properties
                )
                for key, value in model_metrics.items():
                    row[key] = value
            
            # Add community signals
            if 'community_signals' in run_sample:
                for signal_type, signal_data in run_sample['community_signals'].items():
                    if isinstance(signal_data, dict):
                        for stat_name, stat_value in signal_data.items():
                            if stat_name in ['mean', 'std', 'min', 'max']:
                                row[f"community_{signal_type}_{stat_name}"] = stat_value
            
            # Add family consistency scores
            if 'family_consistency' in run_sample:
                fc = run_sample['family_consistency']
                for consistency_type, consistency_data in fc.items():
                    if isinstance(consistency_data, dict) and 'score' in consistency_data:
                        row[f"consistency_{consistency_type}_score"] = consistency_data['score']
            
            # Add unseen community combination score
            if 'unseen_community_combination_score' in run_sample:
                row['unseen_community_combination_score'] = run_sample['unseen_community_combination_score']
            
            df_data.append(row)
        
        return pd.DataFrame(df_data)

class RankingAnalyzer:
    """Handles ranking analysis with uncertainty."""
    
    @staticmethod
    def calculate_ranking_with_uncertainty(df, selected_models, selected_metric, x_axis, 
                                         second_param='None', n_samples=100, seed=42, task_name='community'):
        """Calculate ranking with uncertainty by sampling from normal distributions."""
        np.random.seed(seed)
        ranking_data = []
        
        for idx, row in df.iterrows():
            model_performances = {}
            model_uncertainties = {}
            
            for model in selected_models:
                mean_col = f"model.{task_name}.{model}.{selected_metric}.mean"
                std_col = f"model.{task_name}.{model}.{selected_metric}.ci95"
                
                if mean_col in row and not pd.isna(row[mean_col]):
                    model_performances[model] = row[mean_col]
                    if std_col in row and not pd.isna(row[std_col]):
                        model_uncertainties[model] = row[std_col] / 1.96
                    else:
                        model_uncertainties[model] = 0.01
            
            if len(model_performances) > 1:
                performance_values = list(model_performances.values())
                if np.std(performance_values) < 1e-10:
                    # All models have essentially the same performance
                    equal_rank = (len(model_performances) + 1) / 2
                    for model in model_performances.keys():
                        ranking_data.append({
                            'run_id': row.get('run_id', idx),
                            'model': model,
                            'mean_rank': equal_rank,
                            'std_rank': 0.0,
                            'x_axis': row.get(x_axis, 0),
                            'second_param': row.get(second_param, None) if second_param != 'None' else None
                        })
                else:
                    # Sample rankings multiple times
                    rankings = []
                    for _ in range(n_samples):
                        sampled_performances = {}
                        for model, mean in model_performances.items():
                            std = model_uncertainties.get(model, 0.01)
                            sampled_performances[model] = np.random.normal(mean, std)
                        
                        sorted_models = sorted(sampled_performances.items(), key=lambda x: x[1], reverse=True)
                        ranking = {model: rank + 1 for rank, (model, _) in enumerate(sorted_models)}
                        rankings.append(ranking)
                    
                    # Calculate mean and std of rankings
                    for model in model_performances.keys():
                        model_rankings = [r[model] for r in rankings]
                        mean_rank = np.mean(model_rankings)
                        std_rank = np.std(model_rankings)
                        
                        ranking_data.append({
                            'run_id': row.get('run_id', idx),
                            'model': model,
                            'mean_rank': mean_rank,
                            'std_rank': std_rank,
                            'x_axis': row.get(x_axis, 0),
                            'second_param': row.get(second_param, None) if second_param != 'None' else None
                        })
        
        return pd.DataFrame(ranking_data)

class StatisticalAnalyzer:
    """Handles statistical analysis and curve fitting."""
    
    @staticmethod
    def propagate_uncertainty_linear_regression(x, y_mean, y_std, x_pred=None):
        """Fit linear regression with proper uncertainty propagation using bootstrap."""
        n_bootstrap = 1000
        
        if x_pred is None:
            x_pred = np.linspace(x.min(), x.max(), 100)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y_mean) | np.isnan(y_std))
        x_clean = x[valid_mask]
        y_mean_clean = y_mean[valid_mask]
        y_std_clean = y_std[valid_mask]
        
        if len(x_clean) < 3 or np.std(x_clean) == 0 or np.std(y_mean_clean) == 0:
            return None
        
        bootstrap_slopes = []
        bootstrap_intercepts = []
        bootstrap_predictions = []
        correlations = []
        r_squared_values = []
        
        for _ in range(n_bootstrap):
            boot_indices = np.random.choice(len(x_clean), size=len(x_clean), replace=True)
            x_boot = x_clean[boot_indices]
            y_mean_boot = y_mean_clean[boot_indices]
            y_std_boot = y_std_clean[boot_indices]
            
            y_boot = np.random.normal(y_mean_boot, y_std_boot)
            
            try:
                reg = LinearRegression().fit(x_boot.reshape(-1, 1), y_boot)
                slope = reg.coef_[0]
                intercept = reg.intercept_
                
                y_pred_boot = reg.predict(x_boot.reshape(-1, 1))
                ss_res = np.sum((y_boot - y_pred_boot) ** 2)
                ss_tot = np.sum((y_boot - np.mean(y_boot)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                try:
                    corr, _ = stats.pearsonr(x_boot, y_boot)
                    if np.isnan(corr):
                        corr = 0.0
                except (ValueError, RuntimeWarning):
                    corr = 0.0
                
                if not (np.isnan(slope) or np.isnan(intercept) or np.isnan(r_squared)):
                    bootstrap_slopes.append(slope)
                    bootstrap_intercepts.append(intercept)
                    r_squared_values.append(r_squared)
                    correlations.append(corr)
                    
                    pred_boot = reg.predict(x_pred.reshape(-1, 1))
                    bootstrap_predictions.append(pred_boot)
            except:
                continue
        
        if len(bootstrap_slopes) < 50:
            return None
        
        bootstrap_slopes = np.array(bootstrap_slopes)
        bootstrap_intercepts = np.array(bootstrap_intercepts)
        bootstrap_predictions = np.array(bootstrap_predictions)
        correlations = np.array(correlations)
        r_squared_values = np.array(r_squared_values)
        
        # Calculate statistics
        slope_mean = np.mean(bootstrap_slopes)
        slope_std = np.std(bootstrap_slopes)
        
        # Bootstrap-based p-values
        if slope_mean > 0:
            slope_p_value = 2 * np.mean(bootstrap_slopes <= 0)
        else:
            slope_p_value = 2 * np.mean(bootstrap_slopes >= 0)
        slope_p_value = max(slope_p_value, 1 / len(bootstrap_slopes))
        
        mean_correlation = np.mean(correlations)
        if mean_correlation > 0:
            corr_p_value = 2 * np.mean(correlations <= 0)
        else:
            corr_p_value = 2 * np.mean(correlations >= 0)
        corr_p_value = max(corr_p_value, 1 / len(correlations))
        
        # Calculate prediction intervals
        mean_line_boot = np.mean(bootstrap_predictions, axis=0)
        pred_lower_boot = np.percentile(bootstrap_predictions, 2.5, axis=0)
        pred_upper_boot = np.percentile(bootstrap_predictions, 97.5, axis=0)
        
        return {
            'x_pred': x_pred,
            'mean_line': mean_line_boot,
            'pred_lower': pred_lower_boot,
            'pred_upper': pred_upper_boot,
            'slope': {
                'mean': slope_mean,
                'std': slope_std,
                'p_value': slope_p_value
            },
            'correlation': {
                'mean': mean_correlation,
                'p_value': corr_p_value
            },
            'r_squared': {
                'mean': np.mean(r_squared_values),
                'std': np.std(r_squared_values)
            }
        }

    @staticmethod
    def propagate_uncertainty_polynomial_regression(x, y_mean, y_std, degree=2, x_pred=None):
        """Fit polynomial regression with proper uncertainty propagation using bootstrap."""
        n_bootstrap = 1000
        
        if x_pred is None:
            x_pred = np.linspace(x.min(), x.max(), 100)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y_mean) | np.isnan(y_std))
        x_clean = x[valid_mask]
        y_mean_clean = y_mean[valid_mask]
        y_std_clean = y_std[valid_mask]
        
        if len(x_clean) < degree + 2:  # Need at least degree + 2 points for polynomial fitting
            return None
        
        # Check for sufficient variation in the data
        if np.std(x_clean) == 0 or np.std(y_mean_clean) == 0:
            return None
        
        bootstrap_predictions = []
        r_squared_values = []
        
        for _ in range(n_bootstrap):
            # Bootstrap resample indices
            boot_indices = np.random.choice(len(x_clean), size=len(x_clean), replace=True)
            x_boot = x_clean[boot_indices]
            y_mean_boot = y_mean_clean[boot_indices]
            y_std_boot = y_std_clean[boot_indices]
            
            # Sample from uncertainty for each bootstrapped point
            y_boot = np.random.normal(y_mean_boot, y_std_boot)
            
            # Fit polynomial regression
            try:
                # Create polynomial features
                poly_features = PolynomialFeatures(degree=degree, include_bias=False)
                x_poly = poly_features.fit_transform(x_boot.reshape(-1, 1))
                
                # Fit linear regression on polynomial features
                reg = LinearRegression().fit(x_poly, y_boot)
                
                # Calculate R² with proper error handling
                y_pred_boot = reg.predict(x_poly)
                ss_res = np.sum((y_boot - y_pred_boot) ** 2)
                ss_tot = np.sum((y_boot - np.mean(y_boot)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                if not np.isnan(r_squared) and not np.isinf(r_squared):
                    r_squared_values.append(r_squared)
                    
                    # Predictions for this bootstrap sample
                    x_pred_poly = poly_features.transform(x_pred.reshape(-1, 1))
                    pred_boot = reg.predict(x_pred_poly)
                    bootstrap_predictions.append(pred_boot)
            except:
                continue
        
        if len(bootstrap_predictions) < 50:  # Need sufficient bootstrap samples
            return None
        
        bootstrap_predictions = np.array(bootstrap_predictions)
        
        # Calculate mean and confidence intervals
        mean_line_boot = np.mean(bootstrap_predictions, axis=0)
        pred_lower_boot = np.percentile(bootstrap_predictions, 2.5, axis=0)
        pred_upper_boot = np.percentile(bootstrap_predictions, 97.5, axis=0)
        
        # Calculate effective uncertainty
        # Use the mean of bootstrap predictions at the original x points for residual calculation
        x_pred_original = x_clean
        mean_pred_original = np.mean(bootstrap_predictions, axis=0)
        
        # Interpolate to get predictions at original x points
        # For polynomial, we need to use the polynomial features at original points
        poly_features_original = PolynomialFeatures(degree=degree, include_bias=False)
        x_poly_original = poly_features_original.fit_transform(x_clean.reshape(-1, 1))
        
        # Fit main regression with mean values to get predictions at original points
        main_reg = LinearRegression().fit(x_poly_original, y_mean_clean)
        y_pred_original = main_reg.predict(x_poly_original)
        
        residuals = y_mean_clean - y_pred_original
        effective_std = np.sqrt(np.mean(y_std_clean**2 + np.var(residuals)))
        
        return {
            'x_pred': x_pred,
            'mean_line': mean_line_boot,
            'pred_lower': pred_lower_boot,
            'pred_upper': pred_upper_boot,
            'r_squared': {
                'mean': np.mean(r_squared_values),
                'std': np.std(r_squared_values)
            },
            'effective_uncertainty': effective_std,
            'n_bootstrap_samples': len(bootstrap_predictions),
            'degree': degree
        }

class PlotGenerator:
    """Handles plot generation with different styles."""
    
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def create_scatter_plot(self, plot_df, x_axis, selected_models, selected_metric, 
                           task_name, second_param=None):
        """Create scatter plot."""
        fig = go.Figure()
        
        for i, model in enumerate(selected_models):
            model_metric_col = f"model.{task_name}.{model}.{selected_metric}.mean"
            model_error_col = f"model.{task_name}.{model}.{selected_metric}.ci95"
            
            if model_metric_col in plot_df.columns:
                model_data = plot_df.dropna(subset=[model_metric_col])
                
                if len(model_data) > 0:
                    error_y_data = model_data.get(model_error_col)
                    color = self.colors[i % len(self.colors)]
                    
                    # Calculate dot sizes if second parameter is specified
                    sizes = 8
                    if second_param and second_param in model_data.columns:
                        param_values = model_data[second_param]
                        if param_values.nunique() > 1:
                            min_size, max_size = 5, 15
                            sizes = min_size + (max_size - min_size) * \
                                   (param_values - param_values.min()) / (param_values.max() - param_values.min())
                    
                    fig.add_trace(go.Scatter(
                        x=model_data[x_axis],
                        y=model_data[model_metric_col],
                        mode='markers',
                        name=model,
                        marker=dict(
                            color=color,
                            size=sizes,
                            line=dict(width=1, color='white')
                        ),
                        error_y=dict(
                            type='data',
                            array=error_y_data,
                            visible=True,
                            thickness=1,
                            width=3
                        ) if error_y_data is not None else None,
                        hovertemplate=f'<b>{model}</b><br>' +
                                     f'{x_axis}: %{{x}}<br>' +
                                     f'{selected_metric}: %{{y}}<br>' +
                                     '<extra></extra>'
                    ))
        
        return fig
    
    def create_line_fit_plot(self, plot_df, x_axis, selected_models, selected_metric, 
                            task_name, fit_type="Linear", poly_degree=2, show_only_lines=False):
        """Create line fit plot with uncertainty bands."""
        fig = go.Figure()
        
        for i, model in enumerate(selected_models):
            model_metric_col = f"model.{task_name}.{model}.{selected_metric}.mean"
            model_error_col = f"model.{task_name}.{model}.{selected_metric}.ci95"
            
            if model_metric_col in plot_df.columns:
                model_data = plot_df.dropna(subset=[model_metric_col])
                
                if len(model_data) > 0:
                    color = self.colors[i % len(self.colors)]
                    error_y_data = model_data.get(model_error_col)
                    
                    # Show data points if not showing only lines
                    if not show_only_lines:
                        fig.add_trace(go.Scatter(
                            x=model_data[x_axis],
                            y=model_data[model_metric_col],
                            mode='markers',
                            name=model,
                            marker=dict(color=color, size=6, line=dict(width=1, color='white')),
                            error_y=dict(
                                type='data',
                                array=error_y_data,
                                visible=True,
                                thickness=1,
                                width=3
                            ) if error_y_data is not None else None,
                            hovertemplate=f'<b>{model}</b><br>{x_axis}: %{{x}}<br>{selected_metric}: %{{y}}<br><extra></extra>'
                        ))
                    
                    # Add fitted line if enough points
                    if len(model_data) >= 3:
                        sorted_data = model_data.sort_values(x_axis)
                        x_fit = sorted_data[x_axis].values
                        y_fit = sorted_data[model_metric_col].values
                        y_std_fit = (sorted_data[model_error_col].values / 1.96 
                                   if model_error_col in sorted_data.columns 
                                   else np.ones_like(y_fit) * 0.01)
                        
                        fit_result = StatisticalAnalyzer.propagate_uncertainty_linear_regression(
                            x_fit, y_fit, y_std_fit
                        )
                        
                        if fit_result is not None:
                            # Add fitted line
                            fig.add_trace(go.Scatter(
                                x=fit_result['x_pred'],
                                y=fit_result['mean_line'],
                                mode='lines',
                                name=f"{model} (Fitted)",
                                line=dict(color=color, width=3, dash='dash'),
                                hovertemplate=f'<b>{model} (Fitted)</b><br>{x_axis}: %{{x}}<br>{selected_metric}: %{{y}}<br><extra></extra>'
                            ))
                            
                            # Add confidence bands
                            if color.startswith('#') and len(color) == 7:
                                r = int(color[1:3], 16)
                                g = int(color[3:5], 16)
                                b = int(color[5:7], 16)
                                rgba_color = f'rgba({r}, {g}, {b}, 0.2)'
                            else:
                                rgba_color = 'rgba(0, 0, 0, 0.2)'
                            
                            fig.add_trace(go.Scatter(
                                x=np.concatenate([fit_result['x_pred'], fit_result['x_pred'][::-1]]),
                                y=np.concatenate([fit_result['pred_upper'], fit_result['pred_lower'][::-1]]),
                                fill='toself',
                                fillcolor=rgba_color,
                                line=dict(color='rgba(0,0,0,0)'),
                                name=f"{model} (95% CI)",
                                showlegend=False,
                                hoverinfo='skip'
                            ))
        
        return fig
    
    def create_grouped_bar_plot(self, plot_df, x_axis, selected_models, selected_metric, 
                               fit_target, task_name, n_ranking_samples=100, ranking_seed=42):
        """Create grouped bar plot."""
        # Calculate grouped bar data
        grouped_df = self._calculate_grouped_bar_data(
            plot_df, x_axis, selected_models, selected_metric, 
            fit_target, n_ranking_samples, ranking_seed, task_name
        )
        
        if grouped_df.empty:
            return None
        
        fig = go.Figure()
        unique_x_vals = sorted(grouped_df['x_value'].unique())
        n_models = len(selected_models)
        bar_width = 0.8 / n_models
        x_positions = np.arange(len(unique_x_vals))
        
        for j, model in enumerate(selected_models):
            model_data = grouped_df[grouped_df['model'] == model]
            
            if not model_data.empty:
                color = self.colors[j % len(self.colors)]
                bar_positions = x_positions + (j - (n_models-1)/2) * bar_width
                
                # Map data to bar positions
                y_values = []
                std_values = []
                for x_val in unique_x_vals:
                    model_x_data = model_data[model_data['x_value'] == x_val]
                    if not model_x_data.empty:
                        y_values.append(model_x_data.iloc[0]['mean_value'])
                        std_values.append(model_x_data.iloc[0]['std_value'])
                    else:
                        y_values.append(None)
                        std_values.append(None)
                
                fig.add_trace(go.Bar(
                    x=bar_positions,
                    y=y_values,
                    name=model,
                    marker_color=color,
                    error_y=dict(
                        type='data',
                        array=std_values,
                        visible=True,
                        thickness=1,
                        width=3
                    ),
                    hovertemplate=f'<b>{model}</b><br>{x_axis}: %{{x}}<br>Mean: %{{y}}<br>Std: %{{error_y.array}}<br><extra></extra>'
                ))
        
        # Update x-axis labels
        fig.update_xaxes(
            ticktext=[str(x) for x in unique_x_vals],
            tickvals=x_positions
        )
        
        return fig
    
    def _calculate_grouped_bar_data(self, df, x_axis, selected_models, selected_metric, 
                                   fit_target, n_ranking_samples, ranking_seed, task_name):
        """Calculate grouped bar data by averaging results across runs."""
        grouped_data = []
        unique_x_values = sorted(df[x_axis].dropna().unique())
        
        for x_val in unique_x_values:
            filtered_df = df[df[x_axis] == x_val]
            
            if len(filtered_df) == 0:
                continue
                
            for model in selected_models:
                if fit_target == "Ranking":
                    ranking_df = RankingAnalyzer.calculate_ranking_with_uncertainty(
                        filtered_df, selected_models, selected_metric, x_axis, 'None',
                        n_ranking_samples, ranking_seed, task_name
                    )
                    
                    if not ranking_df.empty:
                        model_ranking = ranking_df[ranking_df['model'] == model]
                        if not model_ranking.empty:
                            mean_rank = model_ranking['mean_rank'].mean()
                            rank_vars = model_ranking['std_rank']**2
                            mean_rank_std = np.sqrt(np.sum(rank_vars)) / len(model_ranking)
                            
                            grouped_data.append({
                                'x_value': x_val,
                                'model': model,
                                'mean_value': mean_rank,
                                'std_value': mean_rank_std,
                                'metric_type': 'ranking'
                            })
                else:
                    model_metric_col = f"model.{task_name}.{model}.{selected_metric}.mean"
                    model_error_col = f"model.{task_name}.{model}.{selected_metric}.ci95"
                    
                    if model_metric_col in filtered_df.columns:
                        model_data = filtered_df.dropna(subset=[model_metric_col])
                        
                        if len(model_data) > 0:
                            mean_metric = model_data[model_metric_col].mean()
                            
                            if model_error_col in model_data.columns:
                                metric_stds = model_data[model_error_col] / 1.96
                                metric_vars = metric_stds**2
                                mean_metric_std = np.sqrt(np.sum(metric_vars)) / len(model_data)
                            else:
                                mean_metric_std = model_data[model_metric_col].std() / np.sqrt(len(model_data))
                            
                            grouped_data.append({
                                'x_value': x_val,
                                'model': model,
                                'mean_value': mean_metric,
                                'std_value': mean_metric_std,
                                'metric_type': 'metric'
                            })
        
        return pd.DataFrame(grouped_data)
    
    def create_ranking_plot(self, plot_df, x_axis, selected_models, selected_metric, 
                           task_name, n_ranking_samples=100, ranking_seed=42, 
                           plot_style="Scatter", show_only_lines=False):
        """Create ranking plot."""
        ranking_df = RankingAnalyzer.calculate_ranking_with_uncertainty(
            plot_df, selected_models, selected_metric, x_axis, 'None',
            n_ranking_samples, ranking_seed, task_name
        )
        
        if ranking_df.empty:
            return None
        
        fig = go.Figure()
        
        for j, model in enumerate(selected_models):
            model_ranking = ranking_df[ranking_df['model'] == model]
            
            if not model_ranking.empty:
                color = self.colors[j % len(self.colors)]
                
                # Plot ranking data points
                if plot_style != "Line Fit" or not show_only_lines:
                    fig.add_trace(go.Scatter(
                        x=model_ranking['x_axis'],
                        y=model_ranking['mean_rank'],
                        mode='markers',
                        name=f"{model} (Rank)",
                        marker=dict(color=color, size=8, line=dict(width=1, color='white')),
                        error_y=dict(
                            type='data',
                            array=model_ranking['std_rank'],
                            visible=True,
                            thickness=1,
                            width=3
                        ),
                        hovertemplate=f'<b>{model}</b><br>{x_axis}: %{{x}}<br>Mean Rank: %{{y}}<br>Rank Std: %{{error_y.array}}<br><extra></extra>'
                    ))
                
                # Add fitted line for ranking analysis if Line Fit is selected
                if plot_style == 'Line Fit' and len(model_ranking) >= 3:
                    sorted_ranking = model_ranking.sort_values('x_axis')
                    x_fit = sorted_ranking['x_axis'].values
                    y_fit = sorted_ranking['mean_rank'].values
                    y_std_fit = sorted_ranking['std_rank'].values
                    
                    fit_result = StatisticalAnalyzer.propagate_uncertainty_linear_regression(
                        x_fit, y_fit, y_std_fit
                    )
                    
                    if fit_result is not None:
                        # Add fitted line
                        fig.add_trace(go.Scatter(
                            x=fit_result['x_pred'],
                            y=fit_result['mean_line'],
                            mode='lines',
                            name=f"{model} (Rank Fitted)",
                            line=dict(color=color, width=3, dash='dash'),
                            hovertemplate=f'<b>{model} (Rank Fitted)</b><br>{x_axis}: %{{x}}<br>Mean Rank: %{{y}}<br><extra></extra>'
                        ))
                        
                        # Add confidence bands
                        if color.startswith('#') and len(color) == 7:
                            r = int(color[1:3], 16)
                            g = int(color[3:5], 16)
                            b = int(color[5:7], 16)
                            rgba_color = f'rgba({r}, {g}, {b}, 0.2)'
                        else:
                            rgba_color = 'rgba(0, 0, 0, 0.2)'
                        
                        fig.add_trace(go.Scatter(
                            x=np.concatenate([fit_result['x_pred'], fit_result['x_pred'][::-1]]),
                            y=np.concatenate([fit_result['pred_upper'], fit_result['pred_lower'][::-1]]),
                            fill='toself',
                            fillcolor=rgba_color,
                            line=dict(color='rgba(0,0,0,0)'),
                            name=f"{model} (Rank 95% CI)",
                            showlegend=False,
                            hoverinfo='skip'
                        ))
        
        # Set inverted y-axis for ranking (best rank = 1 at top)
        y_min, y_max = ranking_df['mean_rank'].min(), ranking_df['mean_rank'].max()
        y_padding = (y_max - y_min) * 0.05
        fig.update_yaxes(range=[y_max + y_padding, y_min - y_padding])
        
        return fig
    
    def create_ranking_plot(self, plot_df, x_axis, selected_models, selected_metric, 
                           task_name, n_ranking_samples=100, ranking_seed=42, 
                           plot_style="Scatter", show_only_lines=False):
        """Create ranking plot."""
        ranking_df = RankingAnalyzer.calculate_ranking_with_uncertainty(
            plot_df, selected_models, selected_metric, x_axis, 'None',
            n_ranking_samples, ranking_seed, task_name
        )
        
        if ranking_df.empty:
            return None
        
        fig = go.Figure()
        
        for j, model in enumerate(selected_models):
            model_ranking = ranking_df[ranking_df['model'] == model]
            
            if not model_ranking.empty:
                color = self.colors[j % len(self.colors)]
                
                # Plot ranking data points
                if plot_style != "Line Fit" or not show_only_lines:
                    fig.add_trace(go.Scatter(
                        x=model_ranking['x_axis'],
                        y=model_ranking['mean_rank'],
                        mode='markers',
                        name=f"{model} (Rank)",
                        marker=dict(color=color, size=8, line=dict(width=1, color='white')),
                        error_y=dict(
                            type='data',
                            array=model_ranking['std_rank'],
                            visible=True,
                            thickness=1,
                            width=3
                        ),
                        hovertemplate=f'<b>{model}</b><br>{x_axis}: %{{x}}<br>Mean Rank: %{{y}}<br>Rank Std: %{{error_y.array}}<br><extra></extra>'
                    ))
                
                # Add fitted line for ranking analysis if Line Fit is selected
                if plot_style == 'Line Fit' and len(model_ranking) >= 3:
                    sorted_ranking = model_ranking.sort_values('x_axis')
                    x_fit = sorted_ranking['x_axis'].values
                    y_fit = sorted_ranking['mean_rank'].values
                    y_std_fit = sorted_ranking['std_rank'].values
                    
                    fit_result = StatisticalAnalyzer.propagate_uncertainty_linear_regression(
                        x_fit, y_fit, y_std_fit
                    )
                    
                    if fit_result is not None:
                        # Add fitted line
                        fig.add_trace(go.Scatter(
                            x=fit_result['x_pred'],
                            y=fit_result['mean_line'],
                            mode='lines',
                            name=f"{model} (Rank Fitted)",
                            line=dict(color=color, width=3, dash='dash'),
                            hovertemplate=f'<b>{model} (Rank Fitted)</b><br>{x_axis}: %{{x}}<br>Mean Rank: %{{y}}<br><extra></extra>'
                        ))
                        
                        # Add confidence bands
                        if color.startswith('#') and len(color) == 7:
                            r = int(color[1:3], 16)
                            g = int(color[3:5], 16)
                            b = int(color[5:7], 16)
                            rgba_color = f'rgba({r}, {g}, {b}, 0.2)'
                        else:
                            rgba_color = 'rgba(0, 0, 0, 0.2)'
                        
                        fig.add_trace(go.Scatter(
                            x=np.concatenate([fit_result['x_pred'], fit_result['x_pred'][::-1]]),
                            y=np.concatenate([fit_result['pred_upper'], fit_result['pred_lower'][::-1]]),
                            fill='toself',
                            fillcolor=rgba_color,
                            line=dict(color='rgba(0,0,0,0)'),
                            name=f"{model} (Rank 95% CI)",
                            showlegend=False,
                            hoverinfo='skip'
                        ))
        
        # Set inverted y-axis for ranking (best rank = 1 at top)
        y_min, y_max = ranking_df['mean_rank'].min(), ranking_df['mean_rank'].max()
        y_padding = (y_max - y_min) * 0.05
        fig.update_yaxes(range=[y_max + y_padding, y_min - y_padding])
        
        return fig
    
class Dashboard:
    """Main dashboard class."""
    
    def __init__(self):
        self.plot_generator = PlotGenerator()
    
    def run(self):
        """Run the main dashboard."""
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
                results_list, error = DataLoader.load_experiment_data(multi_inductive_experiment_dir)
                
                if error:
                    st.error(f"Error loading data: {error}")
                else:
                    st.session_state.results_list = results_list
                    st.session_state.df = DataProcessor.create_dataframe_from_results(
                        results_list, multi_inductive_experiment_dir
                    )
                    st.session_state.experiment_dir = multi_inductive_experiment_dir
                    st.success(f"Successfully loaded {len(results_list)} experiment runs!")
        
        # Main content
        if 'df' in st.session_state and st.session_state.df is not None:
            self._show_main_content()
        else:
            self._show_initial_content()
    
    def _show_main_content(self):
        """Show main dashboard content when data is loaded."""
        df = st.session_state.df
        experiment_dir = st.session_state.experiment_dir
        task_name = DataProcessor.get_task_name_from_dataframe(df)
        
        # Display basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Runs", len(df))
        with col2:
            st.metric("Parameters", len([col for col in df.columns if col.startswith('param_')]))
        with col3:
            st.metric("Model Metrics", len([col for col in df.columns if col.startswith('model')]))
        
        st.markdown("---")
        
        # Interactive plotting section
        st.header("📊 Interactive Analysis")
        
        # Get column options
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        param_columns = [col for col in df.columns if col.startswith(('sweep_', 'param_'))]
        
        # Parameter selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("X-axis Parameter")
            x_axis = st.selectbox(
                "Select X-axis parameter:",
                options=numeric_columns,
                index=0 if numeric_columns else None
            )
        
        with col2:
            st.subheader("Y-axis Metric")
            available_metrics = self._get_available_metrics(df)
            selected_metric = st.selectbox(
                "Select metric type:",
                options=available_metrics,
                index=0 if available_metrics else None
            )
            
            if DataProcessor.is_regression_task(experiment_dir):
                st.info("ℹ️ Normalized metrics (ending with '_normalized') are available for regression tasks.")
        
        # Second parameter selection
        st.subheader("Second Parameter (Optional)")
        second_param = st.selectbox(
            "Select second parameter for subplots/dot size:",
            options=['None'] + param_columns,
            index=0
        )
        
        # Model selection
        st.subheader("Model Selection")
        available_models = self._get_available_models(df, selected_metric)
        selected_models = st.multiselect(
            "Select models to compare:",
            options=available_models,
            default=available_models[:3] if len(available_models) >= 3 else available_models
        )
        
        # Plot controls
        fit_target = st.radio("Fit Target", ["Y-metric", "Ranking"], horizontal=True)
        plot_style = st.radio("Plot Style", ["Scatter", "Line Fit", "Grouped Bar"], horizontal=True)
        
        # Validate grouped bar plot
        if plot_style == "Grouped Bar" and x_axis:
            unique_x_values = df[x_axis].dropna().nunique()
            if unique_x_values >= 10:
                st.warning(f"⚠️ Grouped Bar plot not recommended: {x_axis} has {unique_x_values} unique values (≥10).")
                plot_style = "Scatter"
            else:
                st.success(f"✅ Grouped Bar plot enabled: {x_axis} has {unique_x_values} unique values (<10)")
        
        # Additional controls
        additional_controls = {}
        if plot_style == "Line Fit":
            st.subheader("Curve Fitting Options")
            col1, col2 = st.columns(2)
            with col1:
                additional_controls['fit_type'] = st.selectbox("Fit Type:", ["Linear", "Polynomial"])
                if additional_controls['fit_type'] == "Polynomial":
                    additional_controls['poly_degree'] = st.number_input("Polynomial Degree:", min_value=2, max_value=5, value=2)
            with col2:
                additional_controls['show_only_lines'] = st.checkbox("Show Only Fitted Lines", value=False)
        
        if fit_target == "Ranking":
            st.subheader("Ranking Analysis Settings")
            col1, col2 = st.columns(2)
            with col1:
                additional_controls['n_ranking_samples'] = st.number_input("Number of ranking samples:", min_value=10, max_value=1000, value=100)
            with col2:
                additional_controls['ranking_seed'] = st.number_input("Random seed:", min_value=0, max_value=9999, value=42)
        
        # Generate plot
        if x_axis and selected_metric and selected_models:
            fig = self._generate_plot(df, x_axis, selected_models, selected_metric, task_name,
                                    second_param, fit_target, plot_style, additional_controls)
            
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        self._show_data_table(df)
        
        # Export functionality
        self._show_export_options(df)
    
    def _get_available_metrics(self, df):
        """Get available metrics from dataframe columns."""
        available_metrics = set()
        for col in df.columns:
            if col.startswith('model') and col.endswith('mean'):
                parts = col.split('.')
                if len(parts) >= 4:
                    metric_name = parts[3]
                    available_metrics.add(metric_name)
        return sorted(list(available_metrics), key=lambda x: (x.replace('_normalized', ''), x))
    
    def _get_available_models(self, df, selected_metric):
        """Get available models for the selected metric."""
        if not selected_metric:
            return []
        
        available_models = set()
        for col in df.columns:
            if col.startswith('model') and col.endswith('mean') and f".{selected_metric}.mean" in col:
                parts = col.split('.')
                if len(parts) >= 4:
                    model_name = parts[2]
                    available_models.add(model_name)
        return sorted(list(available_models))
    
    def _generate_plot(self, df, x_axis, selected_models, selected_metric, task_name,
                      second_param, fit_target, plot_style, additional_controls):
        """Generate the appropriate plot based on parameters."""
        # Filter out NaN values
        first_model_col = f"model.{task_name}.{selected_models[0]}.{selected_metric}.mean"
        plot_df = df.dropna(subset=[x_axis, first_model_col])
        
        if len(plot_df) == 0:
            st.warning("No data available for the selected parameters.")
            return None
        
        # Determine if we need subplots
        use_subplots = (second_param != 'None' and 
                       second_param in plot_df.columns and 
                       plot_df[second_param].nunique() <= 10)
        
        if use_subplots:
            return self._generate_subplot_plot(plot_df, x_axis, selected_models, selected_metric, 
                                             task_name, second_param, fit_target, plot_style, additional_controls)
        else:
            return self._generate_single_plot(plot_df, x_axis, selected_models, selected_metric, 
                                            task_name, second_param, fit_target, plot_style, additional_controls)
    
    def _generate_subplot_plot(self, plot_df, x_axis, selected_models, selected_metric, 
                              task_name, second_param, fit_target, plot_style, additional_controls):
        """Generate subplot plot when second parameter is used."""
        unique_values = sorted(plot_df[second_param].unique())
        n_values = len(unique_values)
        
        # Calculate subplot layout
        n_cols = min(3, n_values)
        n_rows = (n_values + n_cols - 1) // n_cols
        
        # Create subplots
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=[f"{second_param}={value}" for value in unique_values],
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
            shared_yaxes=True,
            shared_xaxes=True
        )
        
        # Calculate global y-axis range
        all_y_values = []
        all_ranking_data = []
        
        for value in unique_values:
            filtered_df = plot_df[plot_df[second_param] == value]
            if fit_target == "Ranking":
                ranking_df = RankingAnalyzer.calculate_ranking_with_uncertainty(
                    filtered_df, selected_models, selected_metric, x_axis, second_param,
                    additional_controls.get('n_ranking_samples', 100),
                    additional_controls.get('ranking_seed', 42), task_name
                )
                if not ranking_df.empty:
                    all_ranking_data.extend(ranking_df['mean_rank'].tolist())
            else:
                for model in selected_models:
                    model_metric_col = f"model.{task_name}.{model}.{selected_metric}.mean"
                    if model_metric_col in filtered_df.columns:
                        model_data = filtered_df.dropna(subset=[model_metric_col])
                        if len(model_data) > 0:
                            all_y_values.extend(model_data[model_metric_col].tolist())
        
        # Calculate y-range
        if fit_target == "Ranking":
            if all_ranking_data:
                y_min, y_max = min(all_ranking_data), max(all_ranking_data)
                y_padding = (y_max - y_min) * 0.05
                y_range = [y_max + y_padding, y_min - y_padding]
            else:
                y_range = [len(selected_models) + 0.5, 0.5]
        else:
            if all_y_values:
                y_min, y_max = min(all_y_values), max(all_y_values)
                y_padding = (y_max - y_min) * 0.05
                y_range = [y_min - y_padding, y_max + y_padding]
            else:
                y_range = None
        
        # Generate plots for each subplot
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, value in enumerate(unique_values):
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1
            
            filtered_df = plot_df[plot_df[second_param] == value]
            
            if plot_style == "Grouped Bar":
                self._add_grouped_bar_subplot(fig, filtered_df, x_axis, selected_models, 
                                            selected_metric, task_name, value, row, col, 
                                            fit_target, additional_controls, colors, i == 0, second_param)
            else:
                                self._add_scatter_line_subplot(fig, filtered_df, x_axis, selected_models, 
                                              selected_metric, task_name, value, row, col,
                                              fit_target, plot_style, additional_controls, 
                                              colors, i == 0, second_param)
        
        # Update layout
        y_axis_title = "Model Rank" if fit_target == 'Ranking' else selected_metric.replace('_', ' ').replace('-', ' ').title()
        
        fig.update_layout(
            title=dict(
                text=f"{plot_style}: {selected_metric.replace('_', ' ').replace('-', ' ').title()} vs {x_axis.replace('_', ' ').replace('-', ' ').title()} by {second_param.replace('_', ' ').replace('-', ' ').title()}",
                x=0.5, xanchor='center', font=dict(size=16, color='black')
            ),
            legend=dict(
                x=1.02, y=0.5, xanchor='left', yanchor='middle',
                bgcolor='rgba(255,255,255,0.9)', bordercolor='black', borderwidth=1,
                font=dict(size=11)
            ),
            plot_bgcolor='white', paper_bgcolor='white',
            height=300 * n_rows, margin=dict(l=60, r=150, t=80, b=60)
        )
        
        # Update axes
        for i in range(1, n_rows + 1):
            for j in range(1, n_cols + 1):
                fig.update_xaxes(
                    title=dict(text=x_axis.replace('_', ' ').replace('-', ' ').title(), font=dict(size=12)),
                    showgrid=True, gridwidth=1, gridcolor='lightgray',
                    zeroline=True, zerolinewidth=1, zerolinecolor='black',
                    row=i, col=j
                )
                fig.update_yaxes(
                    title=dict(text=y_axis_title, font=dict(size=12)),
                    showgrid=True, gridwidth=1, gridcolor='lightgray',
                    zeroline=True, zerolinewidth=1, zerolinecolor='black',
                    range=y_range, row=i, col=j
                )
        
        return fig
    
    def _generate_single_plot(self, plot_df, x_axis, selected_models, selected_metric, 
                             task_name, second_param, fit_target, plot_style, additional_controls):
        """Generate single plot without subplots."""
        if plot_style == "Grouped Bar":
            return self.plot_generator.create_grouped_bar_plot(
                plot_df, x_axis, selected_models, selected_metric, fit_target, task_name,
                additional_controls.get('n_ranking_samples', 100),
                additional_controls.get('ranking_seed', 42)
            )
        elif plot_style == "Line Fit":
            return self.plot_generator.create_line_fit_plot(
                plot_df, x_axis, selected_models, selected_metric, task_name,
                additional_controls.get('fit_type', 'Linear'),
                additional_controls.get('poly_degree', 2),
                additional_controls.get('show_only_lines', False)
            )
        else:  # Scatter
            return self.plot_generator.create_scatter_plot(
                plot_df, x_axis, selected_models, selected_metric, task_name, second_param
            )
    
    def _add_grouped_bar_subplot(self, fig, filtered_df, x_axis, selected_models, 
                                 selected_metric, task_name, value, row, col, 
                                 fit_target, additional_controls, colors, show_legend, second_param):
        """Add grouped bar subplot."""
        grouped_df = self.plot_generator._calculate_grouped_bar_data(
            filtered_df, x_axis, selected_models, selected_metric, 
            fit_target, additional_controls.get('n_ranking_samples', 100),
            additional_controls.get('ranking_seed', 42), task_name
        )
        
        if not grouped_df.empty:
            unique_x_vals = sorted(grouped_df['x_value'].unique())
            n_models = len(selected_models)
            bar_width = max(0.15, 0.8 / n_models)
            x_positions = np.arange(len(unique_x_vals))
            
            for j, model in enumerate(selected_models):
                model_data = grouped_df[grouped_df['model'] == model]
                if not model_data.empty:
                    valid_data = model_data.dropna(subset=['mean_value'])
                    if not valid_data.empty:
                        color = colors[j % len(colors)]
                        x_vals = [x for x in unique_x_vals]
                        y_values = []
                        std_values = []
                        for x_val in x_vals:
                            if x_val in valid_data['x_value'].values:
                                row_data = valid_data[valid_data['x_value'] == x_val].iloc[0]
                                y_values.append(row_data['mean_value'])
                                std_values.append(row_data['std_value'])
                            else:
                                y_values.append(None)
                                std_values.append(None)
                        
                        bar_trace = go.Bar(
                            x=x_vals, y=y_values, name=f"{model}",
                            marker_color=color,
                            error_y=dict(type='data', array=std_values, visible=True, thickness=1, width=3),
                            hovertemplate=f'<b>{model}</b><br>{second_param}: {value}<br>{x_axis}: %{{x}}<br>Mean: %{{y}}<br>Std: %{{error_y.array}}<br><extra></extra>',
                            showlegend=show_legend
                        )
                        fig.add_trace(bar_trace, row=row, col=col)
            
            fig.update_xaxes(
                ticktext=[str(x) for x in unique_x_vals],
                tickvals=unique_x_vals,
                row=row, col=col
            )
    
    def _add_scatter_line_subplot(self, fig, filtered_df, x_axis, selected_models, 
                                  selected_metric, task_name, value, row, col,
                                  fit_target, plot_style, additional_controls, 
                                  colors, show_legend, second_param):
        """Add scatter or line fit subplot."""
        for j, model in enumerate(selected_models):
            model_metric_col = f"model.{task_name}.{model}.{selected_metric}.mean"
            model_error_col = f"model.{task_name}.{model}.{selected_metric}.ci95"
            
            if model_metric_col in filtered_df.columns:
                model_data = filtered_df.dropna(subset=[model_metric_col])
                
                if len(model_data) > 0:
                    error_y_data = None
                    if model_error_col in model_data.columns:
                        error_y_data = model_data[model_error_col]
                    
                    color = colors[j % len(colors)]
                    
                    if fit_target == "Ranking":
                        # Handle ranking data
                        ranking_df = RankingAnalyzer.calculate_ranking_with_uncertainty(
                            filtered_df, selected_models, selected_metric, x_axis, 'None',
                            additional_controls.get('n_ranking_samples', 100),
                            additional_controls.get('ranking_seed', 42), task_name
                        )
                        
                        if not ranking_df.empty:
                            model_ranking = ranking_df[ranking_df['model'] == model]
                            if not model_ranking.empty:
                                fig.add_trace(go.Scatter(
                                    x=model_ranking['x_axis'], y=model_ranking['mean_rank'],
                                    mode='markers', name=f"{model} (Rank)",
                                    marker=dict(color=color, size=8, line=dict(width=1, color='white')),
                                    error_y=dict(type='data', array=model_ranking['std_rank'], visible=True, thickness=1, width=3),
                                    hovertemplate=f'<b>{model}</b><br>{second_param}: {value}<br>{x_axis}: %{{x}}<br>Mean Rank: %{{y}}<br>Rank Std: %{{error_y.array}}<br><extra></extra>',
                                    showlegend=show_legend
                                ), row=row, col=col)
                    else:
                        # Handle regular metrics
                        if plot_style == 'Scatter':
                            fig.add_trace(go.Scatter(
                                x=model_data[x_axis], y=model_data[model_metric_col],
                                mode='markers', name=model,
                                marker=dict(color=color, size=8, line=dict(width=1, color='white')),
                                error_y=dict(type='data', array=error_y_data, visible=True, thickness=1, width=3) if error_y_data is not None else None,
                                hovertemplate=f'<b>{model}</b><br>{second_param}: {value}<br>{x_axis}: %{{x}}<br>{selected_metric}: %{{y}}<br><extra></extra>',
                                showlegend=show_legend
                            ), row=row, col=col)
                        elif plot_style == 'Line Fit':
                            # Add data points if not showing only lines
                            if not additional_controls.get('show_only_lines', False):
                                fig.add_trace(go.Scatter(
                                    x=model_data[x_axis], y=model_data[model_metric_col],
                                    mode='markers', name=model,
                                    marker=dict(color=color, size=6, line=dict(width=1, color='white')),
                                    error_y=dict(type='data', array=error_y_data, visible=True, thickness=1, width=3) if error_y_data is not None else None,
                                    hovertemplate=f'<b>{model}</b><br>{second_param}: {value}<br>{x_axis}: %{{x}}<br>{selected_metric}: %{{y}}<br><extra></extra>',
                                    showlegend=show_legend
                                ), row=row, col=col)
                            
                            # Add fitted line
                            if len(model_data) >= 3:
                                sorted_data = model_data.sort_values(x_axis)
                                x_fit = sorted_data[x_axis].values
                                y_fit = sorted_data[model_metric_col].values
                                y_std_fit = sorted_data[model_error_col].values / 1.96 if model_error_col in sorted_data.columns else np.ones_like(y_fit) * 0.01
                                
                                fit_type = additional_controls.get('fit_type', 'Linear')
                                if fit_type == "Linear":
                                    fit_result = StatisticalAnalyzer.propagate_uncertainty_linear_regression(x_fit, y_fit, y_std_fit)
                                else:
                                    poly_degree = additional_controls.get('poly_degree', 2)
                                    fit_result = StatisticalAnalyzer.propagate_uncertainty_polynomial_regression(x_fit, y_fit, y_std_fit, poly_degree)
                                
                                if fit_result is not None:
                                    fig.add_trace(go.Scatter(
                                        x=fit_result['x_pred'], y=fit_result['mean_line'],
                                        mode='lines', name=f"{model} (Fitted)",
                                        line=dict(color=color, width=3, dash='dash'),
                                        hovertemplate=f'<b>{model} (Fitted)</b><br>{second_param}: {value}<br>{x_axis}: %{{x}}<br>{selected_metric}: %{{y}}<br><extra></extra>',
                                        showlegend=show_legend
                                    ), row=row, col=col)
                                
                                # Add confidence bands
                                if color.startswith('#') and len(color) == 7:
                                    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                                    rgba_color = f'rgba({r}, {g}, {b}, 0.2)'
                                else:
                                    rgba_color = 'rgba(0,0,0,0.2)'
                                
                                fig.add_trace(go.Scatter(
                                    x=np.concatenate([fit_result['x_pred'], fit_result['x_pred'][::-1]]),
                                    y=np.concatenate([fit_result['pred_upper'], fit_result['pred_lower'][::-1]]),
                                    fill='toself', fillcolor=rgba_color,
                                    line=dict(color='rgba(0,0,0,0)'),
                                    name=f"{model} (95% CI)", showlegend=False, hoverinfo='skip'
                                ), row=row, col=col)
    
    def _show_data_table(self, df):
        """Show data table section."""
        st.header("📋 Data Table")
        
        columns_to_show = st.multiselect(
            "Select columns to display:",
            options=df.columns.tolist(),
            default=['run_id', 'method'] + [col for col in df.columns if col.startswith('param_')][:5]
        )
        
        if columns_to_show:
            st.dataframe(df[columns_to_show], use_container_width=True)
    
    def _show_export_options(self, df):
        """Show export functionality."""
        st.header("💾 Export Data")
        
        if st.button("📥 Download as CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="experiment_results.csv",
                mime="text/csv"
            )
    
    def _show_initial_content(self):
        """Show initial content when no data is loaded."""
        st.info("👆 Please select your experiment directory and click 'Load Experiment Data' to begin analysis.")
        
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

# Main execution
if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.run()