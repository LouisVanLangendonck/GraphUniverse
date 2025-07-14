import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from scipy import stats

# Page config
st.set_page_config(
    page_title="GNN Experiment Results Analyzer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 GNN Experiment Results Analyzer")
st.markdown("---")

@st.cache_data
def get_result_family_properties(multi_inductive_experiment_dir, run_sample):
    """Extract family properties from results.json for a given run."""
    run = run_sample['data_files']['inductive_data'].split("/")[0]
    run_path = os.path.join(multi_inductive_experiment_dir, run)
    
    # Find the non-empty directory in the run_path
    for dir_name in os.listdir(run_path):
        dir_path = os.path.join(run_path, dir_name)
        if os.path.isdir(dir_path) and dir_name != "" and dir_name != "data_analysis_report.txt":
            results_file = os.path.join(dir_path, "results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                return results['family_properties']
    return None

@st.cache_data
def add_family_properties_to_run_sample(multi_inductive_experiment_dir, run_sample):
    """Add family properties to run sample."""
    family_properties = get_result_family_properties(multi_inductive_experiment_dir, run_sample)
    if family_properties:
        run_sample['family_properties'] = family_properties
    return run_sample

@st.cache_data
def load_experiment_data(multi_inductive_experiment_dir):
    """Load and process experiment data."""
    final_results_path = os.path.join(multi_inductive_experiment_dir, "final_results.json")
    
    if not os.path.exists(final_results_path):
        return None, "final_results.json not found in the specified directory"
    
    try:
        with open(final_results_path, 'r') as f:
            data = json.load(f)
        
        # Process all run samples
        results_list_with_family_properties = []
        for run_sample in data['all_results']:
            run_sample = add_family_properties_to_run_sample(multi_inductive_experiment_dir, run_sample)
            results_list_with_family_properties.append(run_sample)
        
        return results_list_with_family_properties, None
    except Exception as e:
        return None, str(e)

def flatten_nested_dict(d, parent_key='', sep='_'):
    """Flatten nested dictionary structure."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def is_regression_task(experiment_dir):
    """Check if the experiment is a regression task (not community detection)."""
    return experiment_dir != "multi_results/community_detect"

def get_task_name(experiment_dir):
    """Get the task name based on the experiment directory."""
    if is_regression_task(experiment_dir):
        # For regression tasks, we need to determine the task name dynamically
        # This will be determined from the actual data structure
        return None  # Will be determined dynamically
    else:
        # For community detection tasks, use "community"
        return "community"

def get_task_name_from_dataframe(df):
    """Extract task name from dataframe columns."""
    task_names = set()
    for col in df.columns:
        if col.startswith('model.') and col.endswith('.mean'):
            parts = col.split('.')
            if len(parts) >= 4:  # model.task.model.metric.mean
                task_names.add(parts[1])  # The task name part
    
    # Return the first task name found, or "community" as fallback
    return list(task_names)[0] if task_names else "community"

def extract_model_metrics(model_results, experiment_dir=None, family_properties=None):
    """Extract model performance metrics with fold averaging."""
    metrics = {}
    if not model_results:
        return metrics
    
    # Check if this is a regression task
    is_regression = is_regression_task(experiment_dir) if experiment_dir else False
    
    # Get mean density for normalization if available
    mean_density = None
    if is_regression and family_properties:
        mean_density = family_properties['densities_mean']

    
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
                                # 95% confidence interval (1.96 * std / sqrt(n))
                                std_val = np.std(values)
                                n_folds = len(values)
                                ci_95 = 1.96 * std_val / np.sqrt(n_folds)
                                
                                # For regression tasks, negate MSE values to make maximum = best
                                if is_regression and metric_name.lower() in ['mse', 'rmse', 'mae']:
                                    mean_val = -mean_val
                                    # Note: CI95 remains positive as it's a measure of uncertainty
                                
                                key_mean = f"model.{task_name}.{model_name}.{metric_name}.mean"
                                key_ci = f"model.{task_name}.{model_name}.{metric_name}.ci95"
                                metrics[key_mean] = mean_val
                                metrics[key_ci] = ci_95
                                
                                # Add normalized version for regression tasks if mean_density is available
                                if is_regression and mean_density is not None and mean_density > 0:
                                    # Create normalized version (divide by mean density)
                                    normalized_mean = mean_val / mean_density
                                    # # For normalized metrics, we also negate to keep "higher is better"
                                    # if metric_name.lower() in ['mse', 'rmse', 'mae']:
                                    #     normalized_mean = -normalized_mean
                                    
                                    # Calculate normalized CI (propagate uncertainty)
                                    normalized_ci = ci_95 / mean_density
                                    
                                    key_normalized_mean = f"model.{task_name}.{model_name}.{metric_name}_normalized.mean"
                                    key_normalized_ci = f"model.{task_name}.{model_name}.{metric_name}_normalized.ci95"
                                    metrics[key_normalized_mean] = normalized_mean
                                    metrics[key_normalized_ci] = normalized_ci
                                    
                                    # print(f"DEBUG: Created normalized metric: {key_normalized_mean} = {normalized_mean}")
                                else:
                                    # print(f"DEBUG: Skipped normalized metric for {metric_name}. is_regression={is_regression}, mean_density={mean_density}")
                                    pass
                    
                    # Fallback for non-fold structure
                    else:
                        for metric_name, metric_value in model_data.items():
                            if isinstance(metric_value, (int, float)):
                                # For regression tasks, negate MSE values to make maximum = best
                                if is_regression and metric_name.lower() in ['mse', 'rmse', 'mae']:
                                    metric_value = -metric_value
                                
                                key = f"model.{task_name}.{model_name}.{metric_name}"
                                metrics[key] = metric_value
                                
                                # Add normalized version for regression tasks if mean_density is available
                                if is_regression and mean_density is not None and mean_density > 0:
                                    # Create normalized version (divide by mean density)
                                    normalized_value = metric_value / mean_density
                                    # For normalized metrics, we also negate to keep "higher is better"
                                    if metric_name.lower() in ['mse', 'rmse', 'mae']:
                                        normalized_value = -normalized_value
                                    
                                    key_normalized = f"model.{task_name}.{model_name}.{metric_name}_normalized"
                                    metrics[key_normalized] = normalized_value
    return metrics

def calculate_ranking_with_uncertainty(df, selected_models, selected_metric, x_axis, second_param='None', n_samples=100, seed=42, task_name='community'):
    """Calculate ranking with uncertainty by sampling from normal distributions."""
    np.random.seed(seed)
    
    ranking_data = []
    
    for idx, row in df.iterrows():
        # Collect model performance data for this run
        model_performances = {}
        model_uncertainties = {}
        
        for model in selected_models:
            mean_col = f"model.{task_name}.{model}.{selected_metric}.mean"
            std_col = f"model.{task_name}.{model}.{selected_metric}.ci95"
            
            if mean_col in row and not pd.isna(row[mean_col]):
                model_performances[model] = row[mean_col]
                # Use CI95 as uncertainty (convert to std if needed)
                if std_col in row and not pd.isna(row[std_col]):
                    model_uncertainties[model] = row[std_col] / 1.96  # Convert CI95 to std
                else:
                    model_uncertainties[model] = 0.01  # Small default uncertainty
        
        if len(model_performances) > 1:
            # Check if we have meaningful variation in performance
            performance_values = list(model_performances.values())
            if np.std(performance_values) < 1e-10:  # Very small variation
                # All models have essentially the same performance, assign equal ranks
                equal_rank = (len(model_performances) + 1) / 2
                for model in model_performances.keys():
                    ranking_data.append({
                        'run_id': row.get('run_id', idx),
                        'model': model,
                        'mean_rank': equal_rank,
                        'std_rank': 0.0,  # No uncertainty when all models are equal
                        'x_axis': row.get(x_axis, 0),
                        'second_param': row.get(second_param, None) if second_param != 'None' else None
                    })
            else:
                # Sample rankings multiple times
                rankings = []
                for _ in range(n_samples):
                    # Sample from normal distributions
                    sampled_performances = {}
                    for model, mean in model_performances.items():
                        std = model_uncertainties.get(model, 0.01)
                        sampled_performances[model] = np.random.normal(mean, std)
                    
                    # Calculate ranking (best performance = rank 1)
                    sorted_models = sorted(sampled_performances.items(), key=lambda x: x[1], reverse=True)
                    ranking = {model: rank + 1 for rank, (model, _) in enumerate(sorted_models)}
                    rankings.append(ranking)
                
                # Calculate mean and std of rankings for each model
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
    
    # Check for sufficient variation in the data
    if np.std(x_clean) == 0 or np.std(y_mean_clean) == 0:
        # No variation in data, cannot fit meaningful regression
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
            
            # Calculate correlation with proper error handling for constant arrays
            try:
                corr, _ = stats.pearsonr(x_boot, y_boot)
                if np.isnan(corr):
                    corr = 0.0  # Set correlation to 0 if undefined
            except (ValueError, RuntimeWarning):
                corr = 0.0  # Set correlation to 0 if calculation fails
            
            if not (np.isnan(slope) or np.isnan(intercept) or np.isnan(r_squared)):
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
    try:
        se_pred = effective_std * np.sqrt(1 + 1/n + (x_pred - x_mean)**2 / sxx)
    except ZeroDivisionError:
        # If sxx is zero (no variation in x), use a default uncertainty
        se_pred = effective_std * np.ones_like(x_pred)
    
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
    try:
        corr_t_stat = mean_correlation * np.sqrt((n - 2) / (1 - mean_correlation**2)) if abs(mean_correlation) < 0.99 else 0
        corr_p_value_classical = 2 * (1 - stats.t.cdf(abs(corr_t_stat), n - 2))
    except (ValueError, ZeroDivisionError):
        corr_t_stat = 0
        corr_p_value_classical = 1.0  # Default to non-significant
    
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

def propagate_uncertainty_polynomial_regression(x, y_mean, y_std, degree=2, x_pred=None):
    """
    Fit polynomial regression with proper uncertainty propagation using bootstrap methods.
    
    Returns fitted curve with confidence bands and statistical measures.
    """
    n_bootstrap = 1000  # Bootstrap samples
    
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
        # No variation in data, cannot fit meaningful regression
        return None
    
    # Method 1: Bootstrap resampling of data points with uncertainty
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
        
        # Add family properties (focusing on summary statistics)
        if 'family_properties' in run_sample:
            fp = run_sample['family_properties']
            # Add mean, std, min, max values
            for key, value in fp.items():
                if key.endswith(('_mean', '_std', '_min', '_max')):
                    row[f"family_{key}"] = value
        
        # Add model results
        if 'model_results' in run_sample:
            # Get family properties for normalization
            family_properties = run_sample.get('family_properties', {})
            model_metrics = extract_model_metrics(run_sample['model_results'], experiment_dir, family_properties)
            for key, value in model_metrics.items():
                row[key] = value  # Don't add extra "model_" prefix since extract_model_metrics already includes it
        
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

def calculate_grouped_bar_data(df, x_axis, selected_models, selected_metric, fit_target="Y-metric", n_ranking_samples=100, ranking_seed=42, task_name='community'):
    """
    Calculate grouped bar data by averaging results across runs for each unique x-axis value.
    Returns data suitable for grouped bar plotting.
    """
    grouped_data = []
    
    # Get unique x-axis values
    unique_x_values = sorted(df[x_axis].dropna().unique())
    
    for x_val in unique_x_values:
        # Filter data for this x-axis value
        filtered_df = df[df[x_axis] == x_val]
        
        if len(filtered_df) == 0:
            continue
            
        for model in selected_models:
            if fit_target == "Ranking":
                # Calculate ranking for this subset
                ranking_df = calculate_ranking_with_uncertainty(
                    filtered_df, selected_models, selected_metric, x_axis, 'None',
                    n_ranking_samples, ranking_seed, task_name
                )
                
                if not ranking_df.empty:
                    model_ranking = ranking_df[ranking_df['model'] == model]
                    if not model_ranking.empty:
                        # Average the ranking across runs for this x-value
                        mean_rank = model_ranking['mean_rank'].mean()
                        # Propagate uncertainty: sqrt(sum of variances) / n
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
                        # Add placeholder for missing model data
                        grouped_data.append({
                            'x_value': x_val,
                            'model': model,
                            'mean_value': np.nan,
                            'std_value': np.nan,
                            'metric_type': 'ranking'
                        })
            else:
                # Calculate metric averages for this subset
                model_metric_col = f"model.{task_name}.{model}.{selected_metric}.mean"
                model_error_col = f"model.{task_name}.{model}.{selected_metric}.ci95"
                
                if model_metric_col in filtered_df.columns:
                    model_data = filtered_df.dropna(subset=[model_metric_col])
                    
                    if len(model_data) > 0:
                        # Average the metric across runs for this x-value
                        mean_metric = model_data[model_metric_col].mean()
                        
                        # Propagate uncertainty: sqrt(sum of variances) / n
                        if model_error_col in model_data.columns:
                            # Convert CI95 to std for uncertainty propagation
                            metric_stds = model_data[model_error_col] / 1.96
                            metric_vars = metric_stds**2
                            mean_metric_std = np.sqrt(np.sum(metric_vars)) / len(model_data)
                        else:
                            # Use standard deviation of the means if no CI available
                            mean_metric_std = model_data[model_metric_col].std() / np.sqrt(len(model_data))
                        
                        grouped_data.append({
                            'x_value': x_val,
                            'model': model,
                            'mean_value': mean_metric,
                            'std_value': mean_metric_std,
                            'metric_type': 'metric'
                        })
                    else:
                        # Add placeholder for missing model data
                        grouped_data.append({
                            'x_value': x_val,
                            'model': model,
                            'mean_value': np.nan,
                            'std_value': np.nan,
                            'metric_type': 'metric'
                        })
                else:
                    # Add placeholder for missing model column
                    grouped_data.append({
                        'x_value': x_val,
                        'model': model,
                        'mean_value': np.nan,
                        'std_value': np.nan,
                        'metric_type': 'metric'
                    })
    
    return pd.DataFrame(grouped_data)

# Sidebar for directory selection
st.sidebar.header("📁 Directory Selection")

# Directory input
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
    index=1,  # Default to regression task to show normalized metrics
    help="Select the experiment directory to analyze"
)

# Load data button
if st.sidebar.button("🔄 Load Experiment Data"):
    with st.spinner("Loading experiment data..."):
        results_list, error = load_experiment_data(multi_inductive_experiment_dir)
        
        if error:
            st.error(f"Error loading data: {error}")
        else:
            st.session_state.results_list = results_list
            st.session_state.df = create_dataframe_from_results(results_list, multi_inductive_experiment_dir)
            st.success(f"Successfully loaded {len(results_list)} experiment runs!")

# Main content
if 'df' in st.session_state and st.session_state.df is not None:
    df = st.session_state.df
    
    # Determine task name dynamically
    task_name = get_task_name_from_dataframe(df)
    
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
    
    # Column selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("X-axis Parameter")
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        x_axis = st.selectbox(
            "Select X-axis parameter:",
            options=numeric_columns,
            index=0 if numeric_columns else None
        )
    
    # Second parameter selection
    st.subheader("Second Parameter (Optional)")
    # Get parameter columns (sweep and param columns)
    param_columns = [col for col in df.columns if col.startswith(('sweep_', 'param_'))]
    second_param = st.selectbox(
        "Select second parameter for subplots/dot size:",
        options=['None'] + param_columns,
        index=0
    )
    
    with col2:
        st.subheader("Y-axis Metric")
        # Extract available metrics from model columns
        available_metrics = set()
        
        for col in df.columns:
            if col.startswith('model') and col.endswith('mean'):
                # Extract metric name from the structure
                parts = col.split('.')
                if len(parts) >= 4:  # model.task.model.metric.mean
                    metric_name = parts[3]  # The metric name part
                    available_metrics.add(metric_name)
                elif len(parts) >= 3:  # Try alternative structure
                    metric_name = parts[-2]  # Second to last part before 'mean'
                    available_metrics.add(metric_name)
        
        # Sort metrics to show normalized versions after regular versions
        metric_options = sorted(list(available_metrics), key=lambda x: (x.replace('_normalized', ''), x))
        
        selected_metric = st.selectbox(
            "Select metric type:",
            options=metric_options,
            index=0 if metric_options else None
        )
        
        # Show info about normalized metrics
        if is_regression_task(multi_inductive_experiment_dir):
            st.info("ℹ️ Normalized metrics (ending with '_normalized') are available for regression tasks. These metrics are divided by the mean edge density to make them more comparable across different graph densities.")
        else:
            st.info("ℹ️ Normalized metrics are only available for regression tasks. Switch to a regression experiment directory to see them.")
    
    # Model selection
    st.subheader("Model Selection")
    if selected_metric:
        # Extract available models for the selected metric
        available_models = set()
        for col in df.columns:
            if col.startswith('model') and col.endswith('mean'):
                # Check if this column contains the selected metric
                if f".{selected_metric}.mean" in col:
                    # Extract model name from the structure
                    parts = col.split('.')
                    if len(parts) >= 4:  # model.task.model.metric.mean
                        model_name = parts[2]  # The model name part
                        available_models.add(model_name)
                    elif len(parts) >= 3:  # Try alternative structure
                        model_name = parts[1]  # The model name part
                        available_models.add(model_name)
        
        model_options = sorted(list(available_models))
        
        if model_options:
            selected_models = st.multiselect(
                "Select models to compare:",
                options=model_options,
                default=model_options[:3] if len(model_options) >= 3 else model_options
            )
        else:
            selected_models = []
    else:
        selected_models = []
    
    # Color coding option
    color_by = st.selectbox(
        "Color by (optional):",
        options=['None'] + [col for col in df.columns if col.startswith(('param_', 'sweep_', 'method'))],
        index=0
    )
    
    # 1. Add new controls above the plot
    fit_target = st.radio("Fit Target", ["Y-metric", "Ranking"], horizontal=True)
    plot_style = st.radio("Plot Style", ["Scatter", "Line Fit", "Grouped Bar"], horizontal=True)
    
    # Check if grouped bar is possible based on x-axis selection
    if plot_style == "Grouped Bar" and x_axis:
        unique_x_values = df[x_axis].dropna().nunique()
        if unique_x_values >= 10:
            st.warning(f"⚠️ Grouped Bar plot is not possible: {x_axis} has {unique_x_values} unique values (≥10). Please select a parameter with fewer unique values or choose a different plot style.")
            plot_style = "Scatter"  # Fallback to scatter
        else:
            st.success(f"✅ Grouped Bar plot enabled: {x_axis} has {unique_x_values} unique values (<10)")
    
    # Curve fitting options
    if plot_style == "Line Fit":
        st.subheader("Curve Fitting Options")
        col1, col2, col3 = st.columns(3)
        with col1:
            fit_type = st.selectbox(
                "Fit Type:",
                options=["Linear", "Polynomial"],
                help="Choose between linear or polynomial curve fitting"
            )
        with col2:
            if fit_type == "Polynomial":
                poly_degree = st.number_input(
                    "Polynomial Degree:",
                    min_value=2,
                    max_value=5,
                    value=2,
                    help="Degree of polynomial for curve fitting"
                )
            else:
                poly_degree = 1
        with col3:
            show_only_lines = st.checkbox(
                "Show Only Fitted Lines",
                value=False,
                help="Hide data points and show only the fitted curves"
            )

    # Ranking analysis options
    # Set default values for ranking parameters
    n_ranking_samples = 100
    ranking_seed = 42
    
    if fit_target == "Ranking":
        st.subheader("Ranking Analysis Settings")
        col1, col2 = st.columns(2)
        with col1:
            n_ranking_samples = st.number_input(
                "Number of ranking samples:",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                help="Number of times to sample from normal distribution to calculate ranking uncertainty"
            )
        with col2:
            ranking_seed = st.number_input(
                "Random seed:",
                min_value=0,
                max_value=9999,
                value=42,
                help="Seed for reproducible ranking calculations"
            )
    
    # Generate plot
    if x_axis and selected_metric and selected_models:
        fig = None
        
        # Filter out NaN values for the first model to check data availability
        first_model_col = f"model.{task_name}.{selected_models[0]}.{selected_metric}.mean"
        plot_df = df.dropna(subset=[x_axis, first_model_col])
        
        if len(plot_df) == 0:
            st.warning("No data available for the selected parameters.")
        else:

            color_param = None if color_by == 'None' else color_by
            
            # Handle second parameter
            if second_param != 'None' and second_param in plot_df.columns:
                unique_values = plot_df[second_param].unique()
                n_values = len(unique_values)
                
                if n_values <= 10:
                    # Create actual subplots for each value
                    from plotly.subplots import make_subplots
                    
                    # Calculate subplot layout
                    n_cols = min(3, n_values)
                    n_rows = (n_values + n_cols - 1) // n_cols
                    
                    # Create subplots
                    fig = make_subplots(
                        rows=n_rows, 
                        cols=n_cols,
                        subplot_titles=[f"{second_param}={value}" for value in sorted(unique_values)],
                        vertical_spacing=0.1,
                        horizontal_spacing=0.1,
                        shared_yaxes=True,
                        shared_xaxes=True
                    )
                    
                    # Handle Grouped Bar plots for subplots
                    if plot_style == "Grouped Bar":
                        # Calculate global y-axis range across all subplots first
                        all_y_values = []
                        all_ranking_data = []
                        
                        for value in sorted(unique_values):
                            filtered_df = plot_df[plot_df[second_param] == value]
                            grouped_df = calculate_grouped_bar_data(
                                filtered_df, x_axis, selected_models, selected_metric, 
                                fit_target, n_ranking_samples, ranking_seed, task_name
                            )
                            
                            if not grouped_df.empty:
                                if fit_target == "Ranking":
                                    # For ranking analysis, collect ranking data
                                    valid_data = grouped_df.dropna(subset=['mean_value'])
                                    if not valid_data.empty:
                                        all_ranking_data.extend(valid_data['mean_value'].tolist())
                                else:
                                    # For regular metrics, collect metric data
                                    valid_data = grouped_df.dropna(subset=['mean_value'])
                                    if not valid_data.empty:
                                        all_y_values.extend(valid_data['mean_value'].tolist())
                        
                    # Calculate global y-range
                    if fit_target == "Ranking":
                        if all_ranking_data:
                            y_min, y_max = min(all_ranking_data), max(all_ranking_data)
                            y_padding = (y_max - y_min) * 0.05
                            y_range = [y_max + y_padding, y_min - y_padding]  # Inverted for ranking
                        else:
                            y_range = [len(selected_models) + 0.5, 0.5]  # Default ranking range
                    else:
                        if all_y_values:
                            y_min, y_max = min(all_y_values), max(all_y_values)
                            y_padding = (y_max - y_min) * 0.05
                            y_range = [y_min - y_padding, y_max + y_padding]
                        else:
                            y_range = [0, 1]  # Default range
                    
                    # Create subplots using make_subplots
                    n_cols = min(3, len(unique_values))
                    n_rows = (len(unique_values) + n_cols - 1) // n_cols
                    
                    fig = make_subplots(
                        rows=n_rows, 
                        cols=n_cols,
                        subplot_titles=[f"{second_param}={value}" for value in sorted(unique_values)],
                        vertical_spacing=0.1,
                        horizontal_spacing=0.1,
                        shared_yaxes=True,
                        shared_xaxes=True
                    )
                    
                    # Add bars to each subplot
                    for i, value in enumerate(sorted(unique_values)):
                        # Calculate subplot position
                        row = (i // n_cols) + 1
                        col = (i % n_cols) + 1
                        
                        # Filter data for this value
                        filtered_df = plot_df[plot_df[second_param] == value]
                        
                        # Calculate grouped bar data for this subplot
                        grouped_df = calculate_grouped_bar_data(
                            filtered_df, x_axis, selected_models, selected_metric, 
                            fit_target, n_ranking_samples, ranking_seed, task_name
                        )
                        
                        if not grouped_df.empty:
                            # Create grouped bar chart for this subplot
                            unique_x_vals = sorted(grouped_df['x_value'].unique())
                            n_models = len(selected_models)
                            
                            # Calculate bar positions
                            bar_width = max(0.15, 0.8 / n_models)  # Ensure minimum bar width
                            x_positions = np.arange(len(unique_x_vals))
                            
                            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                            
                            for j, model in enumerate(selected_models):
                                model_data = grouped_df[grouped_df['model'] == model]
                                if not model_data.empty:
                                    valid_data = model_data.dropna(subset=['mean_value'])
                                    if not valid_data.empty:
                                        color = colors[j % len(colors)]
                                        # Use actual categorical x-values for bars
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
                                            x=x_vals,
                                            y=y_values,
                                            name=f"{model}",
                                            marker_color=color,
                                            error_y=dict(
                                                type='data',
                                                array=std_values,
                                                visible=True,
                                                thickness=1,
                                                width=3
                                            ),
                                            hovertemplate=f'<b>{model}</b><br>' +
                                                        f'{second_param}: {value}<br>' +
                                                        f'{x_axis}: %{{x}}<br>' +
                                                        f'Mean: %{{y}}<br>' +
                                                        f'Std: %{{error_y.array}}<br>' +
                                                        '<extra></extra>',
                                            showlegend=(i == 0)  # Only show legend for first subplot
                                        )
                                        fig.add_trace(bar_trace, row=row, col=col)
                            # Set x-axis ticks to actual categorical values
                            fig.update_xaxes(
                                ticktext=[str(x) for x in unique_x_vals],
                                tickvals=unique_x_vals,
                                row=row, col=col
                            )
                    
                    # Update layout for grouped bar subplots
                    y_axis_title = "Model Rank" if fit_target == 'Ranking' else selected_metric.replace('_', ' ').replace('-', ' ').title()
                    
                    fig.update_layout(
                        title=dict(
                            text=f"Grouped Bar: {selected_metric.replace('_', ' ').replace('-', ' ').title()} vs {x_axis.replace('_', ' ').replace('-', ' ').title()} by {second_param.replace('_', ' ').replace('-', ' ').title()}",
                            x=0.5,
                            xanchor='center',
                            font=dict(size=16, color='black')
                        ),
                        legend=dict(
                            x=1.02,
                            y=0.5,
                            xanchor='left',
                            yanchor='middle',
                            bgcolor='rgba(255,255,255,0.9)',
                            bordercolor='black',
                            borderwidth=1,
                            font=dict(size=11)
                        ),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=300 * n_rows,
                        margin=dict(l=60, r=150, t=100, b=60),  # Increase top margin
                        title_font_size=18
                    )
                    
                    # Update all subplot axes with proper x-ranges for grouped bars
                    for i in range(1, n_rows + 1):
                        for j in range(1, n_cols + 1):
                            # Calculate appropriate x-range for grouped bars based on actual data
                            subplot_idx = (i-1) * n_cols + (j-1)
                            if subplot_idx < len(unique_values):
                                subplot_value = sorted(unique_values)[subplot_idx]
                                subplot_filtered_df = plot_df[plot_df[second_param] == subplot_value]
                                subplot_grouped_df = calculate_grouped_bar_data(
                                    subplot_filtered_df, x_axis, selected_models, selected_metric, 
                                    fit_target, n_ranking_samples, ranking_seed, task_name
                                )
                                subplot_unique_x_vals = sorted(subplot_grouped_df['x_value'].unique()) if not subplot_grouped_df.empty else []
                                
                                # Calculate proper x-range based on actual number of x-values
                                if subplot_unique_x_vals:
                                    bar_x_min = -0.5
                                    bar_x_max = len(subplot_unique_x_vals) - 0.5
                                else:
                                    bar_x_min = -0.5
                                    bar_x_max = 0.5
                            else:
                                # Empty subplot
                                bar_x_min = -0.5
                                bar_x_max = 0.5
                            
                            fig.update_xaxes(
                                title=dict(text=x_axis.replace('_', ' ').replace('-', ' ').title(), font=dict(size=12)),
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='lightgray',
                                zeroline=True,
                                zerolinewidth=1,
                                zerolinecolor='black',
                                range=[bar_x_min, bar_x_max],
                                row=i, col=j
                            )
                            fig.update_yaxes(
                                title=dict(text=y_axis_title, font=dict(size=12)),
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='lightgray',
                                zeroline=True,
                                zerolinewidth=1,
                                zerolinecolor='black',
                                range=y_range,
                                row=i, col=j
                            )
                        # Calculate subplot position
                        row = (i // n_cols) + 1
                        col = (i % n_cols) + 1
                        
                        # Filter data for this value
                        filtered_df = plot_df[plot_df[second_param] == value]
                        
                        # Calculate grouped bar data for this subplot
                        grouped_df = calculate_grouped_bar_data(
                            filtered_df, x_axis, selected_models, selected_metric, 
                            fit_target, n_ranking_samples, ranking_seed
                        )
                        
                        if not grouped_df.empty:
                            # Create grouped bar chart for this subplot
                            unique_x_vals = sorted(grouped_df['x_value'].unique())
                            n_models = len(selected_models)
                            
                            # Calculate bar positions
                            bar_width = max(0.15, 0.8 / n_models)  # Ensure minimum bar width
                            x_positions = np.arange(len(unique_x_vals))
                            
                            # Debug: Show subplot info
                            
                            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                            
                            for j, model in enumerate(selected_models):
                                model_data = grouped_df[grouped_df['model'] == model]
                                
                                if not model_data.empty:
                                    # Filter out NaN values for plotting
                                    valid_data = model_data.dropna(subset=['mean_value'])
                                    
                                    if not valid_data.empty:
                                        color = colors[j % len(colors)]
                                        
                                        # Calculate bar positions for this model
                                        # Ensure slight offset to avoid exact integer positions
                                        offset = (j - (n_models-1)/2) * bar_width
                                        bar_positions = x_positions + offset
                                        
                                        # Add tiny offset to avoid exact integer positions
                                        if np.any(np.abs(bar_positions - np.round(bar_positions)) < 1e-10):
                                            bar_positions = bar_positions + 1e-6
                                        
                                        # Map data to bar positions correctly
                                        # Create a mapping from x_value to position index
                                        x_to_pos = {x_val: pos for pos, x_val in enumerate(unique_x_vals)}
                                        
                                        # Get the values in the correct order
                                        y_values = []
                                        std_values = []
                                        for x_val in unique_x_vals:
                                            if x_val in valid_data['x_value'].values:
                                                row_data = valid_data[valid_data['x_value'] == x_val].iloc[0]
                                                y_values.append(row_data['mean_value'])
                                                std_values.append(row_data['std_value'])
                                            else:
                                                y_values.append(np.nan)
                                                std_values.append(np.nan)
                                        
                                        # Debug: Show bar plotting info
                                        
                                        # Add bar trace
                                        bar_trace = go.Bar(
                                            x=bar_positions,
                                            y=y_values,
                                            name=f"{model}",
                                            marker_color=color,
                                            error_y=dict(
                                                type='data',
                                                array=std_values,
                                                visible=True,
                                                thickness=1,
                                                width=3
                                            ),
                                            hovertemplate=f'<b>{model}</b><br>' +
                                                            f'{second_param}: {value}<br>' +
                                                            f'{x_axis}: %{{x}}<br>' +
                                                            f'Mean: %{{y}}<br>' +
                                                            f'Std: %{{error_y.array}}<br>' +
                                                            '<extra></extra>',
                                            showlegend=(i == 0)  # Only show legend for first subplot
                                        )
                                        
                                        fig.add_trace(bar_trace, row=row, col=col)
                                        
                                        
                                    else:
                                        pass
                            # Update x-axis labels
                            fig.update_xaxes(
                                ticktext=[str(x) for x in unique_x_vals],
                                tickvals=x_positions,
                                row=row, col=col
                            )
                    
                    # Update layout for grouped bar subplots
                    y_axis_title = "Model Rank" if fit_target == 'Ranking' else selected_metric.replace('_', ' ').replace('-', ' ').title()
                    
                    fig.update_layout(
                        title=dict(
                            text=f"Grouped Bar: {selected_metric.replace('_', ' ').replace('-', ' ').title()} vs {x_axis.replace('_', ' ').replace('-', ' ').title()} by {second_param.replace('_', ' ').replace('-', ' ').title()}",
                            x=0.5,
                            xanchor='center',
                            font=dict(size=16, color='black')
                        ),
                        legend=dict(
                            x=1.02,
                            y=0.5,
                            xanchor='left',
                            yanchor='middle',
                            bgcolor='rgba(255,255,255,0.9)',
                            bordercolor='black',
                            borderwidth=1,
                            font=dict(size=11)
                        ),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=300 * n_rows,
                        margin=dict(l=60, r=150, t=80, b=60)
                    )
                    
                    # Update all subplot axes
                    for i in range(1, n_rows + 1):
                        for j in range(1, n_cols + 1):
                            # Calculate appropriate x-range for grouped bars
                            # Get the unique_x_vals for this specific subplot
                            subplot_value = sorted(unique_values)[(i-1) * n_cols + (j-1)]
                            subplot_filtered_df = plot_df[plot_df[second_param] == subplot_value]
                            subplot_grouped_df = calculate_grouped_bar_data(
                                subplot_filtered_df, x_axis, selected_models, selected_metric, 
                                fit_target, n_ranking_samples, ranking_seed, task_name
                            )
                            subplot_unique_x_vals = sorted(subplot_grouped_df['x_value'].unique()) if not subplot_grouped_df.empty else []
                            
                            bar_x_min = -0.5
                            bar_x_max = len(subplot_unique_x_vals) - 0.5 if subplot_unique_x_vals else 0.5
                            
                            fig.update_xaxes(
                                title=dict(text=x_axis.replace('_', ' ').replace('-', ' ').title(), font=dict(size=12)),
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='lightgray',
                                zeroline=True,
                                zerolinewidth=1,
                                zerolinecolor='black',
                                range=[bar_x_min, bar_x_max],  # Use bar-specific range
                                row=i, col=j
                            )
                            fig.update_yaxes(
                                title=dict(text=y_axis_title, font=dict(size=12)),
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='lightgray',
                                zeroline=True,
                                zerolinewidth=1,
                                zerolinecolor='black',
                                range=None,  # Let plotly auto-scale for scatter/line fit
                                row=i, col=j
                            )
                    else:
                        # Original scatter/line fit logic for subplots
                        
                        # Define a scientific color palette
                        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                        
                        # Calculate global y-axis range across all subplots
                        y_range = None  # Default to auto-scale
                        if fit_target == "Ranking":
                            # For ranking analysis, we need to calculate ranking data first
                            all_ranking_data = []
                            for value in sorted(unique_values):
                                filtered_df = plot_df[plot_df[second_param] == value]
                                ranking_df = calculate_ranking_with_uncertainty(
                                    filtered_df, selected_models, selected_metric, x_axis, second_param,
                                    n_ranking_samples, ranking_seed, task_name
                                )
                                if not ranking_df.empty:
                                    all_ranking_data.extend(ranking_df['mean_rank'].tolist())
                            
                            if all_ranking_data:
                                y_min, y_max = min(all_ranking_data), max(all_ranking_data)
                                y_padding = (y_max - y_min) * 0.05
                                y_range = [y_max + y_padding, y_min - y_padding]  # Inverted for ranking
                            else:
                                y_range = [len(selected_models) + 0.5, 0.5]  # Default ranking range
                        else:
                            all_y_values = []
                            for value in sorted(unique_values):
                                filtered_df = plot_df[plot_df[second_param] == value]
                                for model in selected_models:
                                    model_metric_col = f"model.{task_name}.{model}.{selected_metric}.mean"
                                    if model_metric_col in filtered_df.columns:
                                        model_data = filtered_df.dropna(subset=[model_metric_col])
                                        if len(model_data) > 0:
                                            all_y_values.extend(model_data[model_metric_col].tolist())
                            
                            # Set global ranges with some padding
                            if all_y_values:
                                y_min, y_max = min(all_y_values), max(all_y_values)
                                y_padding = (y_max - y_min) * 0.05
                                y_range = [y_min - y_padding, y_max + y_padding]
                            else:
                                y_range = None  # Let plotly auto-scale
                        
                        # Calculate global x-axis range
                        x_range = None  # Default to auto-scale
                        all_x_values = plot_df[x_axis].dropna().tolist()
                        if all_x_values:
                            x_min, x_max = min(all_x_values), max(all_x_values)
                            x_padding = (x_max - x_min) * 0.05
                            x_range = [x_min - x_padding, x_max + x_padding]
                        else:
                            x_range = None  # Let plotly auto-scale
                        
                        for i, value in enumerate(sorted(unique_values)):
                            # Calculate subplot position
                            row = (i // n_cols) + 1
                            col = (i % n_cols) + 1
                            
                            # Filter data for this value
                            filtered_df = plot_df[plot_df[second_param] == value]
                            
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
                                        
                                        # Handle plotting based on fit_target
                                        if fit_target == "Ranking":
                                            # Only plot ranking data, not original metrics
                                            ranking_df = calculate_ranking_with_uncertainty(
                                                filtered_df, selected_models, selected_metric, x_axis, second_param,
                                                n_ranking_samples, ranking_seed, task_name
                                            )
                                            
                                            if not ranking_df.empty:
                                                # Filter ranking data for this subplot value
                                                subplot_ranking = ranking_df[ranking_df['second_param'] == value]
                                                
                                                for j, model in enumerate(selected_models):
                                                    model_ranking = subplot_ranking[subplot_ranking['model'] == model]
                                                    
                                                    if not model_ranking.empty:
                                                        color = colors[j % len(colors)]
                                                        
                                                        # Plot ranking data points
                                                        fig.add_trace(go.Scatter(
                                                            x=model_ranking['x_axis'],
                                                            y=model_ranking['mean_rank'],
                                                            mode='markers',
                                                            name=f"{model} (Rank)",
                                                            marker=dict(
                                                                color=color,
                                                                size=8,
                                                                line=dict(width=1, color='white')
                                                            ),
                                                            error_y=dict(
                                                                type='data', 
                                                                array=model_ranking['std_rank'], 
                                                                visible=True,
                                                                thickness=1,
                                                                width=3
                                                            ),
                                                            hovertemplate=f'<b>{model}</b><br>' +
                                                                          f'{second_param}: {value}<br>' +
                                                                          f'{x_axis}: %{{x}}<br>' +
                                                                          f'Mean Rank: %{{y}}<br>' +
                                                                          f'Rank Std: %{{error_y.array}}<br>' +
                                                                          '<extra></extra>',
                                                            showlegend=(i == 0)  # Only show legend for first subplot
                                                        ), row=row, col=col)
                                                        
                                                        # Add fitted line for ranking analysis only if Line Fit is selected
                                                        if plot_style == 'Line Fit' and len(model_ranking) >= 3:
                                                            # Sort data by x-axis for proper line fitting
                                                            sorted_ranking = model_ranking.sort_values('x_axis')
                                                            x_fit = sorted_ranking['x_axis'].values
                                                            y_fit = sorted_ranking['mean_rank'].values
                                                            y_std_fit = sorted_ranking['std_rank'].values
                                                            
                                                            # Fit line with uncertainty propagation
                                                            fit_result = propagate_uncertainty_linear_regression(x_fit, y_fit, y_std_fit)
                                                            
                                                            if fit_result is not None:
                                                                # Add fitted line
                                                                fig.add_trace(go.Scatter(
                                                                    x=fit_result['x_pred'],
                                                                    y=fit_result['mean_line'],
                                                                    mode='lines',
                                                                    name=f"{model} (Rank Fitted)",
                                                                    line=dict(color=color, width=3, dash='dash'),
                                                                    hovertemplate=f'<b>{model} (Rank Fitted)</b><br>' +
                                                                                  f'{second_param}: {value}<br>' +
                                                                                  f'{x_axis}: %{{x}}<br>' +
                                                                                  f'Mean Rank: %{{y}}<br>' +
                                                                                  '<extra></extra>',
                                                                    showlegend=(i == 0)  # Only show legend for first subplot
                                                                ), row=row, col=col)
                                                                
                                                                # Add confidence bands
                                                                # Convert hex color to rgba for transparency
                                                                if color.startswith('#') and len(color) == 7:
                                                                    r = int(color[1:3], 16)
                                                                    g = int(color[3:5], 16)
                                                                    b = int(color[5:7], 16)
                                                                    rgba_color = f'rgba({r}, {g}, {b}, 0.2)'
                                                                else:
                                                                    rgba_color = f'rgba(0, 0, 0, 0.2)'
                                                                
                                                                if fit_result is not None:
                                                                    fig.add_trace(go.Scatter(
                                                                        x=np.concatenate([fit_result['x_pred'], fit_result['x_pred'][::-1]]),
                                                                        y=np.concatenate([fit_result['pred_upper'], fit_result['pred_lower'][::-1]]),
                                                                        fill='toself',
                                                                        fillcolor=rgba_color,
                                                                        line=dict(color='rgba(0,0,0,0)'),
                                                                        name=f"{model} (Rank 95% CI)",
                                                                        showlegend=False,
                                                                        hoverinfo='skip'
                                                                    ), row=row, col=col)
                                        else:
                                            # Plot original metrics (not ranking)
                                            if plot_style == 'Scatter':
                                                # Only plot data points, no fitted lines
                                                fig.add_trace(go.Scatter(
                                                    x=model_data[x_axis],
                                                    y=model_data[model_metric_col],
                                                    mode='markers',
                                                    name=model,
                                                    marker=dict(
                                                        color=color,
                                                        size=8,
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
                                                                  f'{second_param}: {value}<br>' +
                                                                  f'{x_axis}: %{{x}}<br>' +
                                                                  f'{selected_metric}: %{{y}}<br>' +
                                                                  '<extra></extra>',
                                                    showlegend=(i == 0)  # Only show legend for first subplot
                                                ), row=row, col=col)
                                            elif plot_style == 'Line Fit':
                                                # Show data points if not showing only lines
                                                if not show_only_lines:
                                                    fig.add_trace(go.Scatter(
                                                        x=model_data[x_axis],
                                                        y=model_data[model_metric_col],
                                                        mode='markers',
                                                        name=model,
                                                        marker=dict(
                                                            color=color,
                                                            size=6,
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
                                                                      f'{second_param}: {value}<br>' +
                                                                      f'{x_axis}: %{{x}}<br>' +
                                                                      f'{selected_metric}: %{{y}}<br>' +
                                                                      '<extra></extra>',
                                                        showlegend=(i == 0)  # Only show legend for first subplot
                                                    ), row=row, col=col)
                                                
                                                # Add fitted line with uncertainty bands only for Line Fit
                                                if len(model_data) >= 3:
                                                    # Sort data by x-axis for proper line fitting
                                                    sorted_data = model_data.sort_values(x_axis)
                                                    x_fit = sorted_data[x_axis].values
                                                    y_fit = sorted_data[model_metric_col].values
                                                    y_std_fit = sorted_data[model_error_col].values / 1.96 if model_error_col in sorted_data.columns else np.ones_like(y_fit) * 0.01
                                                    
                                                    # Choose fitting method based on fit_type
                                                    if fit_type == "Linear":
                                                        fit_result = propagate_uncertainty_linear_regression(x_fit, y_fit, y_std_fit)
                                                    else:  # Polynomial
                                                        fit_result = propagate_uncertainty_polynomial_regression(x_fit, y_fit, y_std_fit, poly_degree)
                                                    
                                                    if fit_result is not None:
                                                        # Add fitted line
                                                        fig.add_trace(go.Scatter(
                                                            x=fit_result['x_pred'],
                                                            y=fit_result['mean_line'],
                                                            mode='lines',
                                                            name=f"{model} (Fitted)",
                                                            line=dict(color=color, width=3, dash='dash'),
                                                            hovertemplate=f'<b>{model} (Fitted)</b><br>' +
                                                                          f'{second_param}: {value}<br>' +
                                                                          f'{x_axis}: %{{x}}<br>' +
                                                                          f'{selected_metric}: %{{y}}<br>' +
                                                                          '<extra></extra>',
                                                            showlegend=(i == 0)  # Only show legend for first subplot
                                                        ), row=row, col=col)
                                                        
                                                        # Add confidence bands
                                                        # Convert hex color to rgba for transparency
                                                        if color.startswith('#') and len(color) == 7:
                                                            r = int(color[1:3], 16)
                                                            g = int(color[3:5], 16)
                                                            b = int(color[5:7], 16)
                                                            rgba_color = f'rgba({r}, {g}, {b}, 0.2)'
                                                        else:
                                                            rgba_color = f'rgba(0, 0, 0, 0.2)'
                                                        
                                                        if fit_result is not None:
                                                            fig.add_trace(go.Scatter(
                                                                x=np.concatenate([fit_result['x_pred'], fit_result['x_pred'][::-1]]),
                                                                y=np.concatenate([fit_result['pred_upper'], fit_result['pred_lower'][::-1]]),
                                                                fill='toself',
                                                                fillcolor=rgba_color,
                                                                line=dict(color='rgba(0,0,0,0)'),
                                                                name=f"{model} (95% CI)",
                                                                showlegend=False,
                                                                hoverinfo='skip'
                                                            ), row=row, col=col)

                    # Update all subplot axes with shared ranges
                    for i in range(1, n_rows + 1):
                        for j in range(1, n_cols + 1):
                            fig.update_xaxes(
                                title=dict(text=x_axis.replace('_', ' ').replace('-', ' ').title(), font=dict(size=12)),
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='lightgray',
                                zeroline=True,
                                zerolinewidth=1,
                                zerolinecolor='black',
                                range=x_range if x_range is not None else None,
                                row=i, col=j
                            )
                            y_axis_title = "Model Rank" if fit_target == 'Ranking' else selected_metric.replace('_', ' ').replace('-', ' ').title()
                            fig.update_yaxes(
                                title=dict(text=y_axis_title, font=dict(size=12)),
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='lightgray',
                                zeroline=True,
                                zerolinewidth=1,
                                zerolinecolor='black',
                                range=y_range if y_range is not None else None,  # Use y_range from subplot section
                                row=i, col=j
                            )
                    
                    # Apply R-style scientific formatting to all subplots
                    if fit_target == "Ranking":
                        plot_title = f"Model Ranking vs {x_axis.replace('_', ' ').replace('-', ' ').title()} by {second_param.replace('_', ' ').replace('-', ' ').title()}"
                    elif plot_style == 'Line Fit':
                        plot_title = f"{selected_metric.replace('_', ' ').replace('-', ' ').title()} vs {x_axis.replace('_', ' ').replace('-', ' ').title()} by {second_param.replace('_', ' ').replace('-', ' ').title()} (with Fitted Lines)"
                    else:
                        plot_title = f"{selected_metric.replace('_', ' ').replace('-', ' ').title()} vs {x_axis.replace('_', ' ').replace('-', ' ').title()} by {second_param.replace('_', ' ').replace('-', ' ').title()}"
                    fig.update_layout(
                        title=dict(
                            text=plot_title,
                            x=0.5,
                            xanchor='center',
                            font=dict(size=16, color='black')
                        ),
                        legend=dict(
                            x=1.02,  # Move legend outside to the right
                            y=0.5,
                            xanchor='left',
                            yanchor='middle',
                            bgcolor='rgba(255,255,255,0.9)',
                            bordercolor='black',
                            borderwidth=1,
                            font=dict(size=11)
                        ),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=300 * n_rows,  # Adjust height based on number of rows
                        margin=dict(l=60, r=150, t=80, b=60)  # Increase right margin for legend
                    )
                    
                else:
                    # Use dot size for many values
                    fig = go.Figure()
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                    
                    # Handle Grouped Bar plots for many values case
                    if plot_style == "Grouped Bar":
                        # Calculate grouped bar data
                        grouped_df = calculate_grouped_bar_data(
                            plot_df, x_axis, selected_models, selected_metric, 
                            fit_target, n_ranking_samples, ranking_seed, task_name
                        )
                        
                        if not grouped_df.empty:
                            # Create grouped bar chart
                            unique_x_vals = grouped_df['x_value'].unique()
                            n_models = len(selected_models)
                            
                            # Calculate bar positions
                            bar_width = 0.8 / n_models
                            x_positions = np.arange(len(unique_x_vals))
                            
                            for j, model in enumerate(selected_models):
                                model_data = grouped_df[grouped_df['model'] == model]
                                
                                if not model_data.empty:
                                    color = colors[j % len(colors)]
                                    
                                    # Calculate bar positions for this model
                                    bar_positions = x_positions + (j - (n_models-1)/2) * bar_width
                                    
                                    # Add bar trace
                                    fig.add_trace(go.Bar(
                                        x=bar_positions,
                                        y=model_data['mean_value'],
                                        name=f"{model}",
                                        marker_color=color,
                                        error_y=dict(
                                            type='data',
                                            array=model_data['std_value'],
                                            visible=True,
                                            thickness=1,
                                            width=3
                                        ),
                                        hovertemplate=f'<b>{model}</b><br>' +
                                                      f'{x_axis}: %{{x}}<br>' +
                                                      f'Mean: %{{y}}<br>' +
                                                      f'Std: %{{error_y.array}}<br>' +
                                                      '<extra></extra>'
                                    ))
                            
                            # Update x-axis labels
                            fig.update_xaxes(
                                ticktext=[str(x) for x in unique_x_vals],
                                tickvals=x_positions
                            )
                            
                            # Update layout for grouped bar
                            y_axis_title = "Model Rank" if fit_target == 'Ranking' else selected_metric.replace('_', ' ').replace('-', ' ').title()
                            
                            fig.update_layout(
                                title=dict(
                                    text=f"Grouped Bar: {selected_metric.replace('_', ' ').replace('-', ' ').title()} vs {x_axis.replace('_', ' ').replace('-', ' ').title()} (dot size = {second_param.replace('_', ' ').replace('-', ' ').title()})",
                                    x=0.5,
                                    xanchor='center',
                                    font=dict(size=16, color='black')
                                ),
                                xaxis=dict(
                                    title=dict(text=x_axis.replace('_', ' ').replace('-', ' ').title(), font=dict(size=14)),
                                    showgrid=True,
                                    gridwidth=1,
                                    gridcolor='lightgray',
                                    zeroline=True,
                                    zerolinewidth=1,
                                    zerolinecolor='black'
                                ),
                                yaxis=dict(
                                    title=dict(text=y_axis_title, font=dict(size=14)),
                                    showgrid=True,
                                    gridwidth=1,
                                    gridcolor='lightgray',
                                    zeroline=True,
                                    zerolinewidth=1,
                                    zerolinecolor='black'
                                ),
                                legend=dict(
                                    x=1.02,
                                    y=0.5,
                                    xanchor='left',
                                    yanchor='middle',
                                    bgcolor='rgba(255,255,255,0.9)',
                                    bordercolor='black',
                                    borderwidth=1,
                                    font=dict(size=11)
                                ),
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                height=600,
                                margin=dict(l=60, r=150, t=80, b=60)
                            )
                        else:
                            st.warning("No data available for grouped bar plot.")
                    else:
                        # Original scatter/line fit logic for many values case
                        
                        # For ranking analysis, we need to calculate the y-axis range
                        if fit_target == "Ranking":
                            ranking_df = calculate_ranking_with_uncertainty(
                                plot_df, selected_models, selected_metric, x_axis, second_param,
                                n_ranking_samples, ranking_seed, task_name
                            )
                            if not ranking_df.empty:
                                y_min, y_max = ranking_df['mean_rank'].min(), ranking_df['mean_rank'].max()
                                y_padding = (y_max - y_min) * 0.05
                                y_range = [y_max + y_padding, y_min - y_padding]  # Inverted for ranking
                            else:
                                y_range = [len(selected_models) + 0.5, 0.5]  # Default ranking range
                        else:
                            y_range = None  # Let plotly auto-scale
                        
                        for i, model in enumerate(selected_models):
                            model_metric_col = f"model.community.{model}.{selected_metric}.mean"
                            model_error_col = f"model.community.{model}.{selected_metric}.ci95"
                            
                            if model_metric_col in plot_df.columns:
                                model_data = plot_df.dropna(subset=[model_metric_col])
                                
                                if len(model_data) > 0:
                                    error_y_data = None
                                    if model_error_col in model_data.columns:
                                        error_y_data = model_data[model_error_col]
                                    
                                    color = colors[i % len(colors)]
                                    
                                    # Calculate dot sizes based on second parameter
                                    min_size = 5
                                    max_size = 15
                                    if second_param in model_data.columns:
                                        sizes = min_size + (max_size - min_size) * (model_data[second_param] - model_data[second_param].min()) / (model_data[second_param].max() - model_data[second_param].min())
                                    else:
                                        sizes = 8
                                    
                                    if fit_target != "Ranking":  # Only plot original metric if not ranking
                                        if plot_style == 'Scatter':
                                            fig.add_trace(go.Scatter(
                                                x=model_data[x_axis],
                                                y=model_data[model_metric_col],
                                                mode='markers',
                                                name=model,
                                                marker=dict(
                                                    color=color,
                                                    size=sizes,
                                                    line=dict(width=1, color='white'),
                                                    showscale=True,
                                                    colorscale='Viridis'
                                                ),
                                                error_y=dict(
                                                    type='data', 
                                                    array=error_y_data, 
                                                    visible=True,
                                                    thickness=1,
                                                    width=3
                                                ) if error_y_data is not None else None,
                                                hovertemplate=f'<b>{model}</b><br>' +
                                                              f'{second_param}: %{{marker.size}}<br>' +
                                                              f'{x_axis}: %{{x}}<br>' +
                                                              f'{selected_metric}: %{{y}}<br>' +
                                                              '<extra></extra>'
                                            ))
                                        elif plot_style == 'Line Fit':
                                            # Only show data points if not showing only lines
                                            if not show_only_lines:
                                                fig.add_trace(go.Scatter(
                                                    x=model_data[x_axis],
                                                    y=model_data[model_metric_col],
                                                    mode='markers',  # Changed from 'lines+markers' to just 'markers'
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
                                                                  f'{second_param}: %{{marker.size}}<br>' +
                                                                  f'{x_axis}: %{{x}}<br>' +
                                                                  f'{selected_metric}: %{{y}}<br>' +
                                                                  '<extra></extra>'
                                                ))
                                            
                                            # Add fitted line with uncertainty bands
                                            if len(model_data) >= 3:  # Need at least 3 points for fitting
                                                # Sort data by x-axis for proper line fitting
                                                sorted_data = model_data.sort_values(x_axis)
                                                x_fit = sorted_data[x_axis].values
                                                y_fit = sorted_data[model_metric_col].values
                                                y_std_fit = sorted_data[model_error_col].values / 1.96 if model_error_col in sorted_data.columns else np.ones_like(y_fit) * 0.01
                                                
                                                # Choose fitting method based on fit_type
                                                if fit_type == "Linear":
                                                    fit_result = propagate_uncertainty_linear_regression(x_fit, y_fit, y_std_fit)
                                                else:  # Polynomial
                                                    fit_result = propagate_uncertainty_polynomial_regression(x_fit, y_fit, y_std_fit, poly_degree)
                                                
                                                if fit_result is not None:
                                                    # Add fitted line
                                                    fig.add_trace(go.Scatter(
                                                        x=fit_result['x_pred'],
                                                        y=fit_result['mean_line'],
                                                        mode='lines',
                                                        name=f"{model} (Fitted)",
                                                        line=dict(color=color, width=3, dash='dash'),
                                                        hovertemplate=f'<b>{model} (Fitted)</b><br>' +
                                                                      f'{second_param}: %{{marker.size}}<br>' +
                                                                      f'{x_axis}: %{{x}}<br>' +
                                                                      f'{selected_metric}: %{{y}}<br>' +
                                                                      '<extra></extra>'
                                                    ))
                                                    
                                                    # Add confidence bands
                                                    # Convert hex color to rgba for transparency
                                                    if color.startswith('#'):
                                                        r = int(color[1:3], 16)
                                                        g = int(color[3:5], 16)
                                                        b = int(color[5:7], 16)
                                                        rgba_color = f'rgba({r}, {g}, {b}, 0.2)'
                                                    else:
                                                        rgba_color = f'rgba(0, 0, 0, 0.2)'
                                                    
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
                                    elif fit_target == "Ranking":
                                        # Calculate ranking data
                                        ranking_df = calculate_ranking_with_uncertainty(
                                            plot_df, selected_models, selected_metric, x_axis, second_param,
                                            n_ranking_samples, ranking_seed, task_name
                                        )
                                        
                                        if not ranking_df.empty:
                                            for j, model in enumerate(selected_models):
                                                model_ranking = ranking_df[ranking_df['model'] == model]
                                                
                                                if not model_ranking.empty:
                                                    color = colors[j % len(colors)]
                                                    
                                                    fig.add_trace(go.Scatter(
                                                        x=model_ranking['x_axis'],
                                                        y=model_ranking['mean_rank'],
                                                        mode='markers',
                                                        name=f"{model} (Rank)",
                                                        marker=dict(
                                                            color=color,
                                                            size=sizes,
                                                            line=dict(width=1, color='white')
                                                        ),
                                                        error_y=dict(
                                                            type='data', 
                                                            array=model_ranking['std_rank'], 
                                                            visible=True,
                                                            thickness=1,
                                                            width=3
                                                        ),
                                                        hovertemplate=f'<b>{model}</b><br>' +
                                                                      f'{second_param}: %{{marker.size}}<br>' +
                                                                      f'{x_axis}: %{{x}}<br>' +
                                                                      f'Mean Rank: %{{y}}<br>' +
                                                                      f'Rank Std: %{{error_y.array}}<br>' +
                                                                      '<extra></extra>'
                                                    ))
                                                    
                                                    # Add fitted line for ranking analysis
                                                    if len(model_ranking) >= 3:  # Need at least 3 points for fitting
                                                        # Sort data by x-axis for proper line fitting
                                                        sorted_ranking = model_ranking.sort_values('x_axis')
                                                        x_fit = sorted_ranking['x_axis'].values
                                                        y_fit = sorted_ranking['mean_rank'].values
                                                        y_std_fit = sorted_ranking['std_rank'].values
                                                        
                                                        # Fit line with uncertainty propagation
                                                        fit_result = propagate_uncertainty_linear_regression(x_fit, y_fit, y_std_fit)
                                                        
                                                        if fit_result is not None:
                                                            # Add fitted line
                                                            fig.add_trace(go.Scatter(
                                                                x=fit_result['x_pred'],
                                                                y=fit_result['mean_line'],
                                                                mode='lines',
                                                                name=f"{model} (Rank Fitted)",
                                                                line=dict(color=color, width=3, dash='dash'),
                                                                hovertemplate=f'<b>{model} (Rank Fitted)</b><br>' +
                                                                              f'{second_param}: %{{marker.size}}<br>' +
                                                                              f'{x_axis}: %{{x}}<br>' +
                                                                              f'Mean Rank: %{{y}}<br>' +
                                                                              '<extra></extra>'
                                                            ))
                                                            
                                                            # Add confidence bands
                                                            if color.startswith('#') and len(color) == 7:
                                                                r = int(color[1:3], 16)
                                                                g = int(color[3:5], 16)
                                                                b = int(color[5:7], 16)
                                                                rgba_color = f'rgba({r}, {g}, {b}, 0.2)'
                                                            else:
                                                                rgba_color = 'rgba(0,0,0,0.2)'
                                                            
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
                    
                    # Apply R-style scientific formatting
                    if fit_target == "Ranking":
                        plot_title = f"Model Ranking vs {x_axis.replace('_', ' ').replace('-', ' ').title()} by {second_param.replace('_', ' ').replace('-', ' ').title()}"
                    elif plot_style == 'Line Fit':
                        plot_title = f"{selected_metric.replace('_', ' ').replace('-', ' ').title()} vs {x_axis.replace('_', ' ').replace('-', ' ').title()} (dot size = {second_param.replace('_', ' ').replace('-', ' ').title()}) (with Fitted Lines)"
                    else:
                        plot_title = f"{selected_metric.replace('_', ' ').replace('-', ' ').title()} vs {x_axis.replace('_', ' ').replace('-', ' ').title()} (dot size = {second_param.replace('_', ' ').replace('-', ' ').title()})"
                    y_axis_title = "Model Rank" if fit_target == 'Ranking' else selected_metric.replace('_', ' ').replace('-', ' ').title()
                    
                    fig.update_layout(
                        title=dict(
                            text=plot_title,
                            x=0.5,
                            xanchor='center',
                            font=dict(size=16, color='black')
                        ),
                        xaxis=dict(
                            title=dict(text=x_axis.replace('_', ' ').replace('-', ' ').title(), font=dict(size=14)),
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='lightgray',
                            zeroline=True,
                            zerolinewidth=1,
                            zerolinecolor='black'
                        ),
                        yaxis=dict(
                            title=dict(text=y_axis_title, font=dict(size=14)),
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='lightgray',
                            zeroline=True,
                            zerolinewidth=1,
                            zerolinecolor='black',
                            range=y_range if fit_target == 'Ranking' else None
                        ),
                        legend=dict(
                            x=1.02,  # Move legend outside to the right
                            y=0.5,
                            xanchor='left',
                            yanchor='middle',
                            bgcolor='rgba(255,255,255,0.9)',
                            bordercolor='black',
                            borderwidth=1,
                            font=dict(size=11)
                        ),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=600,
                        margin=dict(l=60, r=150, t=80, b=60)  # Increase right margin for legend
                    )
            else:
                # Original single plot without second parameter
                fig = go.Figure()
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                
                # Handle Grouped Bar plots for single plot
                if plot_style == "Grouped Bar":
                    # Calculate grouped bar data
                    grouped_df = calculate_grouped_bar_data(
                        plot_df, x_axis, selected_models, selected_metric, 
                        fit_target, n_ranking_samples, ranking_seed, task_name
                    )
                    
                    if not grouped_df.empty:
                        # Create grouped bar chart
                        unique_x_vals = grouped_df['x_value'].unique()
                        n_models = len(selected_models)
                        
                        # Calculate bar positions
                        bar_width = 0.8 / n_models
                        x_positions = np.arange(len(unique_x_vals))
                        
                        for j, model in enumerate(selected_models):
                            model_data = grouped_df[grouped_df['model'] == model]
                            
                            if not model_data.empty:
                                color = colors[j % len(colors)]
                                
                                # Calculate bar positions for this model
                                bar_positions = x_positions + (j - (n_models-1)/2) * bar_width
                                
                                # Add bar trace
                                fig.add_trace(go.Bar(
                                    x=bar_positions,
                                    y=model_data['mean_value'],
                                    name=f"{model}",
                                    marker_color=color,
                                    error_y=dict(
                                        type='data',
                                        array=model_data['std_value'],
                                        visible=True,
                                        thickness=1,
                                        width=3
                                    ),
                                    hovertemplate=f'<b>{model}</b><br>' +
                                                  f'{x_axis}: %{{x}}<br>' +
                                                  f'Mean: %{{y}}<br>' +
                                                  f'Std: %{{error_y.array}}<br>' +
                                                  '<extra></extra>'
                                ))
                        
                        # Update x-axis labels
                        fig.update_xaxes(
                            ticktext=[str(x) for x in unique_x_vals],
                            tickvals=x_positions
                        )
                        
                        # Update layout for grouped bar
                        y_axis_title = "Model Rank" if fit_target == 'Ranking' else selected_metric.replace('_', ' ').replace('-', ' ').title()
                        
                        fig.update_layout(
                            title=dict(
                                text=f"Grouped Bar: {selected_metric.replace('_', ' ').replace('-', ' ').title()} vs {x_axis.replace('_', ' ').replace('-', ' ').title()}",
                                x=0.5,
                                xanchor='center',
                                font=dict(size=16, color='black')
                            ),
                            xaxis=dict(
                                title=dict(text=x_axis.replace('_', ' ').replace('-', ' ').title(), font=dict(size=14)),
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='lightgray',
                                zeroline=True,
                                zerolinewidth=1,
                                zerolinecolor='black'
                            ),
                            yaxis=dict(
                                title=dict(text=y_axis_title, font=dict(size=14)),
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='lightgray',
                                zeroline=True,
                                zerolinewidth=1,
                                zerolinecolor='black'
                            ),
                            legend=dict(
                                x=1.02,
                                y=0.5,
                                xanchor='left',
                                yanchor='middle',
                                bgcolor='rgba(255,255,255,0.9)',
                                bordercolor='black',
                                borderwidth=1,
                                font=dict(size=11)
                            ),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            height=600,
                            margin=dict(l=60, r=150, t=80, b=60)
                        )
                else:
                    # Original scatter/line fit logic for single plot
                    
                    # For ranking analysis, we need to calculate the y-axis range
                    if fit_target == "Ranking":
                        ranking_df = calculate_ranking_with_uncertainty(
                            plot_df, selected_models, selected_metric, x_axis, second_param,
                            n_ranking_samples, ranking_seed, task_name
                        )
                        if not ranking_df.empty:
                            y_min, y_max = ranking_df['mean_rank'].min(), ranking_df['mean_rank'].max()
                            y_padding = (y_max - y_min) * 0.05
                            y_range = [y_max + y_padding, y_min - y_padding]  # Inverted for ranking
                        else:
                            y_range = [len(selected_models) + 0.5, 0.5]  # Default ranking range
                    else:
                        y_range = None  # Let plotly auto-scale
                    
                    # Handle plotting based on fit_target
                    if fit_target == "Ranking":
                        # Only plot ranking data, not original metrics
                        ranking_df = calculate_ranking_with_uncertainty(
                            plot_df, selected_models, selected_metric, x_axis, second_param,
                            n_ranking_samples, ranking_seed, task_name
                        )
                        
                        if not ranking_df.empty:
                            for j, model in enumerate(selected_models):
                                model_ranking = ranking_df[ranking_df['model'] == model]
                                
                                if not model_ranking.empty:
                                    color = colors[j % len(colors)]
                                    
                                    # Plot ranking data points
                                    fig.add_trace(go.Scatter(
                                        x=model_ranking['x_axis'],
                                        y=model_ranking['mean_rank'],
                                        mode='markers',
                                        name=f"{model} (Rank)",
                                        marker=dict(
                                            color=color,
                                            size=8,
                                            line=dict(width=1, color='white')
                                        ),
                                        error_y=dict(
                                            type='data', 
                                            array=model_ranking['std_rank'], 
                                            visible=True,
                                            thickness=1,
                                            width=3
                                        ),
                                        hovertemplate=f'<b>{model}</b><br>' +
                                                      f'{x_axis}: %{{x}}<br>' +
                                                      f'Mean Rank: %{{y}}<br>' +
                                                      f'Rank Std: %{{error_y.array}}<br>' +
                                                      '<extra></extra>'
                                    ))
                                    
                                    # Add fitted line for ranking analysis only if Line Fit is selected
                                    if plot_style == 'Line Fit' and len(model_ranking) >= 3:
                                        # Sort data by x-axis for proper line fitting
                                        sorted_ranking = model_ranking.sort_values('x_axis')
                                        x_fit = sorted_ranking['x_axis'].values
                                        y_fit = sorted_ranking['mean_rank'].values
                                        y_std_fit = sorted_ranking['std_rank'].values
                                        
                                        # Fit line with uncertainty propagation
                                        fit_result = propagate_uncertainty_linear_regression(x_fit, y_fit, y_std_fit)
                                        
                                        if fit_result is not None:
                                            # Add fitted line
                                            fig.add_trace(go.Scatter(
                                                x=fit_result['x_pred'],
                                                y=fit_result['mean_line'],
                                                mode='lines',
                                                name=f"{model} (Rank Fitted)",
                                                line=dict(color=color, width=3, dash='dash'),
                                                hovertemplate=f'<b>{model} (Rank Fitted)</b><br>' +
                                                              f'{x_axis}: %{{x}}<br>' +
                                                              f'Mean Rank: %{{y}}<br>' +
                                                              '<extra></extra>'
                                            ))
                                            
                                            # Add confidence bands
                                            # Convert hex color to rgba for transparency
                                            if color.startswith('#') and len(color) == 7:
                                                r = int(color[1:3], 16)
                                                g = int(color[3:5], 16)
                                                b = int(color[5:7], 16)
                                                rgba_color = f'rgba({r}, {g}, {b}, 0.2)'
                                            else:
                                                rgba_color = f'rgba(0, 0, 0, 0.2)'
                                            
                                            if fit_result is not None:
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
                    else:
                        # Plot original metrics (not ranking)
                        for i, model in enumerate(selected_models):
                            model_metric_col = f"model.{task_name}.{model}.{selected_metric}.mean"
                            model_error_col = f"model.{task_name}.{model}.{selected_metric}.ci95"
                            
                            if model_metric_col in plot_df.columns:
                                model_data = plot_df.dropna(subset=[model_metric_col])
                                
                                if len(model_data) > 0:
                                    error_y_data = None
                                    if model_error_col in model_data.columns:
                                        error_y_data = model_data[model_error_col]
                                    
                                    color = colors[i % len(colors)]
                                    
                                    if plot_style == 'Scatter':
                                        # Only plot data points, no fitted lines
                                        fig.add_trace(go.Scatter(
                                            x=model_data[x_axis],
                                            y=model_data[model_metric_col],
                                            mode='markers',
                                            name=model,
                                            marker=dict(
                                                color=color,
                                                size=8,
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
                                    elif plot_style == 'Line Fit':
                                        # Show data points if not showing only lines
                                        if not show_only_lines:
                                            fig.add_trace(go.Scatter(
                                                x=model_data[x_axis],
                                                y=model_data[model_metric_col],
                                                mode='markers',
                                                name=model,
                                                marker=dict(
                                                    color=color,
                                                    size=6,
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
                                        
                                        # Add fitted line with uncertainty bands only for Line Fit
                                        if len(model_data) >= 3:
                                            # Sort data by x-axis for proper line fitting
                                            sorted_data = model_data.sort_values(x_axis)
                                            x_fit = sorted_data[x_axis].values
                                            y_fit = sorted_data[model_metric_col].values
                                            y_std_fit = sorted_data[model_error_col].values / 1.96 if model_error_col in sorted_data.columns else np.ones_like(y_fit) * 0.01
                                            
                                            # Choose fitting method based on fit_type
                                            if fit_type == "Linear":
                                                fit_result = propagate_uncertainty_linear_regression(x_fit, y_fit, y_std_fit)
                                            else:  # Polynomial
                                                fit_result = propagate_uncertainty_polynomial_regression(x_fit, y_fit, y_std_fit, poly_degree)
                                            
                                            if fit_result is not None:
                                                # Add fitted line
                                                fig.add_trace(go.Scatter(
                                                    x=fit_result['x_pred'],
                                                    y=fit_result['mean_line'],
                                                    mode='lines',
                                                    name=f"{model} (Fitted)",
                                                    line=dict(color=color, width=3, dash='dash'),
                                                    hovertemplate=f'<b>{model} (Fitted)</b><br>' +
                                                                  f'{x_axis}: %{{x}}<br>' +
                                                                  f'{selected_metric}: %{{y}}<br>' +
                                                                  '<extra></extra>'
                                                ))
                                                
                                                # Add confidence bands
                                                # Convert hex color to rgba for transparency
                                                if color.startswith('#') and len(color) == 7:
                                                    r = int(color[1:3], 16)
                                                    g = int(color[3:5], 16)
                                                    b = int(color[5:7], 16)
                                                    rgba_color = f'rgba({r}, {g}, {b}, 0.2)'
                                                else:
                                                    rgba_color = f'rgba(0, 0, 0, 0.2)'
                                                
                                                if fit_result is not None:
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


                
                # Apply R-style scientific formatting
                if fit_target == "Ranking":
                    plot_title = f"Model Ranking vs {x_axis.replace('_', ' ').replace('-', ' ').title()}"
                elif plot_style == 'Line Fit':
                    plot_title = f"{selected_metric.replace('_', ' ').replace('-', ' ').title()} vs {x_axis.replace('_', ' ').replace('-', ' ').title()} (with Fitted Lines)"
                else:
                    plot_title = f"{selected_metric.replace('_', ' ').replace('-', ' ').title()} vs {x_axis.replace('_', ' ').replace('-', ' ').title()}"
                y_axis_title = "Model Rank" if fit_target == 'Ranking' else selected_metric.replace('_', ' ').replace('-', ' ').title()
                
                fig.update_layout(
                    title=dict(
                        text=plot_title,
                        x=0.5,
                        xanchor='center',
                        font=dict(size=16, color='black')
                    ),
                    xaxis=dict(
                        title=dict(text=x_axis.replace('_', ' ').replace('-', ' ').title(), font=dict(size=14)),
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray',
                        zeroline=True,
                        zerolinewidth=1,
                        zerolinecolor='black'
                    ),
                    yaxis=dict(
                        title=dict(text=y_axis_title, font=dict(size=14)),
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray',
                        zeroline=True,
                        zerolinewidth=1,
                        zerolinecolor='black',
                        range=y_range if fit_target == 'Ranking' else None
                    ),
                    legend=dict(
                        x=1.02,  # Move legend outside to the right
                        y=0.5,
                        xanchor='left',
                        yanchor='middle',
                        bgcolor='rgba(255,255,255,0.9)',
                        bordercolor='black',
                        borderwidth=1,
                        font=dict(size=11)
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    height=600,
                    margin=dict(l=60, r=150, t=80, b=60)  # Increase right margin for legend
                )
            
            if fig:
                if not isinstance(fig, go.Figure):
                    fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True, key=f"plot_{x_axis}_{selected_metric}_{fit_target}_{plot_style}")
        

        

    
    # Data table
    st.header("📋 Data Table")
    
    # Column selection for table
    columns_to_show = st.multiselect(
        "Select columns to display:",
        options=df.columns.tolist(),
        default=['run_id', 'method'] + [col for col in df.columns if col.startswith('param_')][:5]
    )
    
    if columns_to_show:
        st.dataframe(df[columns_to_show], use_container_width=True)
    
    # Export functionality
    st.header("💾 Export Data")
    
    if st.button("📥 Download as CSV"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="experiment_results.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Please select your experiment directory and click 'Load Experiment Data' to begin analysis.")
    
    # Show example structure
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