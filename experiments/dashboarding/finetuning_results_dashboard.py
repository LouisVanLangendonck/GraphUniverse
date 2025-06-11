import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path
import numpy as np
from scipy.interpolate import griddata

# Set page config
st.set_page_config(page_title="GNN Experiment Results Dashboard", layout="wide")

@st.cache_data
def load_pretraining_info(config):
    """Load pretraining information from metadata.json for a given config"""
    try:
        pretrained_model_dir = config.get('pretrained_model_dir')
        pretrained_model_id = config.get('pretrained_model_id')
        
        if not pretrained_model_dir or not pretrained_model_id:
            return None
            
        metadata_path = os.path.join(pretrained_model_dir, pretrained_model_id, 'metadata.json')
        
        if not os.path.exists(metadata_path):
            return None
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Extract required information
        pretraining_info = {
            'pretraining_graphs_used': metadata.get('enhanced_info', {}).get('pretraining_graphs_used'),
            'final_metrics': metadata.get('final_metrics', {}),
            'hyperopt_results': metadata.get('hyperopt_results', {}),
            'best_params': metadata.get('hyperopt_results', {}).get('best_params', {}),
            'config': metadata.get('config', {}),
            'pretraining_task': metadata.get('config', {}).get('pretraining_task')
        }
        
        return pretraining_info
    except Exception as e:
        st.warning(f"Error loading pretraining info: {e}")
        return None

@st.cache_data
def load_experiment_data(results_dir):
    """Load and aggregate all experiment results from the directory structure"""
    all_results = []
    
    # Walk through the directory structure
    for exp_dir in Path(results_dir).iterdir():
        if exp_dir.is_dir():
            # Look for experiment directories inside each exp dir
            for sub_dir in exp_dir.iterdir():
                for sub_sub_dir in sub_dir.iterdir():
                    if sub_sub_dir.is_dir():
                        config_file = os.path.join(sub_sub_dir, "config.json")
                        results_file = os.path.join(sub_sub_dir, "results.json")
                        error_file = os.path.join(sub_sub_dir, "error.json")
                        
                        # Skip if error file exists or required files don't exist
                        if os.path.exists(error_file) or not (os.path.exists(config_file) and os.path.exists(results_file)):
                            continue
                        
                        try:
                            # Load config and results
                            with open(config_file, 'r') as f:
                                config = json.load(f)
                            with open(results_file, 'r') as f:
                                results = json.load(f)
                            
                            # Extract experiment info
                            exp_info = {
                                'exp_dir': exp_dir.name,
                                'sub_dir': sub_dir.name,
                                'exp_path': str(sub_dir)
                            }
                            
                            # Extract results parameters from family_consistency
                            family_consistency = results.get('family_consistency', {})
                            results_params = {
                                'pattern_preservation_score': family_consistency.get('pattern_preservation', {}).get('score'),
                                'generation_fidelity_score': family_consistency.get('generation_fidelity', {}).get('score'),
                                'degree_consistency_score': family_consistency.get('degree_consistency', {}).get('score'),
                            }
                            
                            # Extract family properties statistics
                            family_properties = results.get('family_properties', {})
                            family_stats = [
                                'node_counts', 'edge_counts', 'densities', 'avg_degrees', 
                                'clustering_coefficients', 'community_counts', 'homophily_levels'
                            ]
                            for prop in family_stats:
                                for stat in ['mean', 'std', 'min', 'max']:
                                    stat_key = f'{prop}_{stat}'
                                    results_params[stat_key] = family_properties.get(stat_key)
                            
                            # Extract signal statistics
                            community_signals = results.get('community_signals', {})
                            for signal_type in ['degree_signals', 'structure_signals', 'feature_signals']:
                                signal_data = community_signals.get(signal_type, {})
                                for stat in ['mean', 'std', 'min', 'max']:
                                    results_params[f'{signal_type}_{stat}'] = signal_data.get(stat)
                            
                            # Extract model results for each task
                            model_results = results.get('model_results', {})
                            
                            for task_name, task_data in model_results.items():
                                base_model = [model_type for model_type in task_data.keys() if model_type != 'finetuned' and model_type != 'from_scratch']

                                for model_type, model_data in task_data.items():
                                    if isinstance(model_data, dict) and 'test_metrics' in model_data:
                                        # Create a row for this model/task combination
                                        row = {
                                            **exp_info,
                                            **results_params,
                                            'task': task_name,
                                            'model_type': model_type,
                                            'base_model': base_model[0],
                                            **model_data['test_metrics']
                                        }
                                        
                                        # Add training info
                                        row['train_time'] = model_data.get('train_time')
                                        row['best_epoch'] = model_data.get('best_epoch')
                                        
                                        # Add average training silhouette score if available
                                        if 'training_silhouette_scores' in model_data and len(model_data['training_silhouette_scores']) > 0:
                                            row['avg_training_silhouette'] = np.mean(model_data['training_silhouette_scores'])
                                            row['first_silhouette_score'] = model_data['training_silhouette_scores'][0]
                                        else:
                                            row['avg_training_silhouette'] = None
                                            row['first_silhouette_score'] = None
                                        
                                        # Add t-SNE coordinates if available
                                        if 't_sne_training_results' in model_data and 'graph_0' in model_data['t_sne_training_results']:
                                            tsne_data = model_data['t_sne_training_results']['graph_0']
                                            row['tsne_coordinates'] = tsne_data.get('coordinates', {})
                                            row['tsne_silhouette_score'] = tsne_data.get('silhouette_score')
                                        else:
                                            row['tsne_coordinates'] = None
                                            row['tsne_silhouette_score'] = None
                                        
                                        # Try to load pretraining info if available
                                        try:
                                            pretrained_model_dir = config.get('pretrained_model_dir')
                                            pretrained_model_id = config.get('pretrained_model_id')
                                            
                                            if pretrained_model_dir and pretrained_model_id:
                                                metadata_path = os.path.join(pretrained_model_dir, pretrained_model_id, 'metadata.json')
                                                
                                                if os.path.exists(metadata_path):
                                                    with open(metadata_path, 'r') as f:
                                                        metadata = json.load(f)
                                                        
                                                    # Add pretraining information
                                                    row.update({
                                                        'pretraining_graphs_used': metadata.get('enhanced_info', {}).get('pretraining_graphs_used'),
                                                        'pretraining_task': metadata.get('config', {}).get('pretraining_task'),
                                                        'pretraining_loss': metadata.get('final_metrics', {}).get('loss'),
                                                        'pretraining_cosine_similarity': metadata.get('final_metrics', {}).get('cosine_similarity'),
                                                        'pretraining_mse': metadata.get('final_metrics', {}).get('mse'),
                                                        'pretraining_reconstruction_accuracy': metadata.get('final_metrics', {}).get('reconstruction_accuracy'),
                                                        'pretraining_learning_rate': metadata.get('hyperopt_results', {}).get('best_params', {}).get('learning_rate'),
                                                        'pretraining_weight_decay': metadata.get('hyperopt_results', {}).get('best_params', {}).get('weight_decay'),
                                                        'pretraining_hidden_dim': metadata.get('hyperopt_results', {}).get('best_params', {}).get('hidden_dim'),
                                                        'pretraining_num_layers': metadata.get('hyperopt_results', {}).get('best_params', {}).get('num_layers'),
                                                        'pretraining_dropout': metadata.get('hyperopt_results', {}).get('best_params', {}).get('dropout')
                                                    })
                                        except Exception as e:
                                            # Silently continue if pretraining info can't be loaded
                                            pass
                                        
                                        all_results.append(row)
                                        
                        except Exception as e:
                            st.warning(f"Error loading {sub_dir}: {e}")
                            continue
    
    return pd.DataFrame(all_results)

def calculate_differences(df):
    """Calculate differences between finetuned and other models"""
    diff_data = []
    
    # Group by experiment and task to calculate differences
    for (exp_dir, sub_dir, task), group in df.groupby(['exp_dir', 'sub_dir', 'task']):
        # Get the different model types for this experiment/task
        models = {}
        for _, row in group.iterrows():
            models[row['model_type']] = row
        
        # Calculate differences if we have the required models
        if 'finetuned' in models:
            finetuned = models['finetuned']
            
            # Common base info
            base_info = {
                'exp_dir': exp_dir,
                'sub_dir': sub_dir,
                'task': task,
                'exp_path': finetuned['exp_path']
            }
            
            # Copy results parameters from finetuned
            for col in df.columns:
                if col not in ['model_type'] + list(finetuned.index[finetuned.index.str.contains('mse|rmse|mae|r2|accuracy|precision|recall|f1|avg_training_silhouette')]):
                    if col in finetuned.index:
                        base_info[col] = finetuned[col]
            
            # Calculate differences
            for other_model in ['from_scratch'] + [gnn for gnn in models.keys() if gnn not in ['finetuned', 'from_scratch']]:
                if other_model in models:
                    other = models[other_model]
                    
                    # Calculate differences for all test metrics
                    for metric_col in df.columns:
                        if any(metric in metric_col for metric in ['mse', 'rmse', 'mae', 'r2', 'accuracy', 'precision', 'recall', 'f1', 'avg_training_silhouette']):
                            if pd.notna(finetuned.get(metric_col)) and pd.notna(other.get(metric_col)):
                                diff_row = base_info.copy()
                                diff_row['comparison'] = f'finetuned_vs_{other_model}'
                                diff_row['metric_name'] = metric_col
                                diff_row['difference'] = finetuned[metric_col] - other[metric_col]
                                diff_row['finetuned_value'] = finetuned[metric_col]
                                diff_row['other_value'] = other[metric_col]
                                diff_row['other_model'] = other_model
                                diff_data.append(diff_row)
    
    return pd.DataFrame(diff_data)

def create_scatter_plot(df, x_param, y_param, color_by='model_type'):
    """Create scatter plot with proper styling"""
    # Define colors and markers
    colors = px.colors.qualitative.Set1
    markers = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'star']
    
    fig = go.Figure()
    
    # Get unique values for coloring and markers
    unique_models = df[color_by].unique()
    
    for i, model in enumerate(unique_models):
        model_data = df[df[color_by] == model]
        
        # Determine marker based on model type
        if 'finetuned' in model:
            marker = 'circle'
        elif 'from_scratch' in model:
            marker = 'square'  
        else:
            marker = 'diamond'
        
        fig.add_trace(go.Scatter(
            x=model_data[x_param],
            y=model_data[y_param],
            mode='markers',
            name=model,
            marker=dict(
                symbol=marker,
                size=8,
                color=colors[i % len(colors)],
                line=dict(width=1, color='black')
            ),
            hovertemplate=f'<b>{model}</b><br>' +
                         f'{x_param}: %{{x}}<br>' +
                         f'{y_param}: %{{y}}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title=f'{y_param} vs {x_param}',
        xaxis_title=x_param,
        yaxis_title=y_param,
        hovermode='closest',
        showlegend=True,
        height=600
    )
    
    return fig

def create_difference_manifold(df, param1, param2, model, task, metric, show_points=False):
    """Create a 2D manifold plot showing model performance differences across parameter space."""
    # Filter data for this model and task
    model_data = df[
        (df['base_model'] == model) & 
        (df['task'] == task) & 
        (df['metric_name'] == metric) &
        (df['comparison'] == 'finetuned_vs_from_scratch')
    ]
    
    if model_data.empty:
        return None
    
    x_values = model_data[param1].values
    y_values = model_data[param2].values
    z_values = model_data['difference'].values

    # Filter out any rows with nan in x, y, or z
    valid_mask = (~np.isnan(x_values)) & (~np.isnan(y_values)) & (~np.isnan(z_values))
    x_values = x_values[valid_mask]
    y_values = y_values[valid_mask]
    z_values = z_values[valid_mask]

    # Only try interpolation if we have enough unique points
    unique_points = len(set(zip(x_values, y_values)))
    x_range = x_values.max() - x_values.min() if len(x_values) > 0 else float('nan')
    y_range = y_values.max() - y_values.min() if len(y_values) > 0 else float('nan')
    
    fig = go.Figure()
    
    if unique_points >= 4 and x_range > 0 and y_range > 0:
        try:
            # Create a grid for interpolation with adaptive resolution
            x_min, x_max = x_values.min(), x_values.max()
            y_min, y_max = y_values.min(), y_values.max()
            
            # Adaptive padding based on data distribution
            x_padding = max(x_range * 0.2, np.std(x_values) * 0.5)
            y_padding = max(y_range * 0.2, np.std(y_values) * 0.5)
            
            # Adaptive grid resolution based on number of points and data distribution
            n_points = len(x_values)
            base_resolution = 50  # Base resolution for small datasets
            max_resolution = 100  # Maximum resolution for large datasets
            
            # Calculate resolution based on number of points and data distribution
            resolution = min(max_resolution, max(base_resolution, int(np.sqrt(n_points) * 2)))
            
            # Adjust resolution based on model type
            if model.lower() in ['gcn', 'sage', 'gin']:
                resolution = min(resolution, 75)  # Lower resolution for these models
            
            xi = np.linspace(x_min - x_padding, x_max + x_padding, resolution)
            yi = np.linspace(y_min - y_padding, y_max + y_padding, resolution)
            xi_grid, yi_grid = np.meshgrid(xi, yi)
            
            # Adaptive outlier handling based on model type
            z_mean = np.mean(z_values)
            z_std = np.std(z_values)
            
            # More conservative clipping for GCN, SAGE, and GIN
            if model.lower() in ['gcn', 'sage', 'gin']:
                clip_factor = 2.5  # More conservative clipping
            else:
                clip_factor = 3.0  # Standard clipping
            
            z_values_clean = np.clip(z_values, z_mean - clip_factor*z_std, z_mean + clip_factor*z_std)
            
            # Try different interpolation methods with adaptive parameters
            zi = None
            interpolation_methods = ['linear', 'nearest', 'cubic']
            
            # Adjust method order based on model type
            if model.lower() in ['gcn', 'sage', 'gin']:
                interpolation_methods = ['nearest', 'linear', 'cubic']  # Start with simpler methods
            
            for method in interpolation_methods:
                try:
                    # Try with different fill values
                    fill_values = [z_mean, np.nan, 0]
                    if method == 'nearest':
                        fill_values = [z_mean]  # Only use mean for nearest neighbor
                    
                    for fill_value in fill_values:
                        zi = griddata((x_values, y_values), z_values_clean, (xi_grid, yi_grid), 
                                    method=method, fill_value=fill_value)
                        
                        # Check if interpolation produced valid results
                        if not np.all(np.isnan(zi)):
                            # Additional check for model-specific patterns
                            if model.lower() in ['gcn', 'sage', 'gin']:
                                # Check if the interpolation is too smooth (might indicate overfitting)
                                if np.std(zi) < 0.1 * np.std(z_values_clean):
                                    continue
                            break
                    if not np.all(np.isnan(zi)):
                        break
                except:
                    continue
            
            # If interpolation failed, try with a simpler approach
            if zi is None or np.all(np.isnan(zi)):
                # Create a simpler grid with fewer points
                resolution = min(40, resolution)  # Reduce resolution for fallback
                xi = np.linspace(x_min - x_padding, x_max + x_padding, resolution)
                yi = np.linspace(y_min - y_padding, y_max + y_padding, resolution)
                xi_grid, yi_grid = np.meshgrid(xi, yi)
                
                # Try with nearest neighbor interpolation
                zi = griddata((x_values, y_values), z_values_clean, (xi_grid, yi_grid), 
                            method='nearest', fill_value=z_mean)
            
            # Add contour plot if interpolation was successful
            if zi is not None and not np.all(np.isnan(zi)):
                fig.add_trace(go.Contour(
                    x=xi,
                    y=yi,
                    z=zi,
                    colorscale='RdBu_r',  # Red for positive, blue for negative
                    showscale=True,
                    colorbar=dict(
                        title='Difference',
                        titleside='right'
                    ),
                    opacity=0.6,
                    contours=dict(
                        showlabels=True,
                        labelfont=dict(size=8, color='white')
                    ),
                    name='Surface',
                    showlegend=False,
                    zmid=0  # Center the color scale at 0
                ))
        except Exception as e:
            # If interpolation fails, just show scatter plot
            pass
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='markers',
        marker=dict(
            size=8,
            color=z_values,
            colorscale='RdBu_r',  # Red for positive, blue for negative
            line=dict(width=1, color='black'),
            showscale=False,
            cmid=0  # Center the color scale at 0
        ),
        name='Data Points',
        showlegend=False,
        text=[f'{param1}: {x:.3f}<br>{param2}: {y:.3f}<br>Difference: {z:.3f}' 
              for x, y, z in zip(x_values, y_values, z_values)],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"{model.upper()}",
        showlegend=False,
        margin=dict(t=50, b=50, l=50, r=50),
        xaxis_title=param1.replace('_', ' ').title(),
        yaxis_title=param2.replace('_', ' ').title()
    )
    
    return fig

def create_difference_plot(diff_df, x_param, selected_models, selected_metric):
    """Create difference plot with model-dependent colors and comparison-dependent markers"""
    # Generate dynamic colors for all unique models
    unique_models = diff_df['base_model'].unique() if 'base_model' in diff_df.columns else []
    
    # Use a color palette that can handle any number of models
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Dark2
    
    # Create model color mapping
    model_colors = {}
    for i, model in enumerate(unique_models):
        model_colors[model] = colors[i % len(colors)]
    
    # Define markers for comparison types
    comparison_markers = {
        'finetuned_vs_from_scratch': 'circle',
        'finetuned_vs_optimized': 'square'
    }
    
    fig = go.Figure()
    
    # Filter data based on selections
    filtered_df = diff_df[
        (diff_df['base_model'].isin(selected_models)) & 
        (diff_df['metric_name'] == selected_metric) &
        (diff_df['comparison'] == 'finetuned_vs_from_scratch')
    ]
    
    # Group by base_model and comparison for plotting
    for base_model in selected_models:
        plot_data = filtered_df[
            (filtered_df['base_model'] == base_model)
        ]
        
        if not plot_data.empty:
            # Create legend name
            legend_name = f"{base_model} - {selected_metric}"
            
            # For training metrics, get data from the main dataframe
            if x_param in ['avg_training_silhouette', 'first_silhouette_score']:
                x_values = st.session_state.df[
                    (st.session_state.df['base_model'] == base_model) & 
                    (st.session_state.df['task'] == plot_data['task'].iloc[0]) & 
                    (st.session_state.df['model_type'] == 'finetuned')
                ][x_param].values
            else:
                x_values = plot_data[x_param].values
            
            fig.add_trace(go.Scatter(
                x=x_values,
                y=plot_data['difference'],
                mode='markers',
                name=legend_name,
                marker=dict(
                    symbol=comparison_markers.get('finetuned_vs_from_scratch', 'circle'),
                    size=8,
                    color=model_colors.get(base_model, '#000000'),
                    line=dict(width=1, color='black')
                ),
                hovertemplate=f'<b>{legend_name}</b><br>' +
                             f'{x_param}: %{{x}}<br>' +
                             f'Difference: %{{y}}<br>' +
                             f'Metric: {selected_metric}<br>' +
                             '<extra></extra>'
            ))
    
    # Calculate the absolute maximum difference for symmetric y-axis
    max_abs_diff = max(abs(filtered_df['difference'].min()), abs(filtered_df['difference'].max()))
    
    fig.update_layout(
        title=f'{selected_metric} Differences: {x_param}',
        xaxis_title=x_param.replace('_', ' ').title(),
        yaxis_title='Difference',
        hovermode='closest',
        showlegend=True,
        height=600,
        width=None,  # This will make it use container width
        yaxis=dict(
            range=[-max_abs_diff * 1.1, max_abs_diff * 1.1],  # Add 10% padding
            zeroline=True,
            zerolinecolor='gray',
            zerolinewidth=1
        )
    )
    
    # Add horizontal line at y=0 for reference
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    
    return fig

def create_tsne_plot(coordinates, title):
    """Create a t-SNE plot with community coloring"""
    fig = go.Figure()
    
    # Add points for each community
    for community, points in coordinates.items():
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            name=f'Community {community}',
            marker=dict(
                size=8,
                opacity=0.7
            )
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        showlegend=True,
        height=500
    )
    
    return fig

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame()
    if 'diff_df' not in st.session_state:
        st.session_state.diff_df = pd.DataFrame()

# Main Streamlit app
def main():
    st.title("🧠 GNN Experiment Results Dashboard")
    st.markdown("Analyze and visualize graph neural network experiment results")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for data loading
    st.sidebar.header("📁 Data Loading")
    
    # Option to use sample data or load from directory
    use_sample = st.sidebar.checkbox("Use sample data for demonstration", value=True)
    
    # Load data only if not already loaded or if settings changed
    if use_sample and not st.session_state.data_loaded:
        # Create sample data based on the provided JSON structure
        sample_data = []
        
        # Create more diverse sample data
        for i in range(3):
            for model_type in ['finetuned', 'from_scratch', 'gcn', 'sage', 'gat']:
                for task in ['community', 'k_hop_community_counts']:
                    base_data = {
                        'exp_dir': f'exp_{i:03d}', 
                        'sub_dir': f'gcn_link_prediction_20250531_22250{i}',
                        # Family properties statistics
                        'node_counts_mean': 100.0 + i * 10,
                        'node_counts_std': 10.84 + i * 2,
                        'node_counts_min': 80.0 + i * 5,
                        'node_counts_max': 120.0 + i * 10,
                        'edge_counts_mean': 701.37 + i * 100,
                        'edge_counts_std': 302.41 + i * 50,
                        'edge_counts_min': 310.0 + i * 20,
                        'edge_counts_max': 1368.0 + i * 200,
                        'densities_mean': 0.1425 + i * 0.02,
                        'densities_std': 0.0557 + i * 0.01,
                        'densities_min': 0.0508 + i * 0.005,
                        'densities_max': 0.2418 + i * 0.03,
                        'avg_degrees_mean': 13.97 + i * 2,
                        'avg_degrees_std': 5.44 + i * 1,
                        'avg_degrees_min': 5.59 + i * 0.5,
                        'avg_degrees_max': 24.34 + i * 3,
                        'clustering_coefficients_mean': 0.2093 + i * 0.03,
                        'clustering_coefficients_std': 0.0949 + i * 0.01,
                        'clustering_coefficients_min': 0.0501 + i * 0.005,
                        'clustering_coefficients_max': 0.434 + i * 0.05,
                        'community_counts_mean': 4.67 + i * 0.5,
                        'community_counts_std': 1.47 + i * 0.2,
                        'community_counts_min': 3.0 + i * 0.2,
                        'community_counts_max': 7.0 + i * 0.5,
                        # Family consistency scores
                        'pattern_preservation_score': 0.669 + i * 0.1, 
                        'generation_fidelity_score': 0.640 + i * 0.08, 
                        'degree_consistency_score': 0.297 + i * 0.15,
                        # Homophily metrics
                        'homophily_mean': 0.45 + i * 0.05,
                        'homophily_std': 0.12 + i * 0.02,
                        'homophily_min': 0.25 + i * 0.03,
                        'homophily_max': 0.75 + i * 0.05,
                        # Signal statistics
                        'degree_signals_mean': 0.360 + i * 0.05, 
                        'degree_signals_std': 0.100 + i * 0.02, 
                        'degree_signals_min': 0.162 + i * 0.03, 
                        'degree_signals_max': 0.505 + i * 0.08,
                        'structure_signals_mean': 0.284 + i * 0.06, 
                        'structure_signals_std': 0.364 + i * 0.1, 
                        'structure_signals_min': 0.012 + i * 0.005, 
                        'structure_signals_max': 1.612 + i * 0.2,
                        'feature_signals_mean': 0.814 + i * 0.03, 
                        'feature_signals_std': 0.130 + i * 0.02, 
                        'feature_signals_min': 0.605 + i * 0.05, 
                        'feature_signals_max': 0.973 - i * 0.02,
                        'task': task, 
                        'model_type': model_type,
                    }
                    
                    # Add task-specific metrics
                    if task == 'community':
                        base_data.update({
                            'accuracy': 0.400 + np.random.normal(0, 0.05),
                            'f1_macro': 0.342 + np.random.normal(0, 0.04),
                            'precision_macro': 0.425 + np.random.normal(0, 0.03),
                            'recall_macro': 0.388 + np.random.normal(0, 0.04)
                        })
                    else:  # k_hop_community_counts
                        base_data.update({
                            'mse': 185.36 + np.random.normal(0, 20),
                            'rmse': 13.61 + np.random.normal(0, 2),
                            'mae': 10.31 + np.random.normal(0, 1.5),
                            'r2': -0.776 + np.random.normal(0, 0.1)
                        })
                    
                    sample_data.append(base_data)
        
        st.session_state.df = pd.DataFrame(sample_data)
        st.session_state.data_loaded = True
        st.sidebar.success("✅ Sample data loaded successfully!")
        
    elif not use_sample:
        results_dir = st.sidebar.text_input("Results Directory Path", value="./results")
        
        if st.sidebar.button("Load Data") or (not st.session_state.data_loaded):
            if os.path.exists(results_dir):
                with st.spinner("Loading experiment data..."):
                    st.session_state.df = load_experiment_data(results_dir)
                    st.session_state.data_loaded = True
                st.sidebar.success(f"✅ Loaded {len(st.session_state.df)} experiment results!")
            else:
                st.sidebar.error("❌ Directory not found!")
                return
    
    # Check if data is available
    if st.session_state.df.empty:
        st.warning("Please load data to continue.")
        return
    
    # Calculate differences if not already done
    if st.session_state.diff_df.empty and not st.session_state.df.empty:
        st.session_state.diff_df = calculate_differences(st.session_state.df)
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Model Performance", "📈 Difference Analysis", "🗺️ Difference Manifolds", "📋 Data Overview", "🎨 t-SNE Visualization"])
    
    with tab1:
        st.header("Model Performance Analysis")
        
        # Create a single column for the entire width
        st.markdown("**Select Parameters**")
        col1, col2 = st.columns(2)
        
        with col1:
            # X-axis parameter selection - now using family properties instead of config
            family_stats_params = [
                'node_counts_mean', 'node_counts_std', 'node_counts_min', 'node_counts_max',
                'edge_counts_mean', 'edge_counts_std', 'edge_counts_min', 'edge_counts_max',
                'densities_mean', 'densities_std', 'densities_min', 'densities_max',
                'avg_degrees_mean', 'avg_degrees_std', 'avg_degrees_min', 'avg_degrees_max',
                'clustering_coefficients_mean', 'clustering_coefficients_std', 'clustering_coefficients_min', 'clustering_coefficients_max',
                'community_counts_mean', 'community_counts_std', 'community_counts_min', 'community_counts_max', 
                'homophily_levels_mean', 'homophily_levels_std', 'homophily_levels_min', 'homophily_levels_max'
            ]
            
            consistency_params = ['pattern_preservation_score', 'generation_fidelity_score', 'degree_consistency_score']
            
            signal_params = [f'{signal}_{stat}' for signal in ['degree_signals', 'structure_signals', 'feature_signals'] 
                            for stat in ['mean', 'std', 'min', 'max']]
            
            # Add pretraining parameters
            pretraining_params = [
                'pretraining_graphs_used',
                'pretraining_loss',
                'pretraining_cosine_similarity',
                'pretraining_mse',
                'pretraining_reconstruction_accuracy',
                'pretraining_learning_rate',
                'pretraining_weight_decay',
                'pretraining_hidden_dim',
                'pretraining_num_layers',
                'pretraining_dropout'
            ]
            
            # Add homophily parameters
            homophily_params = [
                'homophily_levels_mean',
                'homophily_levels_std',
                'homophily_levels_min',
                'homophily_levels_max'
            ]

            # Add training metrics
            training_metrics = [
                'avg_training_silhouette',
                'first_silhouette_score'
            ]
            
            # Combine all parameters
            all_params = family_stats_params + consistency_params + signal_params + pretraining_params + homophily_params + training_metrics
            
            # Filter parameters that exist in the dataframe
            available_params = [param for param in all_params if param in st.session_state.df.columns]
            
            # Group parameters by category for better organization
            param_groups = {
                'Family Statistics': [p for p in available_params if p in family_stats_params],
                'Consistency Scores': [p for p in available_params if p in consistency_params],
                'Signal Statistics': [p for p in available_params if p in signal_params],
                'Pretraining Metrics': [p for p in available_params if p in pretraining_params],
                'Homophily Metrics': [p for p in available_params if p in homophily_params]
            }
            
            # Create a single selectbox for x-axis parameter selection
            st.markdown("**Select X-axis Parameter**")
            x_param = st.selectbox(
                "Parameter",
                options=available_params,
                format_func=lambda x: x.replace('_', ' ').title(),
                key="x_param_select"
            )
            
            # Display parameter groups for reference
            with st.expander("Available Parameters by Category"):
                for group_name, params in param_groups.items():
                    if params:  # Only show groups that have available parameters
                        st.markdown(f"**{group_name}**")
                        for param in params:
                            st.markdown(f"- {param.replace('_', ' ').title()}")
        
        with col2:
            # Y-axis metric selection
            metric_columns = [col for col in st.session_state.df.columns if any(metric in col for metric in 
                            ['accuracy', 'f1_', 'precision_', 'recall_', 'mse', 'rmse', 'mae', 'r2', 'avg_training_silhouette'])]
            if metric_columns:
                y_param = st.selectbox("Select Y-axis metric:", metric_columns, key="main_y_param")
            else:
                st.warning("No metrics found in the data")
                return
        
        # Task filter
        tasks = st.session_state.df['task'].unique()
        selected_task = st.selectbox("Select task:", tasks, key="main_task")
        
        # Filter data
        plot_df = st.session_state.df[st.session_state.df['task'] == selected_task].copy()
        
        if not plot_df.empty and x_param in plot_df.columns and y_param in plot_df.columns:
            # Create and display plot
            fig = create_scatter_plot(plot_df, x_param, y_param)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display summary statistics
            st.subheader("Summary Statistics")
            summary_cols = ['model_type', x_param, y_param]
            try:
                summary_df = plot_df[summary_cols].groupby('model_type').agg({
                    x_param: ['mean', 'std', 'min', 'max'],
                    y_param: ['mean', 'std', 'min', 'max']
                }).round(4)
                st.dataframe(summary_df)
            except Exception as e:
                st.warning(f"Could not generate summary statistics: {e}")
        else:
            st.warning("No data available for the selected filters or missing columns.")
    
    with tab2:
        st.header("Difference Analysis")
        
        # Create a single column for the entire width
        st.markdown("**Select Parameters**")
        col1, col2 = st.columns(2)
        
        with col1:
            # X-axis parameter selection for difference analysis
            st.markdown("**Select X-axis Parameter**")
            x_param_diff = st.selectbox(
                "Parameter",
                options=available_params,
                format_func=lambda x: x.replace('_', ' ').title(),
                key="diff_x_param_select"
            )
            
            # Display parameter groups for reference
            with st.expander("Available Parameters by Category"):
                for group_name, params in param_groups.items():
                    if params:  # Only show groups that have available parameters
                        st.markdown(f"**{group_name}**")
                        for param in params:
                            st.markdown(f"- {param.replace('_', ' ').title()}")
        
        with col2:
            st.subheader("Plot Parameters")
            available_metrics = st.session_state.diff_df['metric_name'].unique() if 'metric_name' in st.session_state.diff_df.columns else []
            selected_metric = st.selectbox("Select metric:", available_metrics, key="diff_metric")
            
            # Model selection
            available_models = st.session_state.diff_df['base_model'].unique() if 'base_model' in st.session_state.diff_df.columns else []
            selected_models = st.multiselect("Select models to compare:", available_models, default=available_models[:1] if len(available_models) > 0 else [], key="diff_models")
        
        # Create the difference plot in full width
        if selected_models:
            fig_diff = create_difference_plot(st.session_state.diff_df, x_param_diff, selected_models, selected_metric)
            st.plotly_chart(fig_diff, use_container_width=True)
    
    with tab3:
        st.header("Difference Manifolds")
        
        if not st.session_state.diff_df.empty and 'base_model' in st.session_state.diff_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Parameter selection for difference manifold
                st.markdown("**Select Parameters for Manifold**")
                
                # First parameter selection
                st.markdown("**First Parameter**")
                param1 = st.selectbox(
                    "Parameter 1",
                    options=available_params,
                    format_func=lambda x: x.replace('_', ' ').title(),
                    key="manifold_param1_select"
                )
                
                # Second parameter selection
                st.markdown("**Second Parameter**")
                param2 = st.selectbox(
                    "Parameter 2",
                    options=available_params,
                    format_func=lambda x: x.replace('_', ' ').title(),
                    key="manifold_param2_select"
                )
                
                # Display parameter groups for reference
                with st.expander("Available Parameters by Category"):
                    for group_name, params in param_groups.items():
                        if params:  # Only show groups that have available parameters
                            st.markdown(f"**{group_name}**")
                            for param in params:
                                st.markdown(f"- {param.replace('_', ' ').title()}")
            
            with col2:
                st.subheader("Manifold Parameters")
                # Task selection
                available_tasks = st.session_state.diff_df['task'].unique()
                selected_task = st.selectbox("Select task:", available_tasks, key="manifold_task")
                
                # Metric selection
                available_metrics = st.session_state.diff_df['metric_name'].unique()
                selected_metric = st.selectbox("Select metric:", available_metrics, key="manifold_metric")
                
                # Show points option
                show_points = st.checkbox("Show individual points", value=True, key="manifold_show_points")
            
            # Get all available models
            available_models = st.session_state.diff_df['base_model'].unique()
            
            if selected_task and selected_metric:
                # Calculate number of rows and columns for subplots to be as square as possible
                n_models = len(available_models)
                n_cols = int(np.ceil(np.sqrt(n_models)))
                n_rows = int(np.ceil(n_models / n_cols))
                
                # Create subplot figure
                fig = make_subplots(
                    rows=n_rows,
                    cols=n_cols,
                    subplot_titles=[model.upper() for model in available_models],
                    vertical_spacing=0.1,
                    horizontal_spacing=0.1,
                    shared_xaxes=True,  # Share x-axis between subplots
                    shared_yaxes=True   # Share y-axis between subplots
                )
                
                # Calculate global min and max differences for consistent color scale
                all_differences = []
                for model in available_models:
                    model_data = st.session_state.diff_df[
                        (st.session_state.diff_df['base_model'] == model) & 
                        (st.session_state.diff_df['task'] == selected_task) & 
                        (st.session_state.diff_df['metric_name'] == selected_metric) &
                        (st.session_state.diff_df['comparison'] == 'finetuned_vs_from_scratch')
                    ]
                    if not model_data.empty:
                        all_differences.extend(model_data['difference'].values)
                
                max_abs_diff = max(abs(min(all_differences)), abs(max(all_differences)))
                
                # Get pretraining task from the first model's data
                pretraining_task = None
                for model in available_models:
                    model_data = st.session_state.diff_df[
                        (st.session_state.diff_df['base_model'] == model) & 
                        (st.session_state.diff_df['task'] == selected_task) & 
                        (st.session_state.diff_df['metric_name'] == selected_metric)
                    ]
                    if not model_data.empty and 'pretraining_task' in model_data.columns:
                        pretraining_task = model_data['pretraining_task'].iloc[0]
                        break
                
                # Create manifold for each model
                for i, model in enumerate(available_models):
                    row = (i // n_cols) + 1
                    col = (i % n_cols) + 1
                    
                    # Get model data
                    model_data = st.session_state.diff_df[
                        (st.session_state.diff_df['base_model'] == model) & 
                        (st.session_state.diff_df['task'] == selected_task) & 
                        (st.session_state.diff_df['metric_name'] == selected_metric) &
                        (st.session_state.diff_df['comparison'] == 'finetuned_vs_from_scratch')
                    ]
                    
                    if not model_data.empty:
                        # For training metrics, get data from the main dataframe
                        if param1 in ['avg_training_silhouette', 'first_silhouette_score']:
                            x_values = st.session_state.df[
                                (st.session_state.df['base_model'] == model) & 
                                (st.session_state.df['task'] == selected_task) & 
                                (st.session_state.df['model_type'] == 'finetuned')
                            ][param1].values
                        else:
                            x_values = model_data[param1].values

                        if param2 in ['avg_training_silhouette', 'first_silhouette_score']:
                            y_values = st.session_state.df[
                                (st.session_state.df['base_model'] == model) & 
                                (st.session_state.df['task'] == selected_task) & 
                                (st.session_state.df['model_type'] == 'finetuned')
                            ][param2].values
                        else:
                            y_values = model_data[param2].values

                        z_values = model_data['difference'].values

                        # Filter out any rows with nan in x, y, or z
                        valid_mask = (~np.isnan(x_values)) & (~np.isnan(y_values)) & (~np.isnan(z_values))
                        x_values = x_values[valid_mask]
                        y_values = y_values[valid_mask]
                        z_values = z_values[valid_mask]

                        # Only try interpolation if we have enough unique points
                        unique_points = len(set(zip(x_values, y_values)))
                        x_range = x_values.max() - x_values.min() if len(x_values) > 0 else float('nan')
                        y_range = y_values.max() - y_values.min() if len(y_values) > 0 else float('nan')
                        
                        if unique_points >= 4 and x_range > 0 and y_range > 0:
                            try:
                                # Create interpolation grid
                                x_min, x_max = x_values.min(), x_values.max()
                                y_min, y_max = y_values.min(), y_values.max()
                                
                                x_padding = x_range * 0.1
                                y_padding = y_range * 0.1
                                
                                xi = np.linspace(x_min - x_padding, x_max + x_padding, 40)
                                yi = np.linspace(y_min - y_padding, y_max + y_padding, 40)
                                xi_grid, yi_grid = np.meshgrid(xi, yi)
                                
                                # Try different interpolation methods
                                zi = None
                                interpolation_errors = []
                                for method in ['linear', 'nearest', 'cubic']:
                                    try:
                                        zi = griddata((x_values, y_values), z_values, (xi_grid, yi_grid), 
                                                    method=method, fill_value=np.nan)
                                        if not np.all(np.isnan(zi)):
                                            break
                                    except Exception as e:
                                        interpolation_errors.append(f"{method}: {str(e)}")
                                        continue
                                
                                # Add contour plot if interpolation was successful
                                if zi is not None and not np.all(np.isnan(zi)):
                                    fig.add_trace(
                                        go.Contour(
                                            x=xi,
                                            y=yi,
                                            z=zi,
                                            colorscale='RdBu_r',
                                            showscale=(i == 0),  # Only show colorbar for first plot
                                            colorbar=dict(
                                                title='Difference',
                                                titleside='right'
                                            ),
                                            opacity=0.6,
                                            contours=dict(
                                                showlabels=True,
                                                labelfont=dict(size=8, color='white')
                                            ),
                                            name='Surface',
                                            showlegend=False,
                                            zmid=0,  # Center the color scale at 0
                                            zmin=-max_abs_diff,  # Set symmetric color scale
                                            zmax=max_abs_diff
                                        ),
                                        row=row,
                                        col=col
                                    )
                                
                                # Add scatter points if show_points is True
                                if show_points:
                                    fig.add_trace(
                                        go.Scatter(
                                            x=x_values,
                                            y=y_values,
                                            mode='markers',
                                            marker=dict(
                                                size=8,
                                                color=z_values,
                                                colorscale='RdBu_r',
                                                line=dict(width=1, color='black'),
                                                showscale=False,
                                                cmid=0
                                            ),
                                            name='Data Points',
                                            showlegend=False,
                                            text=[f'{param1}: {x:.3f}<br>{param2}: {y:.3f}<br>Difference: {z:.3f}' 
                                                  for x, y, z in zip(x_values, y_values, z_values)],
                                            hovertemplate='%{text}<extra></extra>'
                                        ),
                                        row=row,
                                        col=col
                                    )
                            except Exception as e:
                                st.error(f"Error processing manifold for {model}: {str(e)}")
                
                # Update layout with centered title and axis labels
                title = f"Difference Manifolds - {selected_task} - {selected_metric}"
                if pretraining_task:
                    title = f"{pretraining_task} - {title}"
                
                fig.update_layout(
                    title=dict(
                        text=title,
                        x=0.5,
                        xanchor='center'
                    ),
                    height=400 * n_rows,
                    showlegend=False
                )
                
                # Add shared axis titles centered
                for i in range(1, n_cols + 1):
                    fig.update_xaxes(
                        title_text=param1.replace('_', ' ').title(),
                        row=n_rows,
                        col=i,
                        title_standoff=0,
                        title_font=dict(size=12)
                    )
                for i in range(1, n_rows + 1):
                    fig.update_yaxes(
                        title_text=param2.replace('_', ' ').title(),
                        row=i,
                        col=1,
                        title_standoff=0,
                        title_font=dict(size=12)
                    )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Data Overview")
        
        # Dataset info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Experiments", len(st.session_state.df))
        with col2:
            st.metric("Unique Models", st.session_state.df['model_type'].nunique())
        with col3:
            st.metric("Tasks", st.session_state.df['task'].nunique())
        
        # Model type distribution
        st.subheader("Model Type Distribution")
        model_counts = st.session_state.df['model_type'].value_counts()
        fig_bar = px.bar(x=model_counts.index, y=model_counts.values, 
                        title="Number of Experiments by Model Type")
        fig_bar.update_layout(xaxis_title="Model Type", yaxis_title="Count")
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Raw data table
        st.subheader("Raw Data")
        st.dataframe(st.session_state.df, use_container_width=True)
        
        # Download button
        csv = st.session_state.df.to_csv(index=False)
        st.download_button(
            label="📥 Download data as CSV",
            data=csv,
            file_name="experiment_results.csv",
            mime="text/csv"
        )

    with tab5:
        st.header("t-SNE Visualization")
        
        # Filter for experiments that have t-SNE data
        tsne_data = st.session_state.df[st.session_state.df['tsne_coordinates'].notna()]
        
        if len(tsne_data) == 0:
            st.warning("No t-SNE data available in the loaded results.")
            return
        
        # Select experiment
        exp_options = tsne_data[['exp_dir', 'sub_dir', 'task', 'model_type']].apply(
            lambda x: f"{x['exp_dir']} - {x['sub_dir']} - {x['task']} - {x['model_type']}", axis=1
        ).unique()
        
        selected_exp = st.selectbox("Select Experiment", exp_options)
        
        if selected_exp:
            # Parse selected experiment
            exp_dir, sub_dir, task, model_type = selected_exp.split(" - ")
            
            # Get data for selected experiment
            exp_data = tsne_data[
                (tsne_data['exp_dir'] == exp_dir) & 
                (tsne_data['sub_dir'] == sub_dir) & 
                (tsne_data['task'] == task)
            ]
            
            # Get finetuned and from_scratch data
            finetuned_data = exp_data[exp_data['model_type'] == 'finetuned'].iloc[0]
            from_scratch_data = exp_data[exp_data['model_type'] == 'from_scratch'].iloc[0]
            
            # Display silhouette scores
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Finetuned Silhouette Score", f"{finetuned_data['first_silhouette_score']:.4f}")
            with col2:
                st.metric("From Scratch Silhouette Score", f"{from_scratch_data['first_silhouette_score']:.4f}")
            
            # Create t-SNE plots
            col1, col2 = st.columns(2)
            with col1:
                fig_finetuned = create_tsne_plot(
                    finetuned_data['tsne_coordinates'],
                    f"Finetuned t-SNE Plot (Silhouette: {finetuned_data['first_silhouette_score']:.4f})"
                )
                st.plotly_chart(fig_finetuned, use_container_width=True)
            
            with col2:
                fig_from_scratch = create_tsne_plot(
                    from_scratch_data['tsne_coordinates'],
                    f"From Scratch t-SNE Plot (Silhouette: {from_scratch_data['first_silhouette_score']:.4f})"
                )
                st.plotly_chart(fig_from_scratch, use_container_width=True)
            
            # Display experiment information
            st.subheader("Experiment Information")
            
            # Create columns for different types of information
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Model Information**")
                st.markdown(f"- Base Model: {finetuned_data['base_model']}")
                st.markdown(f"- Task: {task}")
                st.markdown(f"- Training Time: {finetuned_data['train_time']:.2f}s")
                st.markdown(f"- Best Epoch: {finetuned_data['best_epoch']}")
            
            with col2:
                st.markdown("**Family Properties**")
                st.markdown(f"- Nodes: {finetuned_data.get('node_counts_mean', 'N/A'):.1f} ± {finetuned_data.get('node_counts_std', 'N/A'):.1f}")
                st.markdown(f"- Edges: {finetuned_data.get('edge_counts_mean', 'N/A'):.1f} ± {finetuned_data.get('edge_counts_std', 'N/A'):.1f}")
                st.markdown(f"- Density: {finetuned_data.get('densities_mean', 'N/A'):.3f} ± {finetuned_data.get('densities_std', 'N/A'):.3f}")
                if 'community_counts_mean' in finetuned_data and 'community_counts_std' in finetuned_data:
                    st.markdown(f"- Communities: {finetuned_data['community_counts_mean']:.1f} ± {finetuned_data['community_counts_std']:.1f}")
                else:
                    st.markdown("- Communities: N/A")
            
            with col3:
                st.markdown("**Consistency Scores**")
                st.markdown(f"- Pattern Preservation: {finetuned_data['pattern_preservation_score']:.3f}")
                st.markdown(f"- Generation Fidelity: {finetuned_data['generation_fidelity_score']:.3f}")
                st.markdown(f"- Degree Consistency: {finetuned_data['degree_consistency_score']:.3f}")

if __name__ == "__main__":
    main()