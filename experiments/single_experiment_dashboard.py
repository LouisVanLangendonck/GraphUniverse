"""
Interactive dashboard for visualizing MMSB experiment results.

Usage:
    streamlit run experiments/single_experiment_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import glob

# Set page config
st.set_page_config(
    page_title="MMSB Experiment Results",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4B89DC;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #3D5A80;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #F0F4F8;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def load_experiment_data(results_dir: str) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Load experiment results and metadata."""
    # Load CSV data
    df = pd.read_csv(Path(results_dir) / "all_results.csv")
    
    # Load metadata
    with open(Path(results_dir) / "experiment_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    return df, metadata

def get_model_columns(df: pd.DataFrame) -> tuple[Dict[str, List[str]], List[str]]:
    """Get columns grouped by model and metric type."""
    model_cols = {}
    metric_names = set()
    
    # Find all unique model prefixes and metric names
    for col in df.columns:
        # Check for task_model_metric pattern (e.g., community_GCN_accuracy)
        parts = col.split('_')
        if len(parts) >= 3 and parts[0] in ['community', 'regime', 'role']:
            task = parts[0]
            model_name = parts[1]
            metric = '_'.join(parts[2:])  # Join remaining parts as metric name
            
            model_key = f"{task}_{model_name}"
            if model_key not in model_cols:
                model_cols[model_key] = []
            model_cols[model_key].append(col)
            metric_names.add(metric)
    
    return model_cols, sorted(list(metric_names))

def get_parameter_columns(df: pd.DataFrame) -> List[str]:
    """Get columns that represent experiment parameters."""
    # Parameters are columns that aren't model metrics or metadata
    exclude_prefixes = ['community_', 'regime_', 'role_', 'experiment_', 'graph_']
    return [col for col in df.columns if not any(col.startswith(prefix) for prefix in exclude_prefixes)]

def create_scatter_plot(
    df: pd.DataFrame,
    x_param: str,
    y_metric: str,
    color_param: Optional[str] = None,
    size_param: Optional[str] = None,
    models: Optional[List[str]] = None
) -> go.Figure:
    """Create an interactive scatter plot."""
    fig = go.Figure()
    
    if models:
        for model in models:
            # Get the specific metric column for this model
            model_y_col = f"{model}_{y_metric}"
            if model_y_col not in df.columns:
                continue
                
            fig.add_trace(go.Scatter(
                x=df[x_param],
                y=df[model_y_col],
                mode='markers',
                name=model,
                marker=dict(
                    size=10 if not size_param else df[size_param],
                    colorscale='Viridis' if color_param else None,
                    color=df[color_param] if color_param else None,
                    showscale=True if color_param else False
                )
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df[x_param],
            y=df[y_metric],
            mode='markers',
            marker=dict(
                size=10 if not size_param else df[size_param],
                colorscale='Viridis' if color_param else None,
                color=df[color_param] if color_param else None,
                showscale=True if color_param else False
            )
        ))
    
    fig.update_layout(
        title=f"{y_metric} vs {x_param}",
        xaxis_title=x_param,
        yaxis_title=y_metric,
        hovermode='closest'
    )
    
    return fig

def create_3d_scatter_plot(
    df: pd.DataFrame,
    x_param: str,
    y_param: str,
    z_metric: str,
    color_param: Optional[str] = None,
    models: Optional[List[str]] = None
) -> go.Figure:
    """Create an interactive 3D scatter plot."""
    fig = go.Figure()
    
    if models:
        for model in models:
            # Get the specific metric column for this model
            model_z_col = f"{model}_{z_metric}"
            if model_z_col not in df.columns:
                continue
                
            fig.add_trace(go.Scatter3d(
                x=df[x_param],
                y=df[y_param],
                z=df[model_z_col],
                mode='markers',
                name=model,
                marker=dict(
                    size=5,
                    colorscale='Viridis' if color_param else None,
                    color=df[color_param] if color_param else None,
                    showscale=True if color_param else False
                )
            ))
    else:
        fig.add_trace(go.Scatter3d(
            x=df[x_param],
            y=df[y_param],
            z=df[z_metric],
            mode='markers',
            marker=dict(
                size=5,
                colorscale='Viridis' if color_param else None,
                color=df[color_param] if color_param else None,
                showscale=True if color_param else False
            )
        ))
    
    fig.update_layout(
        title=f"3D Plot: {x_param} vs {y_param} vs {z_metric}",
        scene=dict(
            xaxis_title=x_param,
            yaxis_title=y_param,
            zaxis_title=z_metric
        )
    )
    
    return fig

# Main dashboard
st.markdown('<div class="main-header">MMSB Experiment Results Dashboard</div>', unsafe_allow_html=True)

# Sidebar - Load Data
st.sidebar.title("Data Selection")

# Find all result directories
result_dirs = glob.glob("single_graph_multiple_results/*/")
if not result_dirs:
    st.error("No experiment results found in 'single_graph_multiple_results' directory")
    st.stop()

# Select experiment
selected_dir = st.sidebar.selectbox(
    "Select Experiment",
    result_dirs,
    format_func=lambda x: Path(x).name
)

# Load data
df, metadata = load_experiment_data(selected_dir)

# Display experiment metadata
st.markdown('<div class="section-header">Experiment Overview</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Experiments", len(df))
    st.metric("Parameters Varied", len(metadata['varied_parameters']))

with col2:
    st.metric("GNN Models", len(metadata['model_types']['gnn']))
    st.metric("Other Models", int(metadata['model_types']['mlp']) + int(metadata['model_types']['rf']))

with col3:
    st.metric("Max Epochs", metadata['training_params']['epochs'])
    st.metric("Early Stopping Patience", metadata['training_params']['patience'])

# Get available columns for plotting
model_cols, metric_names = get_model_columns(df)

# Get available tasks
available_tasks = sorted(list(set(col.split('_')[0] for col in df.columns 
                              if any(col.startswith(f"{task}_") for task in ['community', 'regime', 'role']))))

# Task Selection
st.sidebar.markdown("### Task Selection")
selected_task = st.sidebar.selectbox(
    "Select Task",
    available_tasks,
    index=0 if 'community' in available_tasks else 0
)

# Filter model columns for selected task
task_model_cols = {k: v for k, v in model_cols.items() if k.startswith(selected_task)}

# Model Selection
st.sidebar.markdown("### Model Selection")
selected_models = st.sidebar.multiselect(
    "Select Models to Compare",
    list(task_model_cols.keys()),
    default=list(task_model_cols.keys())[:2]
)

# Get parameter columns
param_cols = get_parameter_columns(df)

# Visualization Options
st.markdown('<div class="section-header">Visualization</div>', unsafe_allow_html=True)

viz_type = st.radio(
    "Visualization Type",
    ["2D Scatter Plot", "3D Scatter Plot"]
)

if viz_type == "2D Scatter Plot":
    col1, col2 = st.columns(2)
    
    with col1:
        x_param = st.selectbox("X-axis Parameter", param_cols)
        y_metric = st.selectbox("Y-axis Metric", metric_names)
    
    with col2:
        color_param = st.selectbox("Color Parameter (optional)", ["None"] + param_cols)
        size_param = st.selectbox("Size Parameter (optional)", ["None"] + param_cols)
    
    # Create plot
    fig = create_scatter_plot(
        df,
        x_param,
        y_metric,
        color_param if color_param != "None" else None,
        size_param if size_param != "None" else None,
        selected_models
    )
    st.plotly_chart(fig, use_container_width=True)

else:  # 3D Scatter Plot
    col1, col2 = st.columns(2)
    
    with col1:
        x_param = st.selectbox("X-axis Parameter", param_cols)
        y_param = st.selectbox("Y-axis Parameter", param_cols)
        z_metric = st.selectbox("Z-axis Metric", metric_names)
    
    with col2:
        color_param = st.selectbox("Color Parameter (optional)", ["None"] + param_cols)
    
    # Create 3D plot
    fig = create_3d_scatter_plot(
        df,
        x_param,
        y_param,
        z_metric,
        color_param if color_param != "None" else None,
        selected_models
    )
    st.plotly_chart(fig, use_container_width=True)

# Data Table View
st.markdown('<div class="section-header">Data Table</div>', unsafe_allow_html=True)
show_data = st.checkbox("Show Raw Data")
if show_data:
    st.dataframe(df) 