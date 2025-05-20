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
from typing import List, Dict, Any, Optional, Tuple
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

def load_experiment_data(results_dir: str) -> List[Dict[str, Any]]:
    """Load experiment results from JSON."""
    with open(Path(results_dir) / "all_results.json", 'r') as f:
        data = json.load(f)
    return data

def get_available_tasks_and_metrics(data: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, List[str]]]:
    """Get available tasks and their metrics from the data."""
    tasks = set()
    task_metrics = {}
    
    # Get all tasks and metrics from the first experiment
    if data:
        first_exp = data[0]
        for model_key in first_exp['model_results'].keys():
            task = model_key.split('--')[0]
            tasks.add(task)
            
            # Get metrics for this task
            if task not in task_metrics:
                task_metrics[task] = set()
            
            # Get metrics from any split (they should be the same)
            split_data = next(iter(first_exp['model_results'][model_key].values()))
            if isinstance(split_data, dict):
                task_metrics[task].update(split_data.keys())
    
    return sorted(list(tasks)), {task: sorted(list(metrics)) for task, metrics in task_metrics.items()}

def get_numeric_properties(data: List[Dict[str, Any]]) -> List[str]:
    """Get all numeric top-level properties that can be used as x-axis (including input params and graph- properties)."""
    if not data:
        return []
    first_exp = data[0]
    exclude = {'model_results', 'experiment_id', 'timestamp', 'combination', 'repeat'}
    numeric_keys = []
    for key, value in first_exp.items():
        if key not in exclude and isinstance(value, (int, float)):
            numeric_keys.append(key)
    return numeric_keys

def create_metric_plot(
    data: List[Dict[str, Any]],
    task: str,
    x_property: str,
    metric: str,
    selected_models: List[str],
    selected_splits: List[str],
    title_suffix: str = ""
) -> go.Figure:
    """Create a plot for a specific metric showing all experiments as dots with different marker symbols for splits."""
    fig = go.Figure()
    
    # Define marker symbols for splits
    split_markers = {
        'train': 'circle',
        'val': 'diamond',
        'test': 'square'
    }
    # Define colors for models
    model_colors = {
        'gcn': '#1f77b4',
        'sage': '#ff7f0e',
        'mlp': '#2ca02c',
        'rf': '#d62728'
    }
    # Add split marker legend entries in black (only once)
    for split in selected_splits:
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            name=f"{split}",
            marker=dict(
                symbol=split_markers[split],
                color='black',
                size=12
            ),
            showlegend=True
        ))
    # Add model color legend entries (only once)
    for model in selected_models:
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            name=f"{model.upper()}",
            marker=dict(
                symbol='circle',
                color=model_colors[model],
                size=12
            ),
            showlegend=True
        ))
    # Add actual data traces (no legend)
    for model in selected_models:
        model_key = f"{task}--{model}"
        for split in selected_splits:
            x_values = []
            y_values = []
            for exp in data:
                if model_key in exp['model_results']:
                    model_data = exp['model_results'][model_key]
                    if split in model_data and metric in model_data[split]:
                        x_values.append(exp[x_property])
                        y_values.append(model_data[split][metric])
            if x_values and y_values:
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='markers',
                    name=None,
                    marker=dict(
                        symbol=split_markers[split],
                        color=model_colors[model],
                        size=10
                    ),
                    showlegend=False
                ))
    # Update layout
    fig.update_layout(
        title=f"{task} {metric} vs {x_property}{title_suffix}",
        xaxis_title=x_property,
        yaxis_title=metric,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="left",
            x=0,
            font=dict(size=14)
        ),
        margin=dict(t=80, b=80)
    )
    return fig

# Main dashboard
st.markdown('<div class="main-header">MMSB Experiment Results Dashboard</div>', unsafe_allow_html=True)

# Sidebar - Load Data
st.sidebar.header("Data Selection")
results_dir = st.sidebar.text_input("Results Directory", "single_graph_multiple_results")

# Load data
try:
    data = load_experiment_data(results_dir)
    st.sidebar.success("Data loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Get available tasks and metrics
tasks, task_metrics = get_available_tasks_and_metrics(data)

# Task Selection
task = st.sidebar.selectbox("Select Task", tasks)

# Metric Selection
available_metrics = task_metrics[task]
metric = st.sidebar.selectbox("Select Metric", available_metrics)

# X-axis selection (numeric properties)
numeric_properties = get_numeric_properties(data)
x_property = st.sidebar.selectbox("Select X-axis (Numeric Property)", numeric_properties)

# Model selection
available_models = ['gcn', 'sage', 'mlp', 'rf']
selected_models = st.sidebar.multiselect(
    "Select Models to Display",
    available_models,
    default=available_models
)

# Split selection
splits = ['train', 'val', 'test']
selected_splits = st.sidebar.multiselect(
    "Select Splits to Display",
    splits,
    default=splits
)

# Second graph options
st.sidebar.markdown("---")
st.sidebar.header("Second Graph Options")
show_second_graph = st.sidebar.checkbox("Show Second Graph", value=False)

if show_second_graph:
    second_task = st.sidebar.selectbox("Select Second Task", tasks, index=1 if len(tasks) > 1 else 0)
    second_metric = st.sidebar.selectbox("Select Second Metric", task_metrics[second_task])
    second_x_property = st.sidebar.selectbox("Select Second X-axis", numeric_properties, index=1 if len(numeric_properties) > 1 else 0)

# Create and display plots
if selected_models and selected_splits:
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown(f'<div style="margin-bottom: 20px;"></div>', unsafe_allow_html=True)
        fig1 = create_metric_plot(data, task, x_property, metric, selected_models, selected_splits)
        st.plotly_chart(fig1, use_container_width=True)
    if show_second_graph:
        with col2:
            st.markdown(f'<div style="margin-bottom: 20px;"></div>', unsafe_allow_html=True)
            fig2 = create_metric_plot(
                data, 
                second_task, 
                second_x_property, 
                second_metric, 
                selected_models, 
                selected_splits,
                " (Second Graph)"
            )
            st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("Please select at least one model and one split to display")

# Display experiment metadata
st.markdown('<div class="section-header">Experiment Overview</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.metric("Total Experiments", len(data))
    st.metric("Available Tasks", len(tasks))

with col2:
    st.metric("Available Models", len(available_models))
    st.metric("Available Metrics", len(available_metrics))

# Data Table View
st.markdown('<div class="section-header">Data Table</div>', unsafe_allow_html=True)
show_data = st.checkbox("Show Raw Data")
if show_data:
    st.json(data) 