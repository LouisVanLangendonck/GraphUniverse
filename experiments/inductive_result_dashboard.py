"""
Interactive dashboard for visualizing inductive graph learning experiment results.

Usage:
    streamlit run experiments/inductive_result_dashboard.py
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
    page_title="Inductive Graph Learning Results",
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

def load_experiment_data(results_dir: str) -> Dict[str, Any]:
    """Load experiment results from JSON."""
    with open(Path(results_dir) / "final_results.json", 'r') as f:
        data = json.load(f)
    return data

def get_available_tasks_and_metrics(data: Dict[str, Any]) -> Tuple[List[str], Dict[str, List[str]]]:
    """Get available tasks and their metrics from the data."""
    tasks = set()
    task_metrics = {}
    
    # Get all tasks and metrics from the first experiment
    if data['all_results']:
        first_exp = data['all_results'][0]
        for task in first_exp['model_results'].keys():
            tasks.add(task)
            
            # Get metrics for this task
            if task not in task_metrics:
                task_metrics[task] = set()
            
            # Get metrics from model results, removing model prefixes
            model_data = first_exp['model_results'][task]
            for metric_key in model_data.keys():
                # Split on first underscore to separate model name from metric
                parts = metric_key.split('_', 1)
                if len(parts) == 2:
                    metric_name = parts[1]  # Get the metric name without model prefix
                    task_metrics[task].add(metric_name)
    
    return sorted(list(tasks)), {task: sorted(list(metrics)) for task, metrics in task_metrics.items()}

def get_numeric_properties(data: Dict[str, Any]) -> List[str]:
    """Get all numeric properties that can be used as x-axis."""
    if not data['all_results']:
        return []
    
    first_exp = data['all_results'][0]
    exclude = {'model_results', 'run_id', 'timestamp', 'config_path', 'family_stats'}
    
    # Get sweep parameters
    sweep_params = list(data['config']['sweep_parameters'].keys())
    
    # Get random parameters
    random_params = list(data['config']['random_parameters'].keys())
    
    # Get family stats properties
    family_stats = list(first_exp['family_stats'].keys())
    family_stats = [f"family_stats.{stat}" for stat in family_stats if isinstance(first_exp['family_stats'][stat], (int, float))]
    
    return sweep_params + random_params + family_stats

def create_metric_plot(
    data: Dict[str, Any],
    task: str,
    x_property: str,
    metric: str,
    selected_models: List[str],
    title_suffix: str = ""
) -> go.Figure:
    """Create a plot for a specific metric showing all experiments as dots."""
    fig = go.Figure()
    
    # Define colors for models
    model_colors = {
        'gcn': '#1f77b4',
        'sage': '#ff7f0e',
        'mlp': '#2ca02c',
        'rf': '#d62728'
    }
    
    # Add model color legend entries
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
    
    # Add actual data traces
    for model in selected_models:
        x_values = []
        y_values = []
        for exp in data['all_results']:
            if task in exp['model_results']:
                model_data = exp['model_results'][task]
                metric_key = f"{model}_{metric}"  # Construct the full metric key
                if metric_key in model_data:
                    # Get x value based on property type
                    if x_property.startswith('family_stats.'):
                        stat_name = x_property.split('.')[1]
                        x_value = exp['family_stats'][stat_name]
                    else:
                        x_value = exp['all_parameters'][x_property]
                    
                    x_values.append(x_value)
                    y_values.append(model_data[metric_key])
        
        if x_values and y_values:
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='markers',
                name=None,
                marker=dict(
                    symbol='circle',
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
st.markdown('<div class="main-header">Inductive Graph Learning Results Dashboard</div>', unsafe_allow_html=True)

# Sidebar - Load Data
st.sidebar.header("Data Selection")
results_dir = st.sidebar.text_input("Results Directory", "multi_inductive_results/multi_inductive_20250526_114836")

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

# Second graph options
st.sidebar.markdown("---")
st.sidebar.header("Second Graph Options")
show_second_graph = st.sidebar.checkbox("Show Second Graph", value=False)

if show_second_graph:
    second_task = st.sidebar.selectbox("Select Second Task", tasks, index=1 if len(tasks) > 1 else 0)
    second_metric = st.sidebar.selectbox("Select Second Metric", task_metrics[second_task])
    second_x_property = st.sidebar.selectbox("Select Second X-axis", numeric_properties, index=1 if len(numeric_properties) > 1 else 0)

# Create and display plots
if selected_models:
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown(f'<div style="margin-bottom: 20px;"></div>', unsafe_allow_html=True)
        fig1 = create_metric_plot(data, task, x_property, metric, selected_models)
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
                " (Second Graph)"
            )
            st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("Please select at least one model to display")

# Display experiment metadata
st.markdown('<div class="section-header">Experiment Overview</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.metric("Total Experiments", len(data['all_results']))
    st.metric("Available Tasks", len(tasks))
    st.metric("Success Rate", f"{data['summary_stats']['success_rate']:.1%}")

with col2:
    st.metric("Available Models", len(available_models))
    st.metric("Available Metrics", len(available_metrics))
    st.metric("Total Time", f"{data['summary_stats']['total_time']/3600:.1f}h")

# Data Table View
st.markdown('<div class="section-header">Data Table</div>', unsafe_allow_html=True)
show_data = st.checkbox("Show Raw Data")
if show_data:
    st.json(data) 