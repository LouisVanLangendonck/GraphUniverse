import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import numpy as np
from pathlib import Path

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

def extract_model_metrics(model_results):
    """Extract model performance metrics with fold averaging."""
    metrics = {}
    if not model_results:
        return metrics
    
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
                                key_mean = f"model.{task_name}.{model_name}.{metric_name}.mean"
                                key_ci = f"model.{task_name}.{model_name}.{metric_name}.ci95"
                                metrics[key_mean] = mean_val
                                metrics[key_ci] = ci_95
                    
                    # Fallback for non-fold structure
                    else:
                        for metric_name, metric_value in model_data.items():
                            if isinstance(metric_value, (int, float)):
                                key = f"model.{task_name}.{model_name}.{metric_name}"
                                metrics[key] = metric_value
    return metrics

def create_dataframe_from_results(results_list):
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
            model_metrics = extract_model_metrics(run_sample['model_results'])
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
        "multi_results/final_sweep_community",
        "multi_results/final_sweep_community_2"
    ],
    index=0,
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
            st.session_state.df = create_dataframe_from_results(results_list)
            st.success(f"Successfully loaded {len(results_list)} experiment runs!")

# Main content
if 'df' in st.session_state and st.session_state.df is not None:
    df = st.session_state.df
    
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
        
        metric_options = sorted(list(available_metrics))
        selected_metric = st.selectbox(
            "Select metric type:",
            options=metric_options,
            index=0 if metric_options else None
        )
    
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
    
    # Plot type selection
    plot_type = st.radio(
        "Plot Type:",
        options=['Scatter', 'Line', 'Box Plot'],
        horizontal=True
    )
    
    # Generate plot
    if x_axis and selected_metric and selected_models:
        fig = None
        
        # Filter out NaN values for the first model to check data availability
        first_model_col = f"model.community.{selected_models[0]}.{selected_metric}.mean"
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
                        horizontal_spacing=0.1
                    )
                    
                    # Define a scientific color palette
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                    
                    for i, value in enumerate(sorted(unique_values)):
                        # Calculate subplot position
                        row = (i // n_cols) + 1
                        col = (i % n_cols) + 1
                        
                        # Filter data for this value
                        filtered_df = plot_df[plot_df[second_param] == value]
                        
                        for j, model in enumerate(selected_models):
                            model_metric_col = f"model.community.{model}.{selected_metric}.mean"
                            model_error_col = f"model.community.{model}.{selected_metric}.ci95"
                            
                            if model_metric_col in filtered_df.columns:
                                model_data = filtered_df.dropna(subset=[model_metric_col])
                                
                                if len(model_data) > 0:
                                    error_y_data = None
                                    if model_error_col in model_data.columns:
                                        error_y_data = model_data[model_error_col]
                                    
                                    color = colors[j % len(colors)]
                                    
                                    if plot_type == 'Scatter':
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
                                    elif plot_type == 'Line':
                                        fig.add_trace(go.Scatter(
                                            x=model_data[x_axis],
                                            y=model_data[model_metric_col],
                                            mode='lines+markers',
                                            name=model,
                                            line=dict(color=color, width=2),
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
                    
                    # Apply R-style scientific formatting to all subplots
                    fig.update_layout(
                        title=dict(
                            text=f"{selected_metric} vs {x_axis} by {second_param}",
                            x=0.5,
                            font=dict(size=16, color='black')
                        ),
                        legend=dict(
                            x=0.02,
                            y=0.98,
                            bgcolor='rgba(255,255,255,0.8)',
                            bordercolor='black',
                            borderwidth=1,
                            font=dict(size=12)
                        ),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=300 * n_rows,  # Adjust height based on number of rows
                        margin=dict(l=60, r=30, t=80, b=60)
                    )
                    
                    # Update all subplot axes
                    for i in range(1, n_rows + 1):
                        for j in range(1, n_cols + 1):
                            fig.update_xaxes(
                                title=dict(text=x_axis, font=dict(size=12)),
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='lightgray',
                                zeroline=True,
                                zerolinewidth=1,
                                zerolinecolor='black',
                                row=i, col=j
                            )
                            fig.update_yaxes(
                                title=dict(text=selected_metric, font=dict(size=12)),
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='lightgray',
                                zeroline=True,
                                zerolinewidth=1,
                                zerolinecolor='black',
                                row=i, col=j
                            )
                    
                else:
                    # Use dot size for many values
                    fig = go.Figure()
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                    
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
                                
                                if plot_type == 'Scatter':
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
                                elif plot_type == 'Line':
                                    fig.add_trace(go.Scatter(
                                        x=model_data[x_axis],
                                        y=model_data[model_metric_col],
                                        mode='lines+markers',
                                        name=model,
                                        line=dict(color=color, width=2),
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
                    
                    # Apply R-style scientific formatting
                    fig.update_layout(
                        title=dict(
                            text=f"{selected_metric} vs {x_axis} (dot size = {second_param})",
                            x=0.5,
                            font=dict(size=16, color='black')
                        ),
                        xaxis=dict(
                            title=dict(text=x_axis, font=dict(size=14)),
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='lightgray',
                            zeroline=True,
                            zerolinewidth=1,
                            zerolinecolor='black'
                        ),
                        yaxis=dict(
                            title=dict(text=selected_metric, font=dict(size=14)),
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='lightgray',
                            zeroline=True,
                            zerolinewidth=1,
                            zerolinecolor='black'
                        ),
                        legend=dict(
                            x=0.02,
                            y=0.98,
                            bgcolor='rgba(255,255,255,0.8)',
                            bordercolor='black',
                            borderwidth=1,
                            font=dict(size=12)
                        ),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=600,
                        margin=dict(l=60, r=30, t=80, b=60)
                    )
            else:
                # Original single plot without second parameter
                fig = go.Figure()
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                
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
                            
                            if plot_type == 'Scatter':
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
                            elif plot_type == 'Line':
                                fig.add_trace(go.Scatter(
                                    x=model_data[x_axis],
                                    y=model_data[model_metric_col],
                                    mode='lines+markers',
                                    name=model,
                                    line=dict(color=color, width=2),
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
                
                # Apply R-style scientific formatting
                fig.update_layout(
                    title=dict(
                        text=f"{selected_metric} vs {x_axis}",
                        x=0.5,
                        font=dict(size=16, color='black')
                    ),
                    xaxis=dict(
                        title=dict(text=x_axis, font=dict(size=14)),
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray',
                        zeroline=True,
                        zerolinewidth=1,
                        zerolinecolor='black'
                    ),
                    yaxis=dict(
                        title=dict(text=selected_metric, font=dict(size=14)),
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray',
                        zeroline=True,
                        zerolinewidth=1,
                        zerolinecolor='black'
                    ),
                    legend=dict(
                        x=0.02,
                        y=0.98,
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='black',
                        borderwidth=1,
                        font=dict(size=12)
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    height=600,
                    margin=dict(l=60, r=30, t=80, b=60)
                )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            if fig:
                if not isinstance(fig, go.Figure):
                    fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
    
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