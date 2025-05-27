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
        
        # Add model results
        for task_name, task_results in result['model_results'].items():
            for model_name, model_data in task_results.items():
                if model_data['success']:
                    for metric, value in model_data['test_metrics'].items():
                        base_row[f'{task_name}_{model_name}_{metric}'] = value
                    base_row[f'{task_name}_{model_name}_train_time'] = model_data['train_time']
        
        rows.append(base_row)
    
    return pd.DataFrame(rows)

def calculate_model_rankings(df, task='community', metric='accuracy'):
    """Calculate model rankings for each run and overall averages."""
    model_columns = [col for col in df.columns if f'{task}_' in col and f'_{metric}' in col and 'train_time' not in col]
    model_names = [col.replace(f'{task}_', '').replace(f'_{metric}', '') for col in model_columns]
    
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
    for idx, row in ranking_df.iterrows():
        # Rank in descending order (higher metric = better = lower rank number)
        ranks = rankdata(-row.values, method='min')
        ranking_df.loc[idx] = ranks
    
    # Rename columns to indicate rankings
    ranking_df.columns = [f'{task}_{name}_rank' for name in model_names]
    
    # Calculate average rankings
    avg_rankings = ranking_df.mean()
    
    # Calculate distance from average ranking for each run
    distance_df = ranking_df.copy()
    for col in ranking_df.columns:
        distance_df[col] = ranking_df[col] - avg_rankings[col]
    
    # Return the indices of clean runs so we can filter the main dataframe too
    clean_indices = df_clean.index
    
    return ranking_df, distance_df, avg_rankings, model_names, clean_indices

def create_2d_parameter_manifold(df, param1, param2, model, task='community', metric='accuracy', distance_df=None):
    """Create a 2D manifold plot showing model performance across parameter space."""
    if distance_df is not None:
        # Use ranking distance
        z_col = f'{task}_{model}_rank'
        z_values = distance_df[z_col].values
        title = f'{model.upper()}'
        colorbar_title = 'Ranking Distance'
        colorscale = 'RdBu'  # Changed to RdBu (red for negative/better, blue for positive/worse)
    else:
        # Use raw metric values
        z_col = f'{task}_{model}_{metric}'
        z_values = df[z_col].values
        title = f'{model.upper()}'
        colorbar_title = metric.capitalize()
        colorscale = 'Viridis'
    
    x_values = df[param1].values
    y_values = df[param2].values
    
    fig = go.Figure()
    
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
                    showscale=False,  # Hide colorbar for contour to avoid duplication
                    opacity=0.7,
                    contours=dict(
                        showlabels=True,
                        labelfont=dict(size=8, color='white')
                    ),
                    name='Interpolated Surface'
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
            colorbar=dict(title=colorbar_title)
        ),
        name='Data Points',
        text=[f'{param1}: {x:.3f}<br>{param2}: {y:.3f}<br>{colorbar_title}: {z:.3f}' 
              for x, y, z in zip(x_values, y_values, z_values)],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        showlegend=False,  # Hide individual legends
        width=400,
        height=400
    )
    
    return fig

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
            json_file = experiments_dir / selected_experiment / "final_results.json"
            
            if json_file.exists():
                data = load_experiment_data(json_file)
                df = process_results_to_dataframe(data)
                
                # Display basic info
                st.sidebar.success(f"✅ Loaded {len(df)} successful runs")
                st.sidebar.info(f"📊 Total runs attempted: {data['summary_stats']['total_runs_attempted']}")
                st.sidebar.info(f"⏱️ Success rate: {data['summary_stats']['success_rate']:.1%}")
                
            else:
                st.error(f"JSON file not found: {json_file}")
                return
        else:
            st.error("No experiment folders found in experiments/multi_results/")
            return
    else:
        st.error("experiments/multi_results/ directory not found")
        return
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Basic Plots", "🏆 Model Rankings", "🗺️ Parameter Manifolds", "📊 Summary Stats", "🔍 Data Explorer"])
    
    with tab1:
        st.header("Basic 2D Plotting")
        
        col1, col2, col3 = st.columns(3)
        
        # Get available parameters and metrics
        param_columns = [col for col in df.columns if col.startswith(('sweep_', 'random_', 'family_', 'consistency_', 'signal_'))]
        metric_columns = [col for col in df.columns if any(metric in col for metric in ['accuracy', 'f1_macro', 'precision', 'recall']) and 'train_time' not in col]
        model_names = list(set([col.split('_')[1] for col in metric_columns if '_' in col]))
        
        with col1:
            x_param = st.selectbox("X-axis Parameter:", param_columns, key="basic_x")
            
        with col2:
            y_metric = st.selectbox("Y-axis Metric:", metric_columns, key="basic_y")
            
        with col3:
            selected_models = st.multiselect("Models to Plot:", model_names, default=model_names[:3], key="basic_models")
        
        if selected_models:
            # Filter out runs with NaN values for the selected models
            relevant_columns = []
            for model in selected_models:
                model_metric_cols = [col for col in metric_columns if f'_{model}_' in col and y_metric.split('_')[-1] in col]
                relevant_columns.extend(model_metric_cols)
            
            # Only use runs that have complete data for all selected models
            df_plot = df[relevant_columns + [x_param]].dropna()
            
            if len(df_plot) < len(df):
                st.info(f"ℹ️ Using {len(df_plot)} runs with complete data (filtered out {len(df) - len(df_plot)} runs with NaN values)")
            
            if len(df_plot) == 0:
                st.error("❌ No complete data found for the selected models and metric!")
            else:
                fig = go.Figure()
                
                for model in selected_models:
                    model_metric_col = [col for col in metric_columns if f'_{model}_' in col and y_metric.split('_')[-1] in col][0]
                    
                    fig.add_trace(go.Scatter(
                        x=df_plot[x_param],
                        y=df_plot[model_metric_col],
                        mode='markers',
                        name=model.upper(),
                        marker=dict(size=8, opacity=0.7)
                    ))
                
                fig.update_layout(
                    title=f"{y_metric} vs {x_param}",
                    xaxis_title=x_param,
                    yaxis_title=y_metric,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Model Rankings Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            task = st.selectbox("Task:", ['community'], key="rank_task")  # Extend as needed
            metric = st.selectbox("Metric:", ['accuracy', 'f1_macro', 'precision', 'recall'], key="rank_metric")
        
        ranking_df, distance_df, avg_rankings, model_names, clean_indices = calculate_model_rankings(df, task, metric)
        
        # Average rankings bar chart
        fig_avg = go.Figure(data=[
            go.Bar(
                x=model_names,
                y=avg_rankings.values,
                text=[f'{val:.2f}' for val in avg_rankings.values],
                textposition='auto'
            )
        ])
        fig_avg.update_layout(
            title=f"Average Model Rankings ({metric})",
            xaxis_title="Model",
            yaxis_title="Average Rank (1=best)",
            height=400
        )
        
        st.plotly_chart(fig_avg, use_container_width=True)
        
        # Rankings distribution
        st.subheader("Ranking Distributions")
        
        cols = st.columns(len(model_names))
        for i, model in enumerate(model_names):
            with cols[i]:
                rank_col = f'{task}_{model}_rank'
                fig_dist = px.histogram(
                    ranking_df, 
                    x=rank_col,
                    title=f"{model.upper()}<br>Rank Distribution",
                    nbins=int(max(ranking_df[rank_col])) if max(ranking_df[rank_col]) > 1 else 5
                )
                fig_dist.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab3:
        st.header("Parameter Space Manifolds")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            param1 = st.selectbox("Parameter 1 (X-axis):", param_columns, key="manifold_x")
        with col2:
            param2 = st.selectbox("Parameter 2 (Y-axis):", param_columns, key="manifold_y", 
                                 index=min(1, len(param_columns)-1))
        with col3:
            use_rankings = st.checkbox("Use Ranking Distance", value=True, key="use_rankings")
            manifold_task = st.selectbox("Task:", ['community'], key="manifold_task")
            manifold_metric = st.selectbox("Metric:", ['accuracy', 'f1_macro', 'precision', 'recall'], key="manifold_metric")
        
        if param1 != param2:
            result = calculate_model_rankings(df, manifold_task, manifold_metric)
            if result[0] is None:  # No complete runs found
                st.error("Cannot create manifolds - no complete data available for the selected task/metric.")
            else:
                ranking_df, distance_df, avg_rankings, model_names, clean_indices = result
                
                # Filter the main dataframe to only include clean runs
                df_clean = df.loc[clean_indices]
                
                # Option 1: Show individual plots side by side (better for detailed analysis)
                show_as_grid = st.checkbox("Show as single grid plot", value=False)
                
                if show_as_grid:
                    # Create subplot grid with square organization
                    n_models = len(model_names)
                    n_cols = int(np.ceil(np.sqrt(n_models)))
                    n_rows = int(np.ceil(n_models / n_cols))
                    
                    subplot_titles = [f"{model.upper()}" for model in model_names]
                    
                    fig = make_subplots(
                        rows=n_rows,
                        cols=n_cols,
                        subplot_titles=subplot_titles,
                        vertical_spacing=0.1,
                        horizontal_spacing=0.1,
                        shared_xaxes=True,
                        shared_yaxes=True
                    )
                    
                    for i, model in enumerate(model_names):
                        row = (i // n_cols) + 1
                        col = (i % n_cols) + 1
                        
                        # Create individual manifold
                        individual_fig = create_2d_parameter_manifold(
                            df_clean, param1, param2, model, manifold_task, manifold_metric,
                            distance_df if use_rankings else None
                        )
                        
                        # Add traces to subplot
                        for j, trace in enumerate(individual_fig.data):
                            trace.showlegend = (i == 0)  # Only show legend for first plot
                            fig.add_trace(trace, row=row, col=col)
                    
                    # Update layout with shared axis titles
                    fig.update_layout(
                        title=f"Parameter Space Analysis: {param1} vs {param2}",
                        height=400 * n_rows,
                        width=400 * n_cols,
                        showlegend=True,
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=1,
                            xanchor="left",
                            x=1.02
                        )
                    )
                    
                    # Add shared axis titles
                    fig.update_xaxes(title_text=param1, row=n_rows, col=1)
                    fig.update_yaxes(title_text=param2, row=1, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    # Show individual plots in columns (preserves legends and interactivity)
                    st.subheader(f"Parameter Space Analysis: {param1} vs {param2}")
                    
                    # Calculate square grid dimensions
                    n_models = len(model_names)
                    n_cols = int(np.ceil(np.sqrt(n_models)))
                    n_rows = int(np.ceil(n_models / n_cols))
                    
                    for row in range(n_rows):
                        cols = st.columns(n_cols)
                        
                        for col in range(n_cols):
                            idx = row * n_cols + col
                            if idx < len(model_names):
                                with cols[col]:
                                    individual_fig = create_2d_parameter_manifold(
                                        df_clean, param1, param2, model_names[idx], 
                                        manifold_task, manifold_metric,
                                        distance_df if use_rankings else None
                                    )
                                    st.plotly_chart(individual_fig, use_container_width=True)
        else:
            st.warning("Please select different parameters for X and Y axes.")
    
    with tab4:
        st.header("Summary Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Experiment Overview")
            st.json({
                "Total Runs": data['summary_stats']['total_runs_attempted'],
                "Successful Runs": data['summary_stats']['successful_runs'],
                "Failed Runs": data['summary_stats']['failed_runs'],
                "Success Rate": f"{data['summary_stats']['success_rate']:.1%}",
                "Total Time": f"{data['summary_stats']['total_time']:.1f}s",
                "Avg Time per Run": f"{data['summary_stats']['avg_time_per_run']:.1f}s"
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
        display_columns = st.multiselect(
            "Select Columns to Display:",
            df.columns.tolist(),
            default=['run_id', 'method', 'degree_distribution', 'n_graphs'] + metric_columns[:5]
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

if __name__ == "__main__":
    main()