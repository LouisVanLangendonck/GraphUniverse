import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path
import numpy as np

# Set page config
st.set_page_config(page_title="GNN Experiment Results Dashboard", layout="wide")

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
                            'clustering_coefficients', 'community_counts'
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
                if col not in ['model_type'] + list(finetuned.index[finetuned.index.str.contains('mse|rmse|mae|r2|accuracy|precision|recall|f1')]):
                    if col in finetuned.index:
                        base_info[col] = finetuned[col]
            
            # Calculate differences
            for other_model in ['from_scratch'] + [gnn for gnn in models.keys() if gnn not in ['finetuned', 'from_scratch']]:
                if other_model in models:
                    other = models[other_model]
                    
                    # Calculate differences for all test metrics
                    for metric_col in df.columns:
                        if any(metric in metric_col for metric in ['mse', 'rmse', 'mae', 'r2', 'accuracy', 'precision', 'recall', 'f1']):
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

def create_difference_plot(diff_df, x_param, selected_models, selected_comparisons, y_metric='difference'):
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
        (diff_df['comparison'].isin(selected_comparisons))
    ]
    
    # Group by base_model and comparison for plotting
    for base_model in selected_models:
        for comparison in selected_comparisons:
            plot_data = filtered_df[
                (filtered_df['base_model'] == base_model) & 
                (filtered_df['comparison'] == comparison)
            ]
            
            if not plot_data.empty:
                # Create legend name
                legend_name = f"{base_model} - {comparison}"
                
                fig.add_trace(go.Scatter(
                    x=plot_data[x_param],
                    y=plot_data[y_metric],
                    mode='markers',
                    name=legend_name,
                    marker=dict(
                        symbol=comparison_markers.get(comparison, 'circle'),
                        size=8,
                        color=model_colors.get(base_model, '#000000'),
                        line=dict(width=1, color='black')
                    ),
                    hovertemplate=f'<b>{legend_name}</b><br>' +
                                 f'{x_param}: %{{x}}<br>' +
                                 f'{y_metric}: %{{y}}<br>' +
                                 f'Metric: {plot_data["metric_name"].iloc[0] if len(plot_data) > 0 else "N/A"}<br>' +
                                 '<extra></extra>'
                ))
    
    fig.update_layout(
        title=f'{y_metric} vs {x_param}',
        xaxis_title=x_param,
        yaxis_title=y_metric,
        hovermode='closest',
        showlegend=True,
        height=600
    )
    
    # Add horizontal line at y=0 for reference
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    
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
    tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "📈 Difference Analysis", "📋 Data Overview"])
    
    with tab1:
        st.header("Model Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # X-axis parameter selection - now using family properties instead of config
            family_stats_params = [
                'node_counts_mean', 'node_counts_std', 'node_counts_min', 'node_counts_max',
                'edge_counts_mean', 'edge_counts_std', 'edge_counts_min', 'edge_counts_max',
                'densities_mean', 'densities_std', 'densities_min', 'densities_max',
                'avg_degrees_mean', 'avg_degrees_std', 'avg_degrees_min', 'avg_degrees_max',
                'clustering_coefficients_mean', 'clustering_coefficients_std', 'clustering_coefficients_min', 'clustering_coefficients_max',
                'community_counts_mean', 'community_counts_std', 'community_counts_min', 'community_counts_max'
            ]
            
            consistency_params = ['pattern_preservation_score', 'generation_fidelity_score', 'degree_consistency_score']
            
            signal_params = [f'{signal}_{stat}' for signal in ['degree_signals', 'structure_signals', 'feature_signals'] 
                            for stat in ['mean', 'std', 'min', 'max']]
            
            all_x_params = family_stats_params + consistency_params + signal_params
            x_param = st.selectbox("Select X-axis parameter:", all_x_params, index=0, key="main_x_param")
        
        with col2:
            # Y-axis metric selection
            metric_columns = [col for col in st.session_state.df.columns if any(metric in col for metric in 
                            ['accuracy', 'f1_', 'precision_', 'recall_', 'mse', 'rmse', 'mae', 'r2'])]
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
        st.markdown("Compare finetuned models against from_scratch and optimized models")
        
        if not st.session_state.diff_df.empty and 'base_model' in st.session_state.diff_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Selection")
                # Model selection - show all available base models
                available_models = st.session_state.diff_df['base_model'].unique()
                
                if len(available_models) > 0:
                    selected_models = st.multiselect(
                        "Select models to include:",
                        options=available_models,
                        default=available_models,
                        key="selected_models"
                    )
                else:
                    st.warning("No models available for difference analysis")
                    selected_models = []
                
                # Comparison type selection
                available_comparisons = st.session_state.diff_df['comparison'].unique() if 'comparison' in st.session_state.diff_df.columns else []
                
                if len(available_comparisons) > 0:
                    selected_comparisons = st.multiselect(
                        "Select comparison types:",
                        options=available_comparisons,
                        default=available_comparisons,
                        key="selected_comparisons"
                    )
                else:
                    st.warning("No comparison types available")
                    selected_comparisons = []
            
            with col2:
                st.subheader("Plot Parameters")
                x_param_diff = st.selectbox("X-axis parameter:", all_x_params, key="diff_x")
                
                available_metrics = st.session_state.diff_df['metric_name'].unique() if 'metric_name' in st.session_state.diff_df.columns else []
                if available_metrics.size > 0:
                    selected_metric = st.selectbox("Select metric for difference:", available_metrics, key="diff_metric")
                else:
                    st.warning("No metrics available")
                    selected_metric = None
                
                available_tasks = st.session_state.diff_df['task'].unique() if 'task' in st.session_state.diff_df.columns else []
                if available_tasks.size > 0:
                    selected_task_diff = st.selectbox("Select task:", available_tasks, key="diff_task")
                else:
                    st.warning("No tasks available")
                    selected_task_diff = None
            
            # Show color/marker legend
            st.subheader("Legend")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Colors (Models):**")
                for model in available_models:
                    st.markdown(f"🔵 **{model}**")
            with col2:
                st.markdown("**Markers (Comparisons):**")
                st.markdown("● **Circle** - finetuned vs from_scratch")
                st.markdown("■ **Square** - finetuned vs optimized")
            
            # Filter difference data
            if selected_models and selected_comparisons and selected_metric and selected_task_diff:
                diff_plot_df = st.session_state.diff_df[
                    (st.session_state.diff_df['task'] == selected_task_diff) & 
                    (st.session_state.diff_df['metric_name'] == selected_metric)
                ].copy()
                
                if not diff_plot_df.empty and x_param_diff in diff_plot_df.columns:
                    # Create difference plot
                    fig_diff = create_difference_plot(diff_plot_df, x_param_diff, selected_models, selected_comparisons)
                    st.plotly_chart(fig_diff, use_container_width=True)
                    
                    # Summary of differences
                    st.subheader("Difference Summary")
                    filtered_summary_df = diff_plot_df[
                        (diff_plot_df['base_model'].isin(selected_models)) &
                        (diff_plot_df['comparison'].isin(selected_comparisons))
                    ]
                    if not filtered_summary_df.empty:
                        # Group by both base_model and comparison
                        summary_groups = filtered_summary_df.groupby(['base_model', 'comparison'])['difference']
                        diff_summary = summary_groups.agg(['mean', 'std', 'min', 'max']).round(4)
                        st.dataframe(diff_summary)
                    else:
                        st.warning("No data available for selected models and comparisons.")
                else:
                    st.warning("No difference data available for the selected filters.")
            else:
                if not selected_models:
                    st.info("Please select at least one model.")
                elif not selected_comparisons:
                    st.info("Please select at least one comparison type.")
                else:
                    st.info("Please ensure metric and task are selected.")
        else:
            st.warning("No difference data could be calculated. This could be because:")
            st.markdown("- No finetuned models found in the data")
            st.markdown("- No comparison models (from_scratch, gcn, sage, gat) found")
            st.markdown("- Data structure doesn't match expected format")
            
            # Debug info
            if not st.session_state.df.empty:
                st.subheader("Available Model Types in Data:")
                model_types = st.session_state.df['model_type'].unique()
                st.write(model_types)
                
                st.subheader("Sample of Main Data:")
                st.dataframe(st.session_state.df.head())
    
    with tab3:
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

if __name__ == "__main__":
    main()