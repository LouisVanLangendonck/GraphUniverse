import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import json
from glob import glob
import numpy as np

# Page config
st.set_page_config(
    page_title="Transductive Experiment Results Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Transductive Experiment Results Analyzer")
st.markdown("---")

# ----------------------
# Transductive Data Loader (all_results style)
# ----------------------

def load_transductive_all_results(base_dir):
    """Load all result.json files from multi_transductive_results/community/run_*/ and build all_results list."""
    run_files = sorted(glob(os.path.join(base_dir, "run_*/result.json")))
    all_results = []
    for result_path in run_files:
        try:
            with open(result_path, 'r') as f:
                data = json.load(f)
            result = {}
            # Parameters (x-params)
            result['parameters'] = data.get('parameters', {})
            result['repeat'] = data.get('repeat', None)
            result['run_time'] = data.get('run_time', None)
            result['run_dir'] = os.path.dirname(result_path)
            # Model metrics (community)
            result['community'] = data.get('community', {})
            # For each model, store optimal_hyperparams if present
            result['optimal_hyperparams'] = {}
            for model, model_data in result['community'].items():
                if 'optimal_hyperparams' in model_data:
                    result['optimal_hyperparams'][model] = model_data['optimal_hyperparams']
            all_results.append(result)
        except Exception as e:
            st.warning(f"Failed to load {result_path}: {e}")
    return all_results

# ----------------------
# Data Processor: builds DataFrame for dashboard/analysis
# ----------------------

class DataProcessor:
    @staticmethod
    def create_dataframe_from_all_results(all_results):
        """Convert all_results list to a DataFrame with x-params, model metrics, and optimal hyperparams."""
        rows = []
        for run in all_results:
            row = {}
            # X-params
            for k, v in run.get('parameters', {}).items():
                row[f"param_{k}"] = v
            row['repeat'] = run.get('repeat', None)
            row['run_time'] = run.get('run_time', None)
            row['run_dir'] = run.get('run_dir', None)
            # Model metrics
            community = run.get('community', {})
            for model, model_data in community.items():
                test_metrics = model_data.get('test_metrics', {})
                for metric, metric_data in test_metrics.items():
                    if isinstance(metric_data, dict):
                        row[f"model.{model}.{metric}.mean"] = metric_data.get('mean', np.nan)
                        row[f"model.{model}.{metric}.std"] = metric_data.get('std', np.nan)
            # Optimal hyperparams
            for model, hps in run.get('optimal_hyperparams', {}).items():
                for hp_name, hp_val in hps.items():
                    row[f"hp.{model}.{hp_name}"] = hp_val
            rows.append(row)
        return pd.DataFrame(rows)

# --- Plotting and Analysis Classes (adapted from final_dashboarding.py) ---
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats

class RankingAnalyzer:
    @staticmethod
    def calculate_ranking_with_uncertainty(df, selected_models, selected_metric, x_axis, 
                                         second_param='None', n_samples=100, seed=42):
        np.random.seed(seed)
        ranking_data = []
        for idx, row in df.iterrows():
            model_performances = {}
            model_uncertainties = {}
            for model in selected_models:
                mean_col = f"model.{model}.{selected_metric}.mean"
                std_col = f"model.{model}.{selected_metric}.std"
                if mean_col in row and not pd.isna(row[mean_col]):
                    model_performances[model] = row[mean_col]
                    if std_col in row and not pd.isna(row[std_col]):
                        model_uncertainties[model] = row[std_col]
                    else:
                        model_uncertainties[model] = 0.01
            if len(model_performances) > 1:
                performance_values = list(model_performances.values())
                if np.std(performance_values) < 1e-10:
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
                    rankings = []
                    for _ in range(n_samples):
                        sampled_performances = {}
                        for model, mean in model_performances.items():
                            std = model_uncertainties.get(model, 0.01)
                            sampled_performances[model] = np.random.normal(mean, std)
                        sorted_models = sorted(sampled_performances.items(), key=lambda x: x[1], reverse=True)
                        ranking = {model: rank + 1 for rank, (model, _) in enumerate(sorted_models)}
                        rankings.append(ranking)
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
    @staticmethod
    def propagate_uncertainty_linear_regression(x, y_mean, y_std, x_pred=None):
        n_bootstrap = 1000
        if x_pred is None:
            x_pred = np.linspace(x.min(), x.max(), 100)
        valid_mask = ~(np.isnan(x) | np.isnan(y_mean) | np.isnan(y_std))
        x_clean = x[valid_mask]
        y_mean_clean = y_mean[valid_mask]
        y_std_clean = y_std[valid_mask]
        if len(x_clean) < 3 or np.std(x_clean) == 0 or np.std(y_mean_clean) == 0:
            return None
        bootstrap_predictions = []
        for _ in range(n_bootstrap):
            boot_indices = np.random.choice(len(x_clean), size=len(x_clean), replace=True)
            x_boot = x_clean[boot_indices]
            y_mean_boot = y_mean_clean[boot_indices]
            y_std_boot = y_std_clean[boot_indices]
            y_boot = np.random.normal(y_mean_boot, y_std_boot)
            try:
                reg = LinearRegression().fit(x_boot.reshape(-1, 1), y_boot)
                pred_boot = reg.predict(x_pred.reshape(-1, 1))
                bootstrap_predictions.append(pred_boot)
            except:
                continue
        if len(bootstrap_predictions) < 50:
            return None
        bootstrap_predictions = np.array(bootstrap_predictions)
        mean_line_boot = np.mean(bootstrap_predictions, axis=0)
        pred_lower_boot = np.percentile(bootstrap_predictions, 2.5, axis=0)
        pred_upper_boot = np.percentile(bootstrap_predictions, 97.5, axis=0)
        return {
            'x_pred': x_pred,
            'mean_line': mean_line_boot,
            'pred_lower': pred_lower_boot,
            'pred_upper': pred_upper_boot
        }

    @staticmethod
    def propagate_uncertainty_polynomial_regression(x, y_mean, y_std, degree=2, x_pred=None):
        n_bootstrap = 1000
        if x_pred is None:
            x_pred = np.linspace(x.min(), x.max(), 100)
        valid_mask = ~(np.isnan(x) | np.isnan(y_mean) | np.isnan(y_std))
        x_clean = x[valid_mask]
        y_mean_clean = y_mean[valid_mask]
        y_std_clean = y_std[valid_mask]
        if len(x_clean) < degree + 2:
            return None
        bootstrap_predictions = []
        for _ in range(n_bootstrap):
            boot_indices = np.random.choice(len(x_clean), size=len(x_clean), replace=True)
            x_boot = x_clean[boot_indices]
            y_mean_boot = y_mean_clean[boot_indices]
            y_std_boot = y_std_clean[boot_indices]
            y_boot = np.random.normal(y_mean_boot, y_std_boot)
            try:
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                x_poly = poly.fit_transform(x_boot.reshape(-1, 1))
                reg = LinearRegression().fit(x_poly, y_boot)
                x_pred_poly = poly.transform(x_pred.reshape(-1, 1))
                pred_boot = reg.predict(x_pred_poly)
                bootstrap_predictions.append(pred_boot)
            except:
                continue
        if len(bootstrap_predictions) < 50:
            return None
        bootstrap_predictions = np.array(bootstrap_predictions)
        mean_line_boot = np.mean(bootstrap_predictions, axis=0)
        pred_lower_boot = np.percentile(bootstrap_predictions, 2.5, axis=0)
        pred_upper_boot = np.percentile(bootstrap_predictions, 97.5, axis=0)
        return {
            'x_pred': x_pred,
            'mean_line': mean_line_boot,
            'pred_lower': pred_lower_boot,
            'pred_upper': pred_upper_boot
        }

class PlotGenerator:
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    def create_scatter_plot(self, plot_df, x_axis, selected_models, selected_metric, second_param=None):
        fig = go.Figure()
        for i, model in enumerate(selected_models):
            model_metric_col = f"model.{model}.{selected_metric}.mean"
            model_error_col = f"model.{model}.{selected_metric}.std"
            if model_metric_col in plot_df.columns:
                model_data = plot_df.dropna(subset=[model_metric_col])
                if len(model_data) > 0:
                    error_y_data = model_data.get(model_error_col)
                    color = self.colors[i % len(self.colors)]
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
        fig.update_layout(xaxis_title=x_axis, yaxis_title=selected_metric)
        return fig

    def create_line_fit_plot(self, plot_df, x_axis, selected_models, selected_metric, show_only_lines=False, fit_type='Linear', poly_degree=2):
        fig = go.Figure()
        for i, model in enumerate(selected_models):
            model_metric_col = f"model.{model}.{selected_metric}.mean"
            model_error_col = f"model.{model}.{selected_metric}.std"
            model_data = plot_df.dropna(subset=[x_axis, model_metric_col])
            color = self.colors[i % len(self.colors)]
            if len(model_data) >= (3 if fit_type == 'Linear' else poly_degree + 2):
                x = model_data[x_axis].values
                y = model_data[model_metric_col].values
                yerr = model_data[model_error_col].values if model_error_col in model_data.columns else np.ones_like(y) * 0.01
                if fit_type == 'Linear':
                    fit_result = StatisticalAnalyzer.propagate_uncertainty_linear_regression(x, y, yerr)
                else:
                    fit_result = StatisticalAnalyzer.propagate_uncertainty_polynomial_regression(x, y, yerr, degree=poly_degree)
                if not show_only_lines:
                    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=model, marker=dict(color=color)))
                if fit_result is not None:
                    fig.add_trace(go.Scatter(x=fit_result['x_pred'], y=fit_result['mean_line'], mode='lines', name=f'{model} {fit_type} Fit', line=dict(color=color, dash='dash')))
                    fig.add_trace(go.Scatter(x=np.concatenate([fit_result['x_pred'], fit_result['x_pred'][::-1]]), y=np.concatenate([fit_result['pred_upper'], fit_result['pred_lower'][::-1]]), fill='toself', fillcolor=f'rgba{tuple(int(color[1+i*2:3+i*2],16) for i in range(3)) + (0.1,)}' if color.startswith('#') else 'rgba(255,0,0,0.1)', line=dict(color='rgba(0,0,0,0)'), name=f'{model} 95% CI', showlegend=False, hoverinfo='skip'))
        fig.update_layout(xaxis_title=x_axis, yaxis_title=selected_metric)
        return fig

    def create_grouped_bar_plot(self, plot_df, x_axis, selected_models, selected_metric):
        fig = go.Figure()
        unique_x_vals = sorted(plot_df[x_axis].dropna().unique())
        n_models = len(selected_models)
        bar_width = 0.8 / n_models if n_models > 0 else 0.8
        x_positions = np.arange(len(unique_x_vals))
        for j, model in enumerate(selected_models):
            model_metric_col = f"model.{model}.{selected_metric}.mean"
            model_error_col = f"model.{model}.{selected_metric}.std"
            model_data = plot_df.dropna(subset=[x_axis, model_metric_col])
            color = self.colors[j % len(self.colors)]
            y_values = []
            std_values = []
            for x_val in unique_x_vals:
                vals = model_data[model_data[x_axis] == x_val][model_metric_col]
                y_values.append(vals.mean() if not vals.empty else None)
                stds = model_data[model_data[x_axis] == x_val][model_error_col] if model_error_col in model_data.columns else None
                std_values.append(stds.mean() if stds is not None and not stds.empty else 0)
            bar_positions = x_positions + (j - (n_models-1)/2) * bar_width
            fig.add_trace(go.Bar(
                x=bar_positions,
                y=y_values,
                name=model,
                marker_color=color,
                error_y=dict(type='data', array=std_values, visible=True, thickness=1, width=3),
                hovertemplate=f'<b>{model}</b><br>{x_axis}: %{{x}}<br>Mean: %{{y}}<br>Std: %{{error_y.array}}<br><extra></extra>'
            ))
        fig.update_xaxes(
            ticktext=[str(x) for x in unique_x_vals],
            tickvals=x_positions,
            title=x_axis
        )
        fig.update_yaxes(title=selected_metric)
        return fig

# --- Hyperparameter Analysis Classes (adapted from hyperparameter_analysis.py) ---
class HyperparameterAnalyzer:
    @staticmethod
    def get_models_with_hyperopt(df):
        models = set()
        for col in df.columns:
            if col.startswith('hp.'):
                parts = col.split('.')
                if len(parts) >= 3:
                    models.add(parts[1])
        return sorted(models)
    @staticmethod
    def get_hyperparameters_for_model(df, selected_model):
        hps = set()
        for col in df.columns:
            if col.startswith(f'hp.{selected_model}.'):
                hps.add(col.split('.', 2)[2])
        return sorted(hps)
    @staticmethod
    def collect_hyperparameter_data(df, selected_model, selected_hyperparam, selected_metric):
        hp_col = f"hp.{selected_model}.{selected_hyperparam}"
        metric_col = f"model.{selected_model}.{selected_metric}.mean"
        if hp_col not in df.columns or metric_col not in df.columns:
            return [], [], []
        hp_values = df[hp_col].tolist()
        perf_values = df[metric_col].tolist()
        indices = df.index.tolist()
        return hp_values, perf_values, indices
    @staticmethod
    def create_hyperparameter_distribution_plot(hyperparam_values, selected_hyperparam):
        def to_str_or_none(v):
            if pd.isna(v) or v is None or v == "None":
                return "None"
            return str(v)
        values = [to_str_or_none(v) for v in hyperparam_values]
        value_counts = pd.Series(values, dtype='object').value_counts(dropna=False, sort=False)
        fig = px.bar(
            x=value_counts.index.astype(str),
            y=value_counts.values,
            title=f"Distribution of {selected_hyperparam}",
            labels={'x': selected_hyperparam, 'y': 'Count'}
        )
        fig.update_xaxes(title=selected_hyperparam)
        fig.update_yaxes(title='Count')
        return fig
    @staticmethod
    def create_performance_vs_hyperparameter_plot(hyperparam_values, performance_values, selected_hyperparam, selected_metric):
        is_numerical = all(isinstance(x, (int, float, np.floating, np.integer)) for x in hyperparam_values)
        if is_numerical:
            fig = px.scatter(x=hyperparam_values, y=performance_values, title=f"{selected_metric} vs {selected_hyperparam}", labels={'x': selected_hyperparam, 'y': selected_metric})
        else:
            df_plot = pd.DataFrame({selected_hyperparam: hyperparam_values, selected_metric: performance_values})
            fig = px.box(df_plot, x=selected_hyperparam, y=selected_metric, title=f"{selected_metric} by {selected_hyperparam}")
        fig.update_xaxes(title=selected_hyperparam)
        fig.update_yaxes(title=selected_metric)
        return fig
    @staticmethod
    def get_common_hyperparameters_across_models(df):
        # Find hyperparameters that are present for all models
        model_hp_sets = []
        models = set()
        for col in df.columns:
            if col.startswith('hp.'):
                parts = col.split('.')
                if len(parts) >= 3:
                    models.add(parts[1])
        for model in models:
            hps = set()
            for col in df.columns:
                if col.startswith(f'hp.{model}.'):
                    hps.add(col.split('.', 2)[2])
            model_hp_sets.append(hps)
        if not model_hp_sets:
            return set()
        return set.intersection(*model_hp_sets)
    @staticmethod
    def collect_cross_model_hyperparameter_data(df, selected_hyperparam):
        # Collect values for the selected hyperparam across all models
        values = []
        for col in df.columns:
            if col.startswith('hp.') and col.endswith(f'.{selected_hyperparam}'):
                values.extend(df[col].tolist())
        return values

class HyperparameterDashboard:
    def __init__(self):
        self.analyzer = HyperparameterAnalyzer()
    def run(self, df):
        st.header("🔧 Hyperparameter Analysis")
        models = self.analyzer.get_models_with_hyperopt(df)
        if not models:
            st.warning("No hyperparameter optimization results found in the data.")
            return
        selected_model = st.selectbox("Model:", models)
        hps = self.analyzer.get_hyperparameters_for_model(df, selected_model)
        if not hps:
            st.warning(f"No hyperparameters found for model {selected_model}.")
            return
        selected_hyperparam = st.selectbox("Hyperparameter:", hps)
        available_metrics = sorted({c.split('.')[2] for c in df.columns if c.startswith(f'model.{selected_model}.') and c.endswith('.mean')})
        if not available_metrics:
            st.warning(f"No metrics found for model {selected_model}.")
            return
        selected_metric = st.selectbox("Performance Metric:", available_metrics)
        hp_values, perf_values, indices = self.analyzer.collect_hyperparameter_data(df, selected_model, selected_hyperparam, selected_metric)
        if not hp_values:
            st.warning("No data for this hyperparameter/metric combination.")
            return
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Hyperparameter Distribution")
            fig = self.analyzer.create_hyperparameter_distribution_plot(hp_values, selected_hyperparam)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader(f"Performance vs Hyperparameter")
            fig = self.analyzer.create_performance_vs_hyperparameter_plot(hp_values, perf_values, selected_hyperparam, selected_metric)
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Data Table")
        st.dataframe(pd.DataFrame({selected_hyperparam: hp_values, selected_metric: perf_values}), use_container_width=True)

        # Cross-model hyperparameter analysis
        st.markdown("---")
        st.header("🔍 Cross-Model Hyperparameter Analysis")
        common_hps = self.analyzer.get_common_hyperparameters_across_models(df)
        if common_hps:
            selected_common_hp = st.selectbox("Common Hyperparameter (across all models):", sorted(common_hps), key='cross_model_hp')
            cross_values = self.analyzer.collect_cross_model_hyperparameter_data(df, selected_common_hp)
            st.subheader(f"Distribution of {selected_common_hp} (All Models)")
            fig = self.analyzer.create_hyperparameter_distribution_plot(cross_values, selected_common_hp)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hyperparameters are present for all models.")

# --- Main Dashboard UI ---
class Dashboard:
    def __init__(self):
        self.plot_generator = PlotGenerator()
    def run(self, df):
        st.header("📊 Main Dashboard: Transductive Results")
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        param_columns = [col for col in df.columns if col.startswith('param_')]
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("X-axis Parameter")
            x_axis = st.selectbox("Select X-axis parameter:", options=numeric_columns, index=0 if numeric_columns else None, key='x_axis')
        with col2:
            st.subheader("Y-axis Metric")
            available_metrics = sorted({c.split('.')[2] for c in df.columns if c.startswith('model.') and c.endswith('.mean')})
            selected_metric = st.selectbox("Select metric:", options=available_metrics, index=0 if available_metrics else None, key='metric')
        st.subheader("Model Selection")
        available_models = sorted({c.split('.')[1] for c in df.columns if c.startswith('model.') and c.endswith('.mean') and f'.{selected_metric}.mean' in c})
        selected_models = st.multiselect("Select models to compare:", options=available_models, default=available_models[:3] if len(available_models) >= 3 else available_models, key='models')
        st.subheader("Second Parameter (Optional)")
        second_param = st.selectbox("Select second parameter for subplots/dot size:", options=['None'] + param_columns, index=0, key='second_param')
        plot_style = st.radio("Plot Style", ["Scatter", "Line Fit", "Grouped Bar"], horizontal=True, key='plot_style')
        show_only_lines = False
        fit_type = 'Linear'
        poly_degree = 2
        if plot_style == "Line Fit":
            show_only_lines = st.checkbox("Show Only Fitted Lines", value=False, key='show_only_lines')
            fit_type = st.selectbox("Fit Type", ["Linear", "Polynomial"], key='fit_type')
            if fit_type == "Polynomial":
                poly_degree = st.number_input("Polynomial Degree", min_value=2, max_value=5, value=2, key='poly_degree')
        if x_axis and selected_metric and selected_models:
            if plot_style == "Scatter":
                fig = self.plot_generator.create_scatter_plot(df, x_axis, selected_models, selected_metric, second_param if second_param != 'None' else None)
            elif plot_style == "Line Fit":
                fig = self.plot_generator.create_line_fit_plot(df, x_axis, selected_models, selected_metric, show_only_lines=show_only_lines, fit_type=fit_type, poly_degree=poly_degree)
            elif plot_style == "Grouped Bar":
                fig = self.plot_generator.create_grouped_bar_plot(df, x_axis, selected_models, selected_metric)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", csv, file_name="transductive_results.csv", mime="text/csv")

# --- Main App ---
def main():
    st.sidebar.header("Transductive Results Directory")
    base_dir = st.sidebar.text_input("Results directory", value="multi_transductive_results/community")
    page = st.sidebar.radio("Page", ["Main Dashboard", "Hyperparameter Analysis"])
    if st.sidebar.button("Load Results"):
        with st.spinner("Loading results..."):
            all_results = load_transductive_all_results(base_dir)
            if not all_results:
                st.error("No results found.")
                return
            st.session_state.all_results = all_results
            st.session_state.df = DataProcessor.create_dataframe_from_all_results(all_results)
            st.success(f"Loaded {len(all_results)} runs.")
    if 'df' in st.session_state:
        if page == "Main Dashboard":
            dashboard = Dashboard()
            dashboard.run(st.session_state.df)
        elif page == "Hyperparameter Analysis":
            hyper_dash = HyperparameterDashboard()
            hyper_dash.run(st.session_state.df)

if __name__ == "__main__":
    main() 