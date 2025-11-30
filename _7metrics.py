import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    explained_variance_score, max_error, median_absolute_error
)
from _0config import config, TOOLTIPS, INFO_TEXTS, CHART_HEIGHT, CHART_WIDTH
from _2misc_utils import debug_print, plot_feature_importance
from _2ui_utils import create_tooltip, create_info_button

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive regression metrics."""
    metrics = {
        'Mean Absolute Error': mean_absolute_error(y_true, y_pred),
        'Mean Squared Error': mean_squared_error(y_true, y_pred),
        'Root Mean Squared Error': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R-squared Value': r2_score(y_true, y_pred),
        'Explained Variance Score': explained_variance_score(y_true, y_pred),
        'Max Error': max_error(y_true, y_pred),
        'Median Absolute Error': median_absolute_error(y_true, y_pred)
    }
    
    # Calculate Mean Absolute Percentage Error (MAPE) and Mean Bias Deviation (MBD)
    # Avoid division by zero
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        mbd = np.mean((y_pred[mask] - y_true[mask]) / y_true[mask]) * 100
    else:
        mape = np.nan
        mbd = np.nan
    
    metrics['Mean Absolute Percentage Error'] = mape
    metrics['Mean Bias Deviation'] = mbd
    
    return metrics

def display_metrics(metrics, title="Model Metrics"):
    """Display metrics in Streamlit."""
    st.subheader(title)
    create_tooltip(TOOLTIPS.get("model_metrics", "Model evaluation metrics"))
    create_info_button("model_metrics")
    
    # Display metrics in columns for better layout
    cols = st.columns(3)
    for i, (metric_name, metric_value) in enumerate(metrics.items()):
        with cols[i % 3]:
            if np.isnan(metric_value):
                st.metric(label=metric_name, value="N/A")
            else:
                st.metric(label=metric_name, value=f"{metric_value:.4f}")

def plot_residuals(y_true, y_pred, title="Residual Analysis"):
    """Plot regression residuals with multiple visualization options."""
    residuals = y_true - y_pred
    
    plot_type = st.selectbox("Select residuals plot type", 
                            ["Scatter", "Histogram", "Box Plot"],
                            key=f"residuals_plot_type_{hash(str(y_true[:5]))}")
    
    fig = go.Figure()
    
    if plot_type == "Scatter":
        fig.add_trace(go.Scatter(
            x=y_pred, 
            y=residuals, 
            mode='markers',
            marker=dict(
                size=8,
                opacity=0.6,
                color='blue'
            )
        ))
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(
            title=title,
            xaxis_title='Predicted Values',
            yaxis_title='Residuals'
        )
    elif plot_type == "Histogram":
        fig.add_trace(go.Histogram(
            x=residuals,
            nbinsx=30,
            name='Residuals',
            opacity=0.75
        ))
        fig.update_layout(
            title=f'{title} - Distribution',
            xaxis_title='Residuals',
            yaxis_title='Frequency'
        )
    elif plot_type == "Box Plot":
        fig.add_trace(go.Box(
            y=residuals,
            name='Residuals',
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))
        fig.update_layout(
            title=f'{title} - Box Plot',
            yaxis_title='Residuals'
        )
    
    fig.update_layout(
        height=CHART_HEIGHT,
        width=CHART_WIDTH
    )
    st.plotly_chart(fig)

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted"):
    """Plot prediction vs actual values with multiple visualization options."""
    st.subheader(title)
    
    plot_type = st.selectbox("Select plot type", 
                            ["Scatter", "Hexbin", "2D Density"],
                            key=f"actual_vs_pred_plot_type_{hash(str(y_true[:5]))}")
    
    fig = go.Figure()
    
    if plot_type == "Scatter":
        fig.add_trace(go.Scatter(
            x=y_true, 
            y=y_pred, 
            mode='markers',
            name='Predictions',
            marker=dict(
                size=8,
                opacity=0.6,
                color='blue'
            )
        ))
    elif plot_type == "Hexbin":
        fig.add_trace(go.Histogram2d(
            x=y_true, 
            y=y_pred,
            colorscale='Viridis',
            nbinsx=50,
            nbinsy=50,
            name='Density'
        ))
    elif plot_type == "2D Density":
        try:
            xy = np.vstack([y_true, y_pred])
            z = gaussian_kde(xy)(xy)
            
            fig.add_trace(go.Scatter(
                x=y_true,
                y=y_pred,
                mode='markers',
                marker=dict(
                    color=z,
                    colorscale='Viridis',
                    size=8,
                    opacity=0.6,
                    showscale=True,
                    colorbar=dict(title='Density')
                ),
                name='Density'
            ))
        except Exception as e:
            st.warning(f"Could not create density plot: {str(e)}. Falling back to scatter.")
            fig.add_trace(go.Scatter(
                x=y_true, 
                y=y_pred, 
                mode='markers',
                name='Predictions'
            ))
    
    # Add ideal line
    fig.add_trace(go.Scatter(
        x=[y_true.min(), y_true.max()], 
        y=[y_true.min(), y_true.max()], 
        mode='lines', 
        name='Ideal',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        height=CHART_HEIGHT,
        width=CHART_WIDTH,
        showlegend=True,
        template='plotly_white'
    )
    
    st.plotly_chart(fig)
    
    # Add additional statistics
    error = y_pred - y_true
    st.write("Prediction Statistics:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Error", f"{error.mean():.4f}")
    with col2:
        st.metric("Error Std Dev", f"{error.std():.4f}")
    with col3:
        st.metric("Max Absolute Error", f"{abs(error).max():.4f}")

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and display various metrics and plots."""
    st.subheader("Model Evaluation")
    create_info_button("model_evaluation")
    
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    display_metrics(metrics)
    
    plot_actual_vs_predicted(y_test, y_pred)
    plot_residuals(y_test, y_pred)

def calculate_feature_correlations(X, y):
    """Calculate correlations between features and target variable."""
    correlations = X.apply(lambda x: x.corr(y) if x.dtype in ['int64', 'float64'] else 0)
    return correlations.sort_values(ascending=False)

def display_feature_correlations(correlations):
    """Display feature correlations in Streamlit."""
    st.subheader("Feature Correlations with Target")
    create_tooltip(TOOLTIPS.get("feature_correlations", "Correlation between features and target"))
    fig = go.Figure(go.Bar(
        x=correlations.values,
        y=correlations.index,
        orientation='h'
    ))
    fig.update_layout(
        title='Feature Correlations with Target Variable',
        xaxis_title='Correlation',
        yaxis_title='Features',
        height=CHART_HEIGHT,
        width=CHART_WIDTH
    )
    st.plotly_chart(fig)

def evaluate_predictions(y_true, y_pred, cluster_name=None):
    """Evaluate predictions and display results."""
    metrics = calculate_metrics(y_true, y_pred)
    
    title = "Prediction Metrics"
    if cluster_name:
        title += f" for Cluster: {cluster_name}"
    
    display_metrics(metrics, title)
    plot_actual_vs_predicted(y_true, y_pred)
    plot_residuals(y_true, y_pred)

def compare_models(models_metrics):
    """Compare multiple models based on their metrics."""
    st.subheader("Model Comparison")
    create_info_button("model_comparison")
    
    comparison_df = pd.DataFrame(models_metrics).T
    st.dataframe(comparison_df)
    
    # Plot comparison for each metric
    for metric in comparison_df.columns:
        fig = go.Figure(go.Bar(
            x=comparison_df.index,
            y=comparison_df[metric],
            text=comparison_df[metric].round(4),
            textposition='auto'
        ))
        fig.update_layout(
            title=f'Comparison of {metric}',
            xaxis_title='Models',
            yaxis_title=metric,
            height=CHART_HEIGHT,
            width=CHART_WIDTH
        )
        st.plotly_chart(fig)

def calculate_cluster_metrics(clustered_predictions, y_true):
    """Calculate metrics for each cluster."""
    cluster_metrics = {}
    for cluster, predictions in clustered_predictions.items():
        cluster_metrics[cluster] = calculate_metrics(y_true[predictions.index], predictions)
    return cluster_metrics

def display_cluster_metrics(cluster_metrics):
    """Display metrics for each cluster."""
    st.subheader("Cluster-wise Metrics")
    create_info_button("cluster_metrics")
    for cluster, metrics in cluster_metrics.items():
        st.write(f"Cluster: {cluster}")
        display_metrics(metrics)
        st.write("---")

def plot_prediction_distribution(y_true, y_pred, title="Prediction Distribution"):
    """Plot the distribution of predictions compared to actual values."""
    try:
        fig = go.Figure()
        
        # Histogram for actual values
        fig.add_trace(go.Histogram(
            x=y_true, 
            name='Actual Values', 
            opacity=0.75,
            marker_color='blue'
        ))
        
        # Histogram for predicted values
        fig.add_trace(go.Histogram(
            x=y_pred, 
            name='Predicted Values', 
            opacity=0.75,
            marker_color='red'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Values',
            yaxis_title='Frequency',
            barmode='overlay',
            height=CHART_HEIGHT,
            width=CHART_WIDTH
        )
        
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error in plotting prediction distribution: {str(e)}")

def plot_error_distribution(y_true, y_pred, title="Prediction Error Distribution"):
    """Plot the distribution of prediction errors."""
    try:
        errors = y_true - y_pred
        
        fig = go.Figure()
        
        # Histogram for errors
        fig.add_trace(go.Histogram(
            x=errors, 
            name='Prediction Errors', 
            opacity=0.75,
            marker_color='green',
            nbinsx=30
        ))
        
        # Add vertical line at zero
        fig.add_vline(x=0, line_dash="dash", line_color="red", 
                     annotation_text="Zero Error", annotation_position="top")
        
        fig.update_layout(
            title=title,
            xaxis_title='Prediction Errors (Actual - Predicted)',
            yaxis_title='Frequency',
            height=CHART_HEIGHT,
            width=CHART_WIDTH
        )
        
        st.plotly_chart(fig)
        
        # Add error statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Error", f"{errors.mean():.4f}")
        with col2:
            st.metric("Std Dev", f"{errors.std():.4f}")
        with col3:
            st.metric("Median Error", f"{np.median(errors):.4f}")
            
    except Exception as e:
        st.error(f"Error in plotting error distribution: {str(e)}")