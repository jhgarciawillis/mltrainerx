import os
import shutil
import logging
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

from logging.handlers import RotatingFileHandler
from _0config import CHART_HEIGHT, CHART_WIDTH, ALLOWED_EXTENSIONS, MAX_FILE_SIZE

def setup_directory(directory_path):
    """Ensures that the directory exists; if not, it creates it."""
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)

def setup_logging(log_file):
    """Sets up logging to both console and file with rotation."""
    logger = logging.getLogger('mltrainer')
    logger.setLevel(logging.INFO)
    
    file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=5)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def debug_print(*args):
    """Displays debugging information in Streamlit."""
    message = ' '.join(map(str, args))
    st.text(f"DEBUG: {message}")

def flatten_clustered_data(clustered_data):
    """Flatten the clustered data dictionary into a list of tuples."""
    flattened_data = []
    for cluster_name, cluster_dict in clustered_data.items():
        for label, indices in cluster_dict.items():
            flattened_data.append((cluster_name, label, indices))
    return flattened_data

def plot_feature_importance(feature_importance, title="Feature Importance"):
    """Plot feature importance using Plotly."""
    # Check if the dataframe has 'importance' column, otherwise use 'combined_score'
    if 'importance' in feature_importance.columns:
        x_column = 'importance'
    elif 'combined_score' in feature_importance.columns:
        x_column = 'combined_score'
    else:
        # If neither column exists, use the first numeric column
        numeric_cols = feature_importance.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            x_column = numeric_cols[0]
        else:
            raise ValueError("No suitable numeric column found for feature importance plot")
    
    fig = px.bar(feature_importance, x=x_column, y='feature', orientation='h',
                 title=title, height=CHART_HEIGHT, width=CHART_WIDTH)
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig)

def plot_prediction_vs_actual(y_true, y_pred, title="Prediction vs Actual"):
    """Plot prediction vs actual values using Plotly."""
    df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers'))
    fig.add_trace(go.Scatter(x=[y_true.min(), y_true.max()], 
                             y=[y_true.min(), y_true.max()], 
                             mode='lines', name='Ideal'))
    fig.update_layout(
        title=title,
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        height=CHART_HEIGHT, 
        width=CHART_WIDTH
    )
    st.plotly_chart(fig)

def plot_residuals(y_true, y_pred):
    """Plot regression residuals."""
    residuals = y_true - y_pred
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers'))
    fig.update_layout(
        title='Residual Plot',
        xaxis_title='Predicted Values',
        yaxis_title='Residuals',
        height=CHART_HEIGHT, 
        width=CHART_WIDTH
    )
    st.plotly_chart(fig)

def create_correlation_heatmap(df):
    """Create a correlation heatmap for numeric columns."""
    # Get only numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    # If no numeric columns, inform the user
    if numeric_df.empty:
        st.warning("No numeric columns available for correlation heatmap.")
        return
    
    # Compute correlation matrix
    corr = numeric_df.corr()
    
    # Create correlation heatmap
    fig = px.imshow(corr, 
                    color_continuous_scale='RdBu_r', 
                    zmin=-1, 
                    zmax=1,
                    title='Correlation Heatmap (Numeric Columns)',
                    height=CHART_HEIGHT, 
                    width=CHART_WIDTH)
    
    fig.update_layout(
        xaxis_title='Features',
        yaxis_title='Features'
    )
    
    st.plotly_chart(fig)

def display_data_statistics(df):
    """Display comprehensive statistics for both numeric and categorical columns."""
    # Numeric columns statistics
    numeric_columns = df.select_dtypes(include=['number'])
    if not numeric_columns.empty:
        st.subheader("Numeric Columns Statistics")
        numeric_stats = numeric_columns.describe()
        st.dataframe(numeric_stats)
    
    # Categorical columns statistics
    categorical_columns = df.select_dtypes(include=['object', 'category'])
    if not categorical_columns.empty:
        st.subheader("Categorical Columns Detailed Analysis")
        
        # Create a comprehensive analysis for each categorical column
        for col in categorical_columns.columns:
            st.write(f"\n### Analysis for Column: {col}")
            
            # Value counts
            value_counts = df[col].value_counts()
            st.write("#### Value Distribution")
            st.dataframe(value_counts)
            
            # Percentage distribution
            percentage_distribution = value_counts / len(df) * 100
            fig = px.pie(
                names=percentage_distribution.index, 
                values=percentage_distribution.values,
                title=f'Percentage Distribution - {col}'
            )
            st.plotly_chart(fig)
            
            # Additional statistics
            col_stats = {
                'Total Unique Values': df[col].nunique(),
                'Most Frequent Value': value_counts.index[0],
                'Frequency of Most Frequent': value_counts.iloc[0],
                'Percentage of Most Frequent': f"{percentage_distribution.iloc[0]:.2f}%",
                'Least Frequent Value': value_counts.index[-1],
                'Frequency of Least Frequent': value_counts.iloc[-1],
                'Percentage of Least Frequent': f"{percentage_distribution.iloc[-1]:.2f}%"
            }
            
            st.write("#### Categorical Column Statistics")
            stats_df = pd.DataFrame.from_dict(col_stats, orient='index', columns=['Value'])
            st.dataframe(stats_df)
    
    # Overall dataset information
    st.subheader("Dataset Overview")
    st.write(f"Total number of rows: {len(df)}")
    st.write(f"Total number of columns: {len(df.columns)}")
    
    # Column types
    st.write("### Column Types")
    column_types = df.dtypes
    st.dataframe(column_types)

def check_data_balance(df, target_column):
    """Check and display the balance of classes in the target column."""
    # Add a safety check
    if target_column not in df.columns:
        st.warning(f"Target column '{target_column}' not found in the dataset.")
        st.write("Available columns:", list(df.columns))
        return

    if df[target_column].dtype == 'object' or df[target_column].dtype.name == 'category':
        class_counts = df[target_column].value_counts()
        fig = px.bar(x=class_counts.index, y=class_counts.values, title='Class Distribution')
        st.plotly_chart(fig)
    else:
        st.write("Target variable is continuous. Showing distribution:")
        fig = px.histogram(df, x=target_column, title='Target Variable Distribution')
        st.plotly_chart(fig)

def identify_high_correlation_features(df, threshold=0.9):
    """Identify and return pairs of features with high correlation."""
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [(upper.index[i], upper.columns[j]) 
                       for i, j in zip(*np.where(upper > threshold))]
    return high_corr_pairs