import pandas as pd
import streamlit as st
import os
import plotly.graph_objects as go

from _0config import (STREAMLIT_THEME, AVAILABLE_CLUSTERING_METHODS, DBSCAN_PARAMETERS, KMEANS_PARAMETERS, 
                      MODEL_CLASSES, config, STREAMLIT_APP_NAME, CHART_WIDTH, CHART_HEIGHT, TOOLTIPS, INFO_TEXTS)

def truncate_sheet_name(sheet_name, max_length=31):
    """Truncates Excel sheet names to the maximum length allowed by Excel."""
    return sheet_name[:max_length]

def validate_file_upload(uploaded_file):
    """Validate the uploaded file."""
    if uploaded_file is None:
        st.error("Please upload a file.")
        return False
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = ['csv', 'xlsx']
    MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB
    
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    if file_extension.lstrip('.') not in ALLOWED_EXTENSIONS:
        st.error(f"Invalid file format. Please upload a file with one of these extensions: {', '.join(ALLOWED_EXTENSIONS)}")
        return False
    
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"File size exceeds the maximum limit of {MAX_FILE_SIZE / (1024 * 1024):.2f} MB.")
        return False
    
    return True

def set_streamlit_theme():
    """Set the Streamlit theme based on the configuration."""
    st.set_page_config(
        page_title=STREAMLIT_APP_NAME,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Apply custom theme
    st.markdown(f"""
        <style>
            .reportview-container .main .block-container{{
                max-width: {CHART_WIDTH}px;
                padding-top: 5rem;
                padding-right: 1rem;
                padding-left: 1rem;
                padding-bottom: 5rem;
            }}
            .reportview-container .main {{
                color: {STREAMLIT_THEME['textColor']};
                background-color: {STREAMLIT_THEME['backgroundColor']};
            }}
            .sidebar .sidebar-content {{
                background-color: {STREAMLIT_THEME['secondaryBackgroundColor']};
            }}
            .Widget>label {{
                color: {STREAMLIT_THEME['textColor']};
            }}
            .stButton>button {{
                color: {STREAMLIT_THEME['backgroundColor']};
                background-color: {STREAMLIT_THEME['primaryColor']};
                border-radius: 0.3rem;
            }}
        </style>
        """, unsafe_allow_html=True)

def create_tooltip(text):
    """Create a tooltip with the given text."""
    st.markdown(f"""
    <style>
    .tooltip {{
      position: relative;
      display: inline-block;
      border-bottom: 1px dotted black;
    }}
    .tooltip .tooltiptext {{
      visibility: hidden;
      width: 120px;
      background-color: black;
      color: #fff;
      text-align: center;
      border-radius: 6px;
      padding: 5px 0;
      position: absolute;
      z-index: 1;
      bottom: 125%;
      left: 50%;
      margin-left: -60px;
      opacity: 0;
      transition: opacity 0.3s;
    }}
    .tooltip:hover .tooltiptext {{
      visibility: visible;
      opacity: 1;
    }}
    </style>
    <div class="tooltip">ℹ️
      <span class="tooltiptext">{text}</span>
    </div>
    """, unsafe_allow_html=True)

def create_info_button(key, unique_key=None):
    """
    Create an info button that displays detailed information when clicked.
    
    Args:
    - key: The key for retrieving info text
    - unique_key: Optional unique key to prevent widget ID conflicts
    """
    # If no unique_key is provided, generate one
    if unique_key is None:
        unique_key = f"info_button_{key}_{hash(key)}"
    
    if st.button(f"ℹ️ More Info: {key.replace('_', ' ').title()}", key=unique_key):
        st.info(INFO_TEXTS.get(key, "No additional information available."))

def display_metrics(metrics):
    """Display metrics in a formatted way."""
    st.subheader("Model Performance Metrics")
    col1, col2 = st.columns(2)
    for i, (metric, value) in enumerate(metrics.items()):
        if i % 2 == 0:
            col1.metric(metric, f"{value:.4f}")
        else:
            col2.metric(metric, f"{value:.4f}")

def display_config_manager(config):
    """Display configuration management interface."""
    st.sidebar.subheader("⚙️ Configuration Manager")
    
    # Show current config status
    config_status = {
        "Target Column": config.target_column or "Not set",
        "Numerical Features": len(config.numerical_columns),
        "Categorical Features": len(config.categorical_columns),
        "Models Selected": len(config.models_to_use),
        "Clustering Enabled": config.use_clustering,
    }
    
    with st.sidebar.expander("Current Configuration", expanded=False):
        for key, value in config_status.items():
            st.text(f"{key}: {value}")
    
    # Configuration presets
    st.sidebar.subheader("Quick Presets")
    
    preset = st.sidebar.selectbox(
        "Load preset configuration",
        ["Custom", "Quick Training", "High Accuracy", "Fast Prototype", "Production Ready"]
    )
    
    if preset == "Quick Training":
        config.models_to_use = ['rf']
        config.tuning_method = 'None'
        config.use_polynomial_features = False
        config.use_interaction_terms = False
        config.use_clustering = False
        st.sidebar.success("✅ Quick training preset loaded")
        
    elif preset == "High Accuracy":
        config.models_to_use = ['rf', 'xgb', 'lgbm', 'catboost']
        config.tuning_method = 'GridSearchCV'
        config.use_polynomial_features = True
        config.use_interaction_terms = True
        config.use_clustering = True
        st.sidebar.success("✅ High accuracy preset loaded")
        
    elif preset == "Fast Prototype":
        config.models_to_use = ['rf', 'knn']
        config.tuning_method = 'RandomizedSearchCV'
        config.use_polynomial_features = False
        config.use_interaction_terms = False
        config.use_clustering = False
        st.sidebar.success("✅ Fast prototype preset loaded")
        
    elif preset == "Production Ready":
        config.models_to_use = ['xgb', 'lgbm', 'catboost']
        config.tuning_method = 'RandomizedSearchCV'
        config.use_polynomial_features = True
        config.use_interaction_terms = True
        config.use_clustering = True
        config.MODEL_CV_SPLITS = 10
        st.sidebar.success("✅ Production ready preset loaded")
    
    return config

def get_user_inputs(mode):
    """Get user inputs for both Training and Prediction modes."""
    if mode == "Training":
        return get_training_inputs()
    else:
        return get_prediction_inputs()

def get_training_inputs():
    """Get user inputs for Training mode."""
    st.header("Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
        create_tooltip(TOOLTIPS["file_upload"])
        if uploaded_file is not None:
            config.update(file_path=uploaded_file)
            
            if uploaded_file.name.endswith('.xlsx'):
                try:
                    xls = pd.ExcelFile(uploaded_file)
                    sheet_name = st.selectbox("Select sheet", xls.sheet_names)
                    create_tooltip(TOOLTIPS["sheet_selection"])
                    config.update(sheet_name=sheet_name)
                except Exception as e:
                    st.error(f"Error reading Excel file: {str(e)}")
                    return None
        
        train_size = st.slider("Select percentage of data for training", 0.1, 0.9, 0.8)
        create_tooltip(TOOLTIPS["train_test_split"])
        config.update(train_size=train_size)
    
    with col2:
        use_clustering = st.checkbox("Use clustering", value=False)
        create_tooltip(TOOLTIPS["use_clustering"])
        config.update(use_clustering=use_clustering)
        
        if use_clustering:
            display_clustering_options()
        
        models_to_use = st.multiselect("Select models to use", list(MODEL_CLASSES.keys()))
        create_tooltip(TOOLTIPS["models_to_use"])
        config.update(models_to_use=models_to_use)
        
        tuning_method = st.selectbox("Select tuning method", ["None", "GridSearchCV", "RandomizedSearchCV"])
        create_tooltip(TOOLTIPS["tuning_method"])
        config.update(tuning_method=tuning_method)
    
    display_outlier_removal_options(config)
    
    return config

def display_clustering_options(config=None):
    """Display options for clustering configuration."""
    if config is None:
        from _0config import config as global_config
        config = global_config
    
    st.subheader("Clustering Configuration")
    create_info_button("clustering_configuration", unique_key="clustering_config_info")
    
    # 1D Clustering
    st.write("1D Clustering Options:")
    # Use ALL columns except target
    all_columns = [col for col in config.all_columns if col != config.target_column]
    
    for col in all_columns:
        method = st.selectbox(f"Select clustering method for {col}", AVAILABLE_CLUSTERING_METHODS, key=f"cluster_method_{col}")
        create_tooltip(TOOLTIPS["clustering_method"])
        if method != 'None':
            if method == 'DBSCAN':
                eps = st.slider(f"DBSCAN eps for {col}", 0.1, 1.0, DBSCAN_PARAMETERS['eps'], key=f"dbscan_eps_{col}")
                create_tooltip(TOOLTIPS["dbscan_eps"])
                min_samples = st.slider(f"DBSCAN min_samples for {col}", 2, 10, DBSCAN_PARAMETERS['min_samples'], key=f"dbscan_min_samples_{col}")
                create_tooltip(TOOLTIPS["dbscan_min_samples"])
                params = {'eps': eps, 'min_samples': min_samples}
            elif method == 'KMeans':
                n_clusters = st.slider(f"KMeans n_clusters for {col}", 2, 10, KMEANS_PARAMETERS['n_clusters'], key=f"kmeans_n_clusters_{col}")
                create_tooltip(TOOLTIPS["kmeans_n_clusters"])
                params = {'n_clusters': n_clusters}
            config.clustering_config[col] = {'method': method, 'params': params}
        else:
            config.clustering_config[col] = {'method': 'None', 'params': {}}
    
    # 2D Clustering
    st.write("2D Clustering Options:")
    col_pairs = select_2d_clustering_columns(config)
    for pair in col_pairs:    
        method = st.selectbox(f"Select clustering method for {pair}", AVAILABLE_CLUSTERING_METHODS, key=f"cluster_method_{pair}")
        create_tooltip(TOOLTIPS["clustering_method"])
        if method != 'None':
            if method == 'DBSCAN':
                eps = st.slider(f"DBSCAN eps for {pair}", 0.1, 1.0, DBSCAN_PARAMETERS['eps'], key=f"dbscan_eps_{pair}")
                create_tooltip(TOOLTIPS["dbscan_eps"])
                min_samples = st.slider(f"DBSCAN min_samples for {pair}", 2, 10, DBSCAN_PARAMETERS['min_samples'], key=f"dbscan_min_samples_{pair}")
                create_tooltip(TOOLTIPS["dbscan_min_samples"])
                params = {'eps': eps, 'min_samples': min_samples}
            elif method == 'KMeans':
                n_clusters = st.slider(f"KMeans n_clusters for {pair}", 2, 10, KMEANS_PARAMETERS['n_clusters'], key=f"kmeans_n_clusters_{pair}")
                create_tooltip(TOOLTIPS["kmeans_n_clusters"])
                params = {'n_clusters': n_clusters}
            config.set_2d_clustering([pair], method, params)
        else:
            config.set_2d_clustering([pair], 'None', {})
    
    return config

def select_2d_clustering_columns(config=None):
    """Allow users to select pairs of columns for 2D clustering."""
    if config is None:
        from _0config import config as global_config
        config = global_config
    
    st.write("Select column pairs for 2D clustering:")
    create_tooltip(TOOLTIPS["2d_clustering"])
    col_pairs = []
    
    # Include ALL columns, not just numerical and non-unused
    valid_columns = list(set(config.all_columns) - set([config.target_column]))
    
    num_pairs = st.number_input("Number of column pairs for 2D clustering", min_value=0, max_value=len(valid_columns)//2, value=0)
    
    for i in range(num_pairs):
        col1 = st.selectbox(f"Select first column for pair {i+1}", valid_columns, key=f"2d_cluster_col1_{i}")
        remaining_columns = [col for col in valid_columns if col != col1]
        col2 = st.selectbox(f"Select second column for pair {i+1}", remaining_columns, key=f"2d_cluster_col2_{i}")
        if col1 != col2:
            col_pairs.append((col1, col2))
        else:
            st.warning(f"Pair {i+1}: Please select different columns.")
    
    config.set_2d_clustering_columns([col for pair in col_pairs for col in pair])
    return col_pairs

def display_outlier_removal_options(config, available_columns=None):
    """
    Display options for outlier removal with separate target control.
    
    Args:
    - config: Configuration object
    - available_columns: List of available columns (optional)
    
    Returns:
    - Updated config object
    """
    st.subheader("🎯 Outlier Removal Options")
    create_info_button("outlier_removal")
    
    # Determine available columns
    if available_columns is None:
        available_columns = config.numerical_columns
    
    # Add target column if it exists and is numerical
    if config.target_column and config.target_column not in available_columns:
        available_columns = list(set(available_columns) | {config.target_column})
    
    available_columns = list(set(available_columns))
    
    # ========================================================================
    # FEATURE OUTLIER REMOVAL
    # ========================================================================
    st.write("**Feature Outliers**")
    st.info("💡 Removing feature outliers generally improves model performance")
    
    feature_columns = [col for col in available_columns if col != config.target_column]
    
    outlier_columns = []
    if feature_columns:
        select_all_features = st.checkbox("Select all features for outlier removal", value=False)
        
        if select_all_features:
            outlier_columns = feature_columns.copy()
            st.success(f"✅ All {len(feature_columns)} features selected")
        else:
            for col in feature_columns:
                if st.checkbox(f"Remove outliers from: {col}", key=f"outlier_feature_{col}"):
                    outlier_columns.append(col)
    
    # ========================================================================
    # TARGET OUTLIER REMOVAL (Separate control with warnings)
    # ========================================================================
    st.write("**Target Variable Outliers**")
    st.warning("⚠️ **Caution**: Removing target outliers can discard valuable information!")
    
    with st.expander("ℹ️ When to remove target outliers?", expanded=False):
        st.write("""
        **Consider removing target outliers when:**
        - You have confirmed data entry errors
        - Outliers represent impossible values
        - You're building a model for "typical" cases only
        
        **Do NOT remove target outliers when:**
        - They represent legitimate rare events (luxury homes, executive salaries)
        - You need to predict extreme values
        - Outliers are the most interesting cases
        - Your target has a heavy-tailed distribution (common in real estate, finance)
        
        **Alternative approaches:**
        - Use robust models (tree-based: RF, XGBoost) that handle outliers well
        - Log-transform the target instead of removing outliers
        - Train separate models for different value ranges (via clustering)
        """)
    
    remove_target_outliers = st.checkbox(
        f"Remove outliers from target: {config.target_column}", 
        value=False,
        help="Uncheck unless you have specific reasons to remove target outliers"
    )
    
    target_outlier_method = 'none'
    if remove_target_outliers:
        st.error("⚠️ You've chosen to remove target outliers. Please select a method:")
        
        target_outlier_method = st.selectbox(
            "Target outlier detection method",
            options=['none', 'zscore', 'iqr', 'percentile'],
            index=0,
            format_func=lambda x: {
                'none': 'None - Keep all target values',
                'zscore': 'Z-Score (assumes normal distribution)',
                'iqr': 'IQR Method (robust to skewed data) - RECOMMENDED',
                'percentile': 'Percentile Method (custom cutoffs)'
            }[x],
            help="""
            • Z-Score: Removes values >3 standard deviations (assumes normal distribution)
            • IQR: Removes values outside Q1-1.5*IQR to Q3+1.5*IQR (more robust)
            • Percentile: Custom cutoff (e.g., remove bottom 1% and top 1%)
            """
        )
        
        if target_outlier_method != 'none':
            confirm_removal = st.checkbox(
                "⚠️ I understand this may remove valuable data points",
                value=False
            )
            
            if not confirm_removal:
                st.warning("Target outlier removal disabled - confirmation required")
                remove_target_outliers = False
                target_outlier_method = 'none'
    
    # ========================================================================
    # OUTLIER THRESHOLD CONTROL
    # ========================================================================
    if outlier_columns or remove_target_outliers:
        st.write("**Outlier Detection Threshold**")
        
        outlier_threshold = st.slider(
            "Z-score threshold (for features)",
            min_value=2.0,
            max_value=5.0,
            value=3.0,
            step=0.1,
            help="Data points with |z-score| > threshold are considered outliers. Higher = more permissive."
        )
        
        config.OUTLIER_THRESHOLD = outlier_threshold
    else:
        outlier_threshold = 3.0  # Default
    
    # Update config
    config.update_outlier_removal_columns(outlier_columns)
    config.remove_target_outliers = remove_target_outliers
    config.target_outlier_method = target_outlier_method
    
    return config

def get_prediction_inputs():
    """Get user inputs for Prediction mode."""
    st.header("Prediction Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_saved_models = st.radio("Use saved models?", ["Yes", "No"])
        create_tooltip(TOOLTIPS["use_saved_models"])
        
        if use_saved_models == "No":
            uploaded_models = st.file_uploader("Upload trained models", type="joblib", accept_multiple_files=True)
            create_tooltip(TOOLTIPS["upload_models"])
            uploaded_preprocessor = st.file_uploader("Upload preprocessor", type="joblib")
            create_tooltip(TOOLTIPS["upload_preprocessor"])
            config.update(uploaded_models=uploaded_models, uploaded_preprocessor=uploaded_preprocessor)
    
    with col2:
        new_data_file = st.file_uploader("Choose a CSV file with new data for prediction", type="csv")
        create_tooltip(TOOLTIPS["new_data_file"])
        if new_data_file is not None:
            config.update(new_data_file=new_data_file)
    
    return config