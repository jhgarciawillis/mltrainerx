import streamlit as st
import pandas as pd
import os
import json
import joblib
from functools import partial

from _0config import Config, STREAMLIT_APP_NAME, STREAMLIT_APP_ICON, TOOLTIPS, INFO_TEXTS
from _2data_utils import (
    load_data, 
    display_data_info, 
    handle_missing_values, 
    auto_detect_column_types, 
    display_column_selection, 
    save_unused_data, 
    check_and_remove_duplicate_columns, 
    check_and_reset_indices,
    display_sheet_selection
)
from _2ui_utils import (
    display_metrics, 
    get_user_inputs, 
    get_training_inputs, 
    display_clustering_options, 
    select_2d_clustering_columns, 
    get_prediction_inputs, 
    create_tooltip, 
    create_info_button,
    display_outlier_removal_options,
    truncate_sheet_name,
    validate_file_upload,
    display_config_manager
)
from _2misc_utils import (
    debug_print, 
    create_correlation_heatmap, 
    display_data_statistics, 
    check_data_balance,
    plot_prediction_vs_actual,
    plot_residuals
)
from _3preprocessing import (
    load_and_preprocess_data, 
    split_and_preprocess_data, 
    create_global_preprocessor, 
    save_global_preprocessor, 
    load_global_preprocessor
)
from _4cluster import create_clusters, load_clustering_models, load_clustering_predictors
from _5feature import (
    apply_feature_generation, generate_polynomial_features, generate_interaction_terms, 
    generate_statistical_features, combine_feature_engineered_data, 
    generate_features_for_prediction, visualize_feature_distributions, analyze_feature_interactions
)
from _6training import (
    train_and_validate_models, create_ensemble_model, train_models_on_flattened_data, 
    load_trained_models, predict_with_model, save_trained_models
)
from _7metrics import (
    calculate_metrics, display_metrics, plot_residuals, plot_actual_vs_predicted, 
    plot_prediction_distribution, plot_error_distribution
)
from _8prediction import PredictionProcessor, load_saved_models, predict_for_new_data

# Set page config as the first Streamlit command
st.set_page_config(
    page_title=STREAMLIT_APP_NAME,
    page_icon=STREAMLIT_APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    st.title(STREAMLIT_APP_NAME)

    # Mode selection
    mode = st.sidebar.radio("Select Mode", ["Training", "Prediction"])

    # Initialize Config - use session state for persistence and multi-user isolation
    if 'config' not in st.session_state:
        st.session_state.config = Config()
        st.info(f"🆕 New session created: {st.session_state.config.session_id[:8]}")
    
    config = st.session_state.config
    
    # Add config management in sidebar
    with st.sidebar:
        st.subheader("Session Management")
        st.text(f"Session: {config.session_id[:8]}")
        
        if st.button("Reset Configuration"):
            st.session_state.config = Config()
            st.success("Configuration reset!")
            st.rerun()
        
        if st.button("Export Configuration"):
            config_json = json.dumps(config.to_dict(), indent=2)
            st.download_button(
                "Download Config JSON",
                config_json,
                file_name=f"config_{config.session_id[:8]}.json",
                mime="application/json"
            )
        
        uploaded_config = st.file_uploader("Import Configuration", type=['json'], key='config_upload')
        if uploaded_config:
            config_dict = json.load(uploaded_config)
            st.session_state.config = Config.from_dict(config_dict)
            st.success("Configuration imported!")
            st.rerun()
    
    # Display config manager
    config = display_config_manager(config)

    if mode == "Training":
        run_training_mode(config)
    else:
        run_prediction_mode(config)

def run_training_mode(config):
    st.header("Training Mode")

    # 1. Data Input and Initial Configuration
    st.subheader("1. Data Input and Initial Configuration")
    create_info_button("data_input")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    create_tooltip(TOOLTIPS["file_upload"])
    
    if uploaded_file is not None:
        if validate_file_upload(uploaded_file):
            config.update(file_path=uploaded_file)
            
            # REMOVED: Early randomization - now handled in training phase
            
            if uploaded_file.name.endswith('.xlsx'):
                sheet_name = display_sheet_selection(uploaded_file)
                config.update(sheet_name=sheet_name)
            
            # Load the original data WITHOUT randomization
            data = load_data(config.file_path, config.sheet_name)
            
            if data is not None:
                st.info("ℹ️ Data loaded in original order. Randomization (if enabled) will occur during train/test split.")
                
                display_data_info(data)
                create_correlation_heatmap(data)
                display_data_statistics(data)
                
                # 2. Data Preprocessing
                st.subheader("2. Data Preprocessing")
                create_info_button("data_preprocessing")
                
                # Automatic column type detection
                create_tooltip(TOOLTIPS["auto_detect_column_types"])
                initial_types = auto_detect_column_types(data)
                
                # Manual column selection
                create_tooltip(TOOLTIPS["manual_column_selection"])
                selected_columns = display_column_selection(data.columns, initial_types)
                
                if selected_columns:
                    # Update config with selected column types
                    config.set_column_types(**selected_columns)
                    
                    # Check data balance only after target column is set
                    if config.target_column is not None and config.target_column in data.columns:
                        check_data_balance(data, config.target_column)
                    else:
                        st.warning("Please select a target column before proceeding.")
                    
                    # Handle missing values
                    create_tooltip(TOOLTIPS["handle_missing_values"])
                    data = handle_missing_values(data)
                    
                    # Remove duplicate columns
                    data = check_and_remove_duplicate_columns(data)
                    
                    # Reset indices if needed
                    data = check_and_reset_indices(data)
                    
                    # Outlier removal with enhanced control
                    create_tooltip(TOOLTIPS["outlier_removal"])
                    config = display_outlier_removal_options(config)
                    
                    # 3. Feature Engineering
                    st.subheader("3. Feature Engineering")
                    create_info_button("feature_engineering")
                    config = configure_feature_engineering(config)
                    
                    # Visualize feature distributions and interactions
                    if config.numerical_columns:
                        visualize_feature_distributions(data[config.numerical_columns])
                        if config.target_column in data.columns:
                            analyze_feature_interactions(data[config.numerical_columns], data[config.target_column])
                    
                    # 4. Clustering Configuration
                    st.subheader("4. Clustering Configuration")
                    create_info_button("clustering_configuration")
                    config = configure_clustering(config)
                    
                    # 5. Model Selection and Training
                    st.subheader("5. Model Selection and Training")
                    create_info_button("model_selection_training")
                    config = configure_model_training(config)
                    
                    # 6. Advanced Options
                    st.subheader("6. Advanced Options")
                    create_info_button("advanced_options")
                    config = configure_advanced_options(config)
                    
                    # 7. Execution and Results
                    if st.button("Start Training"):
                        with st.spinner("Training in progress..."):
                            execute_training(data, config)
                else:
                    st.warning("Please complete column selection to continue.")

def run_prediction_mode(config):
    st.header("Prediction Mode")
    
    st.subheader("1. Load Saved Models")
    create_info_button("load_saved_models")
    
    use_saved = st.radio("Use saved models?", ["Yes", "No"])
    
    if use_saved == "Yes":
        try:
            # Load models with enhanced error handling
            all_models, artifacts, load_status = load_saved_models_and_preprocessors(config)
            global_preprocessor, clustering_config, clustering_2d_config, cluster_models, cluster_predictors = artifacts
            
            # Display load status
            display_model_load_status(load_status)
            
            # Check if critical components loaded successfully
            if not all_models:
                st.error("❌ No models loaded. Cannot proceed with predictions.")
                st.info("Please ensure you have trained models in the 'Trained' directory.")
                return
            
            st.success(f"✅ Successfully loaded {len(all_models)} model(s)")
            
            # Warning if predictors failed to load but continue anyway
            if not cluster_predictors and clustering_config:
                st.warning("⚠️ Cluster predictors failed to load. Predictions may be limited.")
                st.info("You can still make predictions, but cluster assignment may not work correctly.")
            
            st.subheader("2. Upload New Data")
            create_info_button("upload_prediction_data")
            new_data = upload_prediction_data()
            
            if new_data is not None:
                st.write("New data preview:")
                st.dataframe(new_data.head())
                
                st.subheader("3. Generate Predictions")
                create_info_button("make_predictions")
                
                if st.button("Make Predictions"):
                    with st.spinner("Generating predictions..."):
                        try:
                            predictions = predict_for_new_data(
                                all_models, 
                                new_data, 
                                cluster_predictors,
                                clustering_config, 
                                clustering_2d_config
                            )
                            
                            display_predictions(predictions, config)
                        except Exception as e:
                            st.error(f"❌ Prediction failed: {str(e)}")
                            st.error(f"Error type: {type(e).__name__}")
                            
                            # Provide helpful debugging info
                            with st.expander("🔍 Debug Information"):
                                st.write("**Models loaded:**", list(all_models.keys())[:10])
                                st.write("**Cluster predictors:**", list(cluster_predictors.keys()) if cluster_predictors else "None")
                                st.write("**New data columns:**", list(new_data.columns))
                                st.write("**Full error:**")
                                st.exception(e)
        
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            st.error(f"Error type: {type(e).__name__}")
            st.info("Please ensure you have trained models in the 'Trained' directory.")
            
            # Show which files are present
            if os.path.exists(config.MODELS_DIRECTORY):
                files = os.listdir(config.MODELS_DIRECTORY)
                if files:
                    st.write("Files found in Trained directory:")
                    st.write(files)
                else:
                    st.warning("Trained directory is empty. Please train models first.")
            else:
                st.warning("Trained directory does not exist. Please train models first.")
    
    else:
        st.info("Model upload functionality - Upload your trained models manually")
        uploaded_models = st.file_uploader("Upload trained models", type="joblib", accept_multiple_files=True)
        uploaded_preprocessor = st.file_uploader("Upload preprocessor", type="joblib")

def configure_feature_engineering(config):
    config.use_polynomial_features = st.checkbox("Use polynomial features", value=config.use_polynomial_features)
    create_tooltip(TOOLTIPS["polynomial_features"])
    
    config.use_interaction_terms = st.checkbox("Use interaction terms", value=config.use_interaction_terms)
    create_tooltip(TOOLTIPS["interaction_terms"])
    
    config.use_statistical_features = st.checkbox("Use statistical features", value=config.use_statistical_features)
    create_tooltip(TOOLTIPS["statistical_features"])
    
    return config

def configure_clustering(config):
    config.use_clustering = st.checkbox("Use clustering", value=config.use_clustering)
    create_tooltip(TOOLTIPS["use_clustering"])
    
    if config.use_clustering:
        config = display_clustering_options(config)
    
    return config

def configure_model_training(config):
    config.models_to_use = st.multiselect("Select models to use", list(config.MODEL_CLASSES.keys()), default=config.models_to_use)
    create_tooltip(TOOLTIPS["models_to_use"])
    
    config.tuning_method = st.selectbox("Select tuning method", ["None", "GridSearchCV", "RandomizedSearchCV"], index=["None", "GridSearchCV", "RandomizedSearchCV"].index(config.tuning_method))
    create_tooltip(TOOLTIPS["tuning_method"])
    
    config.train_size = st.slider("Train/Test split ratio", 0.1, 0.9, config.train_size)
    create_tooltip(TOOLTIPS["train_test_split"])
    
    return config

def configure_advanced_options(config):
    config.RANDOM_STATE = st.number_input("Random state", value=config.RANDOM_STATE, min_value=0)
    create_tooltip(TOOLTIPS["random_state"])
    
    config.MODEL_CV_SPLITS = st.number_input("Cross-validation folds", min_value=2, max_value=10, value=config.MODEL_CV_SPLITS)
    create_tooltip(TOOLTIPS["cv_folds"])
    
    return config

def execute_training(data, config):
    """Execute the training pipeline with the given configuration."""
    
    # Validate configuration before starting
    try:
        config.validate()
    except ValueError as e:
        st.error(str(e))
        return
    
    # Display configuration summary
    with st.expander("📋 Configuration Summary", expanded=False):
        st.json(config.to_dict())
    
    preprocessed_data = load_and_preprocess_data(data, config)
    
    # ========================================================================
    # SINGLE RANDOMIZATION POINT - All user control here
    # ========================================================================
    st.subheader("🎲 Data Randomization Control")
    st.info("💡 **Recommendation**: Enable randomization to prevent temporal bias in your model.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        randomize = st.checkbox(
            "Randomize data before training", 
            value=True,
            help="Shuffle data rows before train/test split. Recommended for most cases."
        )
    
    with col2:
        if randomize:
            randomization_method = st.selectbox(
                "Randomization strategy",
                options=['together', 'stratified', 'none_in_split'],
                format_func=lambda x: {
                    'together': 'Together (Recommended)',
                    'stratified': 'Stratified (Imbalanced data)',
                    'none_in_split': 'Let train_test_split handle it'
                }[x],
                help=TOOLTIPS["randomization_strategy"]
            )
        else:
            randomization_method = None
            st.warning("⚠️ Randomization disabled. Use only if your data has important temporal structure.")
    
    with col3:
        if randomize:
            random_seed = st.number_input(
                "Random seed", 
                value=config.RANDOM_STATE,
                min_value=0,
                max_value=99999,
                help="Set seed for reproducibility. Same seed = same split every time."
            )
            config.RANDOM_STATE = random_seed
            st.info(f"Seed: {random_seed}")
    
    # Show what will happen
    if randomize:
        st.success(f"✅ Data will be randomized using '{randomization_method}' method with seed {config.RANDOM_STATE}")
    else:
        st.warning("⚠️ Data will maintain original order (no randomization)")
    
    # ========================================================================
    # End of randomization control
    # ========================================================================
    
    if config.use_clustering:
        clustered_data = create_clusters(
            preprocessed_data, 
            config.clustering_config, 
            config.clustering_2d_config, 
            config.target_column
        )
    else:
        clustered_data = {'no_cluster': {'label_0': preprocessed_data.index.tolist()}}
    
    # Pass randomization settings to split function
    data_splits = split_and_preprocess_data(
        preprocessed_data, 
        clustered_data, 
        config.target_column, 
        config.train_size,
        randomize=randomize,
        randomization_method=randomization_method if randomize else None,
        config=config
    )
    
    # Build feature generation functions with config bound using partial
    feature_generation_functions = []
    if config.use_polynomial_features:
        feature_generation_functions.append(partial(generate_polynomial_features, config=config))
    if config.use_interaction_terms:
        feature_generation_functions.append(partial(generate_interaction_terms, config=config))
    if config.use_statistical_features:
        feature_generation_functions.append(partial(generate_statistical_features, config=config))
    
    clustered_X_train_combined, clustered_X_test_combined = apply_feature_generation(
        data_splits, 
        feature_generation_functions, 
        config=config
    )
    
    all_models, ensemble_cv_results, all_evaluation_metrics = train_and_validate_models(
        data_splits, 
        clustered_X_train_combined, 
        clustered_X_test_combined, 
        config.models_to_use, 
        config.tuning_method,
        config=config
    )
    
    # Save models and configuration together
    save_trained_models(all_models, config.MODELS_DIRECTORY)
    
    # Save configuration with models
    config_path = os.path.join(config.MODELS_DIRECTORY, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    st.success(f"Configuration saved to {config_path}")
    
    # Save other artifacts
    if config.unused_columns:
        save_unused_data(data[config.unused_columns], os.path.join(config.MODELS_DIRECTORY, "unused_data.csv"))
    
    joblib.dump(config.clustering_config, os.path.join(config.MODELS_DIRECTORY, "clustering_config.joblib"))
    joblib.dump(config.clustering_2d_config, os.path.join(config.MODELS_DIRECTORY, "clustering_2d_config.joblib"))
    
    st.success("Training completed successfully!")
    
    display_training_results(all_evaluation_metrics)
    
    # Plot prediction distribution and error distribution
    for cluster_name, cluster_metrics in all_evaluation_metrics.items():
        for model_name, metrics in cluster_metrics.items():
            y_true = data_splits[cluster_name]['y_test']
            y_pred = all_models[cluster_name][model_name].predict(data_splits[cluster_name]['X_test'])
            plot_prediction_distribution(y_true, y_pred, title=f"{model_name} - {cluster_name}")
            plot_error_distribution(y_true, y_pred, title=f"{model_name} - {cluster_name}")


def load_saved_models_and_preprocessors(config):
    """
    Load all saved artifacts including cluster predictors with enhanced error handling.
    
    UPDATED: Now returns load status for better user feedback.
    
    Returns:
        tuple: (all_models, artifacts, load_status)
            - all_models: Dictionary of trained models
            - artifacts: Tuple of (global_preprocessor, clustering_config, clustering_2d_config, 
                                   cluster_models, cluster_predictors)
            - load_status: Dictionary with loading success/failure for each component
    """
    load_status = {
        'models': False,
        'global_preprocessor': False,
        'clustering_config': False,
        'clustering_2d_config': False,
        'cluster_models': False,
        'cluster_predictors': False
    }
    
    # Load models
    try:
        all_models = load_saved_models(config.MODELS_DIRECTORY)
        load_status['models'] = len(all_models) > 0
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        all_models = {}
    
    # Load global preprocessor
    try:
        global_preprocessor = load_global_preprocessor(config.MODELS_DIRECTORY)
        load_status['global_preprocessor'] = global_preprocessor is not None
    except Exception as e:
        st.warning(f"Failed to load global preprocessor: {str(e)}")
        global_preprocessor = None
    
    # Load clustering config
    try:
        clustering_config_path = os.path.join(config.MODELS_DIRECTORY, "clustering_config.joblib")
        if os.path.exists(clustering_config_path):
            clustering_config = joblib.load(clustering_config_path)
            load_status['clustering_config'] = True
        else:
            clustering_config = {}
    except Exception as e:
        st.warning(f"Failed to load clustering config: {str(e)}")
        clustering_config = {}
    
    # Load 2D clustering config
    try:
        clustering_2d_config_path = os.path.join(config.MODELS_DIRECTORY, "clustering_2d_config.joblib")
        if os.path.exists(clustering_2d_config_path):
            clustering_2d_config = joblib.load(clustering_2d_config_path)
            load_status['clustering_2d_config'] = True
        else:
            clustering_2d_config = {}
    except Exception as e:
        st.warning(f"Failed to load 2D clustering config: {str(e)}")
        clustering_2d_config = {}
    
    # Load cluster models
    try:
        cluster_models = load_clustering_models(config.MODELS_DIRECTORY)
        load_status['cluster_models'] = len(cluster_models) > 0
    except Exception as e:
        st.warning(f"Failed to load cluster models: {str(e)}")
        cluster_models = {}
    
    # Load cluster predictors (CRITICAL - with enhanced error handling)
    try:
        cluster_predictors = load_clustering_predictors(config.MODELS_DIRECTORY)
        load_status['cluster_predictors'] = len(cluster_predictors) > 0 if cluster_predictors else False
        
        # If predictors didn't load but clustering config exists, show detailed warning
        if not load_status['cluster_predictors'] and (clustering_config or clustering_2d_config):
            st.warning("⚠️ Clustering is configured but predictors failed to load.")
            st.info("This may happen if models were trained with an older version. Predictions will be limited.")
            
    except Exception as e:
        st.error(f"❌ Failed to load cluster predictors: {str(e)}")
        st.info("Continuing without cluster predictors. Predictions may be limited.")
        cluster_predictors = {}
        load_status['cluster_predictors'] = False
    
    artifacts = (global_preprocessor, clustering_config, clustering_2d_config, cluster_models, cluster_predictors)
    
    return all_models, artifacts, load_status


def display_model_load_status(load_status):
    """Display a summary of what was successfully loaded."""
    st.subheader("📦 Model Loading Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Core Components:**")
        st.write(f"{'✅' if load_status['models'] else '❌'} Trained Models")
        st.write(f"{'✅' if load_status['global_preprocessor'] else '⚠️'} Global Preprocessor")
    
    with col2:
        st.write("**Clustering Components:**")
        st.write(f"{'✅' if load_status['clustering_config'] else '⚠️'} Clustering Config")
        st.write(f"{'✅' if load_status['clustering_2d_config'] else '⚠️'} 2D Clustering Config")
        st.write(f"{'✅' if load_status['cluster_models'] else '⚠️'} Cluster Models")
        st.write(f"{'✅' if load_status['cluster_predictors'] else '⚠️'} Cluster Predictors")
    
    # Overall status
    critical_components = ['models']
    all_critical_loaded = all(load_status[comp] for comp in critical_components)
    
    if all_critical_loaded:
        st.success("✅ All critical components loaded successfully")
    else:
        st.error("❌ Some critical components failed to load")


def upload_prediction_data():
    uploaded_file = st.file_uploader("Upload new data for prediction", type=["csv"])
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

def display_predictions(predictions, config):
    st.write("Predictions:")
    st.dataframe(predictions)
    
    if config.target_column in predictions.columns:
        metrics = calculate_metrics(predictions[config.target_column], predictions['Prediction'])
        st.subheader("Prediction Metrics")
        display_metrics(metrics)
        
        plot_prediction_vs_actual(predictions[config.target_column], predictions['Prediction'])
        plot_residuals(predictions[config.target_column], predictions['Prediction'])
        plot_prediction_distribution(predictions[config.target_column], predictions['Prediction'])
        plot_error_distribution(predictions[config.target_column], predictions['Prediction'])

def display_training_results(all_evaluation_metrics):
    st.subheader("Evaluation Metrics")
    for cluster_name, metrics in all_evaluation_metrics.items():
        st.write(f"Cluster: {cluster_name}")
        for model_name, model_metrics in metrics.items():
            st.write(f"Model: {model_name}")
            display_metrics(model_metrics)

if __name__ == "__main__":
    main()