import os
import joblib
import pandas as pd
import numpy as np
import scipy.sparse
import streamlit as st
import plotly.express as px

from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.ensemble import VotingRegressor
from sklearn.pipeline import Pipeline
from _0config import (
    config, MODELS_DIRECTORY, MODEL_CLASSES, HYPERPARAMETER_GRIDS, RANDOM_STATE,
    ENSEMBLE_CV_SPLITS, ENSEMBLE_CV_SHUFFLE, MODEL_CV_SPLITS, RANDOMIZED_SEARCH_ITERATIONS,
    TOOLTIPS, INFO_TEXTS, CHART_HEIGHT, CHART_WIDTH
)
from _2misc_utils import debug_print, plot_prediction_vs_actual
from _2ui_utils import create_tooltip, create_info_button
from _7metrics import calculate_metrics

def get_model_instance(model_name, config=None):
    """
    Create an instance of a machine learning model based on its name.
    
    Args:
    - model_name: Name of the model
    - config: Configuration object (optional)
    
    Returns:
    - Model instance
    """
    if config is None:
        from _0config import config as global_config
        config = global_config
    
    if model_name in MODEL_CLASSES:
        return MODEL_CLASSES[model_name]()
    else:
        raise ValueError(f"Model {model_name} is not recognized.")

def create_pipeline(model, preprocessor):
    """Create a pipeline that includes the preprocessor and the model."""
    if preprocessor is None:
        # No preprocessing needed
        return model
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

def tune_hyperparameters(model_name, model, x_train, y_train, preprocessor, tuning_strategy, config=None):
    """
    Tune the hyperparameters of the model.
    
    Args:
    - model_name: Name of the model
    - model: Model instance
    - x_train: Training features
    - y_train: Training target
    - preprocessor: Preprocessing pipeline
    - tuning_strategy: 'GridSearchCV', 'RandomizedSearchCV', or 'None'
    - config: Configuration object (optional)
    
    Returns:
    - Best estimator
    """
    if config is None:
        from _0config import config as global_config
        config = global_config
    
    st.write(f"Tuning hyperparameters for {model_name} using {tuning_strategy}")
    create_tooltip(TOOLTIPS["hyperparameter_tuning"])
    create_info_button("hyperparameter_tuning")
    
    pipeline = create_pipeline(model, preprocessor)
    
    # Prepare parameter grid
    if model_name in HYPERPARAMETER_GRIDS:
        if preprocessor is not None:
            param_grid = {'model__' + key: value for key, value in HYPERPARAMETER_GRIDS[model_name].items()}
        else:
            param_grid = HYPERPARAMETER_GRIDS[model_name]
    else:
        param_grid = {}
    
    progress_bar = st.progress(0)
    
    if tuning_strategy == 'GridSearchCV' and param_grid:
        grid_search = GridSearchCV(pipeline, param_grid, cv=MODEL_CV_SPLITS, n_jobs=-1, verbose=1)
        grid_search.fit(x_train, y_train)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_estimator = grid_search.best_estimator_
        
    elif tuning_strategy == 'RandomizedSearchCV' and param_grid:
        random_search = RandomizedSearchCV(
            pipeline, 
            param_grid, 
            n_iter=RANDOMIZED_SEARCH_ITERATIONS, 
            cv=MODEL_CV_SPLITS, 
            n_jobs=-1, 
            verbose=1,
            random_state=config.RANDOM_STATE
        )
        random_search.fit(x_train, y_train)
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        best_estimator = random_search.best_estimator_
        
    else:
        # No tuning or no parameters to tune
        pipeline.fit(x_train, y_train)
        best_params = {}
        best_score = cross_val_score(pipeline, x_train, y_train, cv=MODEL_CV_SPLITS).mean()
        best_estimator = pipeline

    progress_bar.progress(1.0)
    st.write(f"Best parameters: {best_params}")
    st.write(f"Best cross-validation score: {best_score:.4f}")
    return best_estimator

def save_model(model, filename, save_path=MODELS_DIRECTORY):
    """Save the trained model."""
    os.makedirs(save_path, exist_ok=True)
    model_filename = os.path.join(save_path, f"{filename}.joblib")
    joblib.dump(model, model_filename)
    st.write(f"Model saved at: {model_filename}")

def train_and_validate_models(data_splits, clustered_X_train_combined, clustered_X_test_combined, 
                              models_to_use, tuning_method, config=None):
    """
    Train and validate models for all clusters.
    
    Args:
    - data_splits: Dictionary of train/test splits
    - clustered_X_train_combined: Training features for each cluster
    - clustered_X_test_combined: Test features for each cluster
    - models_to_use: List of model names to train
    - tuning_method: 'GridSearchCV', 'RandomizedSearchCV', or 'None'
    - config: Configuration object (optional)
    
    Returns:
    - Tuple of (all_models, ensemble_cv_results, all_evaluation_metrics)
    """
    if config is None:
        from _0config import config as global_config
        config = global_config
    
    st.subheader("Training and Validating Models")
    create_info_button("model_training_validation")

    all_models = {}
    ensemble_cv_results = {}
    all_evaluation_metrics = {}

    progress_bar = st.progress(0)
    for i, (cluster_key, split_data) in enumerate(data_splits.items()):
        st.write(f"\nProcessing data for cluster: {cluster_key}")

        X_train = clustered_X_train_combined[cluster_key]
        X_test = clustered_X_test_combined[cluster_key]
        y_train = split_data['y_train']
        y_test = split_data['y_test']
        preprocessor = split_data['preprocessor']

        st.write(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        st.write(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

        models, cv_results, evaluation_metrics = train_models(
            X_train, y_train, cluster_key, models_to_use, MODELS_DIRECTORY, 
            tuning_method, preprocessor, config=config
        )

        all_models[cluster_key] = models
        ensemble_cv_results[cluster_key] = cv_results
        all_evaluation_metrics[cluster_key] = evaluation_metrics

        progress_bar.progress((i + 1) / len(data_splits))

    st.success("Training and validation completed for all selected clusters and models.")

    return all_models, ensemble_cv_results, all_evaluation_metrics

def train_models_on_flattened_data(flattened_x_train, flattened_y_train, models_to_use, 
                                  tuning_method, global_preprocessor, config=None):
    """
    Train models on flattened (combined) data.
    
    Args:
    - flattened_x_train: Combined training features
    - flattened_y_train: Combined training target
    - models_to_use: List of model names
    - tuning_method: Tuning strategy
    - global_preprocessor: Global preprocessing pipeline
    - config: Configuration object (optional)
    
    Returns:
    - Tuple of (flattened_models, flattened_cv_results)
    """
    if config is None:
        from _0config import config as global_config
        config = global_config
    
    st.subheader("Training Models on Flattened Data")
    create_info_button("flattened_data_training")

    flattened_models = {}
    flattened_cv_results = {}

    if not flattened_x_train.empty:
        # Use the global_preprocessor to transform the data
        flattened_x_train_transformed = global_preprocessor.transform(flattened_x_train)
        
        # Get feature names from the preprocessor
        try:
            feature_names = (global_preprocessor.named_transformers_['num'].get_feature_names_out().tolist() +
                           global_preprocessor.named_transformers_['cat'].get_feature_names_out().tolist())
        except:
            # Fallback if feature names can't be extracted
            feature_names = [f"feature_{i}" for i in range(flattened_x_train_transformed.shape[1])]

        # Convert the transformed data to a DataFrame
        flattened_x_train_transformed_df = pd.DataFrame(
            flattened_x_train_transformed, 
            columns=feature_names, 
            index=flattened_x_train.index
        )

        # Train models on the flattened data
        flattened_models, flattened_cv_results, _ = train_models(
            flattened_x_train_transformed_df, flattened_y_train, "flattened", 
            models_to_use, MODELS_DIRECTORY, tuning_method, None, config=config
        )

        st.success("Model training on flattened data completed.")
    else:
        st.warning("No data available for training models on flattened data.")

    return flattened_models, flattened_cv_results

def create_ensemble_model(all_models, x_train, y_train, preprocessor, 
                         save_path=MODELS_DIRECTORY, config=None):
    """
    Create an ensemble model from multiple trained models.
    
    Args:
    - all_models: Dictionary of trained models
    - x_train: Training features
    - y_train: Training target
    - preprocessor: Preprocessing pipeline
    - save_path: Directory to save ensemble model
    - config: Configuration object (optional)
    
    Returns:
    - Tuple of (ensemble_pipeline, cv_scores)
    """
    if config is None:
        from _0config import config as global_config
        config = global_config
    
    st.subheader("Creating Ensemble Model")
    create_info_button("ensemble_model")

    if not all_models:
        st.warning("No models available for ensembling.")
        return None, []

    # Print the models being included in the ensemble
    st.write("Models included in the ensemble:")
    for model_name in all_models:
        st.write(f" - {model_name}")

    ensemble_models = [(f"{model_name}", model) for model_name, model in all_models.items()]
    ensemble = VotingRegressor(estimators=ensemble_models)

    # Create a pipeline with preprocessor and ensemble
    if preprocessor is not None:
        ensemble_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('ensemble', ensemble)
        ])
    else:
        ensemble_pipeline = ensemble

    st.write("Initiated cross-validation for the ensemble model...")
    cv_scores = cross_val_score(
        ensemble_pipeline, x_train, y_train, 
        cv=KFold(n_splits=ENSEMBLE_CV_SPLITS, shuffle=ENSEMBLE_CV_SHUFFLE, random_state=config.RANDOM_STATE)
    )

    # Print detailed cross-validation scores
    st.write("Detailed cross-validation scores:")
    for fold_index, score in enumerate(cv_scores, start=1):
        st.write(f" - Fold {fold_index}: Score = {score:.4f}")

    st.write(f"Average cross-validation score for ensemble: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} (std dev)")

    # Fit the ensemble on the entire training data
    ensemble_pipeline.fit(x_train, y_train)

    # Saving the ensemble model
    os.makedirs(save_path, exist_ok=True)
    ensemble_filename = os.path.join(save_path, 'ensemble_model.joblib')
    joblib.dump(ensemble_pipeline, ensemble_filename)
    st.success(f"Ensemble model saved successfully at: {ensemble_filename}")

    # Optionally, provide insights into the ensemble's composition
    st.write("Ensemble model composition:")
    for model_name, _ in ensemble_models:
        st.write(f" - {model_name}")

    return ensemble_pipeline, cv_scores.tolist()

def train_models(x_train, y_train, cluster_key, models_to_use, save_path, 
                tuning_method, preprocessor, config=None):
    """
    Train and evaluate multiple models for a specific cluster.
    
    Args:
    - x_train: Training features
    - y_train: Training target
    - cluster_key: Cluster identifier
    - models_to_use: List of model names
    - save_path: Directory to save models
    - tuning_method: Tuning strategy
    - preprocessor: Preprocessing pipeline
    - config: Configuration object (optional)
    
    Returns:
    - Tuple of (models, cv_results, evaluation_metrics)
    """
    if config is None:
        from _0config import config as global_config
        config = global_config
    
    st.write(f"Training models for cluster: {cluster_key}")

    models = {}
    cv_results = {}
    evaluation_metrics = {}

    for model_name in models_to_use:
        st.write(f"Training {model_name} for cluster {cluster_key}")
        model = get_model_instance(model_name, config=config)

        if tuning_method == 'GridSearchCV' or tuning_method == 'RandomizedSearchCV':
            best_model = tune_hyperparameters(
                model_name, model, x_train, y_train, preprocessor, tuning_method, config=config
            )
            cv_scores = cross_val_score(best_model, x_train, y_train, cv=MODEL_CV_SPLITS)
        else:
            st.write(f"Training {model_name} without hyperparameter tuning on {cluster_key}...")
            if preprocessor is not None:
                best_model = create_pipeline(model, preprocessor)
            else:
                best_model = model
            best_model.fit(x_train, y_train)
            cv_scores = cross_val_score(best_model, x_train, y_train, cv=MODEL_CV_SPLITS)

        save_model(best_model, f"{cluster_key}_{model_name}", save_path)

        models[model_name] = best_model
        cv_results[model_name] = cv_scores.tolist()
        y_pred = best_model.predict(x_train)
        evaluation_metrics[model_name] = calculate_metrics(y_train, y_pred)

        st.write(f"Model {model_name} for cluster {cluster_key}:")
        st.write(f"  CV scores: {cv_scores}")
        st.write(f"  Evaluation metrics: {evaluation_metrics[model_name]}")

        # Plot actual vs predicted values
        plot_prediction_vs_actual(y_train, y_pred, title=f"{model_name} - Actual vs Predicted ({cluster_key})")
        
        # Plot learning curve
        plot_learning_curve(best_model, x_train, y_train, model_name, cluster_key)

    return models, cv_results, evaluation_metrics

def predict_with_model(model, x_data, preprocessor=None):
    """Make predictions using a trained model."""
    if preprocessor is not None:
        x_data = preprocessor.transform(x_data)
    return model.predict(x_data)

def load_trained_models(models_directory):
    """Load all trained models from the specified directory."""
    trained_models = {}
    if not os.path.exists(models_directory):
        st.warning(f"Models directory {models_directory} does not exist.")
        return trained_models
    
    for filename in os.listdir(models_directory):
        if filename.endswith('.joblib') and not filename.startswith('preprocessor'):
            model_path = os.path.join(models_directory, filename)
            model_name = filename.replace('.joblib', '')
            try:
                trained_models[model_name] = joblib.load(model_path)
            except Exception as e:
                st.warning(f"Failed to load {filename}: {str(e)}")
    
    return trained_models

def display_model_performance(all_evaluation_metrics):
    """Display performance metrics for all models."""
    st.subheader("Model Performance")
    create_info_button("model_performance")
    for cluster_key, cluster_metrics in all_evaluation_metrics.items():
        st.write(f"Cluster: {cluster_key}")
        for model_name, metrics in cluster_metrics.items():
            st.write(f"Model: {model_name}")
            for metric_name, metric_value in metrics.items():
                st.write(f"  {metric_name}: {metric_value:.4f}")
        st.write("---")

def save_trained_models(all_models, save_path):
    """Save all trained models to the specified directory."""
    os.makedirs(save_path, exist_ok=True)
    for cluster_key, models in all_models.items():
        for model_name, model in models.items():
            filename = f"{cluster_key}_{model_name}"
            save_model(model, filename, save_path)
    st.success("All trained models saved successfully.")

def plot_learning_curve(estimator, X, y, model_name, cluster_key):
    """Plot the learning curve for a trained model."""
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 5)
        )
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        fig = px.line()
        fig.add_scatter(x=train_sizes, y=train_scores_mean, mode='lines', name='Training score',
                        error_y=dict(type='data', array=train_scores_std, visible=True))
        fig.add_scatter(x=train_sizes, y=test_scores_mean, mode='lines', name='Cross-validation score',
                        error_y=dict(type='data', array=test_scores_std, visible=True))
        
        fig.update_layout(
            title=f'Learning Curve - {model_name} ({cluster_key})',
            xaxis_title='Training examples',
            yaxis_title='Score',
            height=CHART_HEIGHT, 
            width=CHART_WIDTH
        )
        st.plotly_chart(fig)
    except Exception as e:
        st.warning(f"Could not plot learning curve for {model_name}: {str(e)}")