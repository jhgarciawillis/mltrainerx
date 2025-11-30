import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os

from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from _0config import config, MODELS_DIRECTORY, TOOLTIPS, INFO_TEXTS, CHART_HEIGHT, CHART_WIDTH
from _2misc_utils import debug_print, plot_prediction_vs_actual, flatten_clustered_data
from _2ui_utils import create_tooltip, create_info_button
from _7metrics import (
    calculate_metrics, display_metrics, plot_residuals, 
    plot_prediction_distribution, plot_error_distribution
)

class PredictionProcessor:
    def __init__(self, truncate_sheet_name_func, calculate_metrics_func, debug_print_func):
        self.truncate_sheet_name_func = truncate_sheet_name_func
        self.calculate_metrics_func = calculate_metrics_func
        self.debug_print_func = debug_print_func

    def create_predictions_file(self, predictions_path, all_models, flattened_models, ensemble, 
                                clustered_X_train_combined, clustered_X_test_combined, 
                                flattened_X_train, flattened_X_test, ensemble_cv_results):
        st.subheader("Creating Predictions File")
        create_info_button("predictions_file")
        with pd.ExcelWriter(predictions_path, engine='xlsxwriter') as writer:
            self._process_clustered_models(writer, all_models, clustered_X_train_combined, 
                                           clustered_X_test_combined, ensemble_cv_results)
            self._process_flattened_models(writer, flattened_models, flattened_X_train, 
                                           flattened_X_test, ensemble_cv_results)
            self._process_ensemble_model(writer, ensemble, flattened_X_train, flattened_X_test, 
                                         ensemble_cv_results)
            
        st.success(f"Predictions saved to: {predictions_path}")

    def _process_clustered_models(self, writer, all_models, clustered_X_train_combined, 
                                  clustered_X_test_combined, ensemble_cv_results):
        st.write("Processing Clustered Models")
        flattened_all_models = flatten_clustered_data(all_models)
        flattened_clustered_X_train_combined = flatten_clustered_data(clustered_X_train_combined)
        flattened_clustered_X_test_combined = flatten_clustered_data(clustered_X_test_combined)
        flattened_ensemble_cv_results = flatten_clustered_data(ensemble_cv_results)

        progress_bar = st.progress(0)
        for i, (cluster_key, models) in enumerate(flattened_all_models.items()):
            X_train = flattened_clustered_X_train_combined[cluster_key]
            X_test = flattened_clustered_X_test_combined[cluster_key]
            y_train = X_train[config.target_column]
            y_test = X_test[config.target_column]
            X_train = X_train.drop(columns=[config.target_column])
            X_test = X_test.drop(columns=[config.target_column])

            for model_name, model in models.items():
                self._process_model(writer, model, X_train, X_test, y_train, y_test, 
                                    f"{cluster_key}_{model_name}", 
                                    flattened_ensemble_cv_results.get(cluster_key, {}).get(model_name, []))
            progress_bar.progress((i + 1) / len(flattened_all_models))

    def _process_flattened_models(self, writer, flattened_models, flattened_X_train, 
                                  flattened_X_test, ensemble_cv_results):
        st.write("Processing Flattened Models")
        y_train = flattened_X_train[config.target_column]
        y_test = flattened_X_test[config.target_column]
        X_train = flattened_X_train.drop(columns=[config.target_column])
        X_test = flattened_X_test.drop(columns=[config.target_column])
        progress_bar = st.progress(0)
        for i, (model_name, model) in enumerate(flattened_models.items()):
            self._process_model(writer, model, X_train, X_test, y_train, y_test, 
                                f"flattened_{model_name}", 
                                ensemble_cv_results.get('flattened', {}).get(model_name, []))
            progress_bar.progress((i + 1) / len(flattened_models))

    def _process_ensemble_model(self, writer, ensemble, flattened_X_train, flattened_X_test, 
                                ensemble_cv_results):
        if ensemble is not None:
            st.write("Processing Ensemble Model")
            y_train = flattened_X_train[config.target_column]
            y_test = flattened_X_test[config.target_column]
            X_train = flattened_X_train.drop(columns=[config.target_column])
            X_test = flattened_X_test.drop(columns=[config.target_column])
            self._process_model(writer, ensemble, X_train, X_test, y_train, y_test, 
                                "Ensemble", ensemble_cv_results.get('ensemble', []))

    def _process_model(self, writer, model, X_train, X_test, y_train, y_test, model_name, cv_scores):
        st.write(f"Processing model: {model_name}")
        
        # Ensure X_train and X_test have the same columns as the model expects
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
            X_train = X_train.reindex(columns=expected_features, fill_value=0)
            X_test = X_test.reindex(columns=expected_features, fill_value=0)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_metrics = self.calculate_metrics_func(y_train, y_pred_train)
        test_metrics = self.calculate_metrics_func(y_test, y_pred_test)
        
        sheet_name = self.truncate_sheet_name_func(f"{model_name}_Predictions")
        self._save_predictions(writer, y_train, y_pred_train, y_test, y_pred_test, sheet_name)
        self._save_metrics(writer, train_metrics, test_metrics, sheet_name)
        self._save_cv_results(writer, cv_scores, sheet_name)
        
        # Display metrics and plots in Streamlit
        self._display_metrics(train_metrics, test_metrics, model_name)
        self._plot_predictions(y_train, y_pred_train, y_test, y_pred_test, model_name)
        self._plot_cv_scores(cv_scores, model_name)
        plot_prediction_distribution(y_test, y_pred_test)
        plot_error_distribution(y_test, y_pred_test)

    def _save_predictions(self, writer, y_train, y_pred_train, y_test, y_pred_test, sheet_name):
        pd.DataFrame({
            'y_train': y_train,
            'y_pred_train': y_pred_train,
            'y_test': y_test,
            'y_pred_test': y_pred_test
        }).to_excel(writer, sheet_name=f"{sheet_name}_predictions", index=False)

    def _save_metrics(self, writer, train_metrics, test_metrics, sheet_name):
        pd.DataFrame({
            'Metric': list(train_metrics.keys()) + list(test_metrics.keys()),
            'Train': list(train_metrics.values()) + [np.nan] * len(test_metrics),
            'Test': [np.nan] * len(train_metrics) + list(test_metrics.values())
        }).to_excel(writer, sheet_name=f"{sheet_name}_metrics", index=False)

    def _save_cv_results(self, writer, cv_results, sheet_name):
        if cv_results:
            pd.DataFrame({'CV Score': cv_results}).to_excel(writer, sheet_name=f"{sheet_name}_cv_results", index_label='Fold')

    def _display_metrics(self, train_metrics, test_metrics, model_name):
        st.write(f"Metrics for {model_name}")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Train Metrics")
            display_metrics(train_metrics, title="Training Performance")
        with col2:
            st.write("Test Metrics")
            display_metrics(test_metrics, title="Test Performance")

    def _plot_predictions(self, y_train, y_pred_train, y_test, y_pred_test, model_name):
        st.write(f"Predictions Plot for {model_name}")
        fig = px.scatter()
        fig.add_scatter(x=y_train, y=y_pred_train, mode='markers', name='Train')
        fig.add_scatter(x=y_test, y=y_pred_test, mode='markers', name='Test')
        fig.add_scatter(x=[min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())],
                        y=[min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())],
                        mode='lines', name='Ideal')
        fig.update_layout(title='Actual vs Predicted',
                          xaxis_title='Actual',
                          yaxis_title='Predicted',
                          height=CHART_HEIGHT,
                          width=CHART_WIDTH)
        st.plotly_chart(fig)

    def _plot_cv_scores(self, cv_scores, model_name):
        if cv_scores:
            st.write(f"Cross-Validation Scores for {model_name}")
            fig = px.box(y=cv_scores, labels={'y': 'CV Scores'})
            fig.update_layout(title='Cross-Validation Scores Distribution',
                              height=CHART_HEIGHT,
                              width=CHART_WIDTH)
            st.plotly_chart(fig)
            st.write(f"Mean CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

def make_predictions(model, X_data):
    """Make predictions using a trained model."""
    return model.predict(X_data)

def display_predictions(predictions, actual_values=None):
    """Display predictions with optional comparison to actual values."""
    st.subheader("Predictions")
    create_info_button("predictions")
    df = pd.DataFrame({'Predicted': predictions})
    if actual_values is not None:
        df['Actual'] = actual_values
        df['Error'] = df['Actual'] - df['Predicted']
    st.dataframe(df)
    
    if actual_values is not None:
        plot_prediction_vs_actual(actual_values, predictions)
        plot_residuals(actual_values, predictions)
        plot_prediction_distribution(actual_values, predictions)
        plot_error_distribution(actual_values, predictions)

def predict_for_new_data(models, new_data, cluster_predictors, clustering_config, clustering_2d_config):
    """
    Make predictions for new data using trained models and cluster predictors.
    
    UPDATED: Now works with class-based predictors from Solution 1
    
    Args:
        models: Dictionary of trained models
        new_data: New data to make predictions on
        cluster_predictors: Dictionary of cluster predictor objects (class instances)
        clustering_config: 1D clustering configuration
        clustering_2d_config: 2D clustering configuration
    
    Returns:
        DataFrame with predictions
    """
    st.subheader("Predictions for New Data")
    create_tooltip(TOOLTIPS.get("new_data_predictions", "Making predictions on new data"))
    
    predictions = []
    
    progress_bar = st.progress(0)
    for idx, (index, row) in enumerate(new_data.iterrows()):
        # Use cluster predictors instead of cluster models
        try:
            clusters = predict_cluster_with_predictors(
                row, 
                cluster_predictors, 
                clustering_config, 
                clustering_2d_config
            )
            
            # Combine all cluster labels to form a unique identifier
            cluster_key = "_".join([f"{col}_{label}" for col, label in clusters.items()])
            
            if cluster_key in models:
                model = models[cluster_key]
                try:
                    prediction = make_predictions(model, row.to_frame().T)
                    predictions.append({
                        "Index": index,
                        "Clusters": str(clusters),
                        "Prediction": prediction[0]
                    })
                except Exception as e:
                    st.warning(f"Prediction failed for index {index}: {str(e)}")
                    predictions.append({
                        "Index": index,
                        "Clusters": str(clusters),
                        "Prediction": f"Error: {str(e)}"
                    })
            else:
                st.warning(f"No model found for cluster configuration: {cluster_key}")
                predictions.append({
                    "Index": index,
                    "Clusters": str(clusters),
                    "Prediction": "No model available"
                })
        except Exception as e:
            st.error(f"Cluster prediction failed for index {index}: {str(e)}")
            predictions.append({
                "Index": index,
                "Clusters": "Error in clustering",
                "Prediction": f"Error: {str(e)}"
            })
        
        progress_bar.progress((idx + 1) / len(new_data))
    
    predictions_df = pd.DataFrame(predictions)
    st.write("Predictions:")
    st.dataframe(predictions_df)
    
    return predictions_df


def predict_cluster_with_predictors(data_point, cluster_predictors, clustering_config, clustering_2d_config):
    """
    Predict cluster assignments using cluster predictor objects.
    
    UPDATED: Now works with class-based predictors (NearestCentroidPredictor, 
    KNNClusterPredictor, ModelBasedPredictor) from Solution 1.
    
    Args:
        data_point: Single data point (Series)
        cluster_predictors: Dictionary of predictor objects (class instances)
        clustering_config: 1D clustering configuration
        clustering_2d_config: 2D clustering configuration
    
    Returns:
        Dictionary of cluster assignments
    """
    clusters = {}
    
    # ========================================================================
    # 1D Clustering Prediction
    # ========================================================================
    for column, config_dict in clustering_config.items():
        method = config_dict['method']
        
        if method == 'None':
            clusters[column] = 'label_0'
            
        elif column in cluster_predictors and cluster_predictors[column] is not None:
            predictor = cluster_predictors[column]
            
            # Prepare data for prediction
            if column in data_point.index:
                # Create 2D array for predictor (shape: [1, 1])
                data = np.array([[data_point[column]]])
                
                try:
                    # Call predictor (works with both predictor(data) and predictor.predict(data))
                    # Because we implemented __call__ method in all predictor classes
                    cluster_label = predictor(data)[0]
                    clusters[column] = f'label_{cluster_label}'
                    
                except Exception as e:
                    st.warning(f"Cluster prediction failed for {column}: {str(e)}")
                    debug_print(f"Predictor type: {type(predictor)}, Data shape: {data.shape}")
                    clusters[column] = 'label_0'  # Default to first cluster
            else:
                st.warning(f"Column '{column}' not found in data point")
                clusters[column] = 'unknown'
        else:
            # No predictor available
            clusters[column] = 'label_0'  # Default if no predictor
    
    # ========================================================================
    # 2D Clustering Prediction
    # ========================================================================
    for column_pair, config_dict in clustering_2d_config.items():
        method = config_dict['method']
        
        if method == 'None':
            clusters[column_pair] = 'label_0'
            
        elif column_pair in cluster_predictors and cluster_predictors[column_pair] is not None:
            predictor = cluster_predictors[column_pair]
            
            # Prepare data for prediction
            if all(col in data_point.index for col in column_pair):
                # Create 2D array for predictor (shape: [1, 2])
                data = np.array([[data_point[column_pair[0]], data_point[column_pair[1]]]])
                
                try:
                    # Call predictor (works with both predictor(data) and predictor.predict(data))
                    cluster_label = predictor(data)[0]
                    clusters[column_pair] = f'label_{cluster_label}'
                    
                except Exception as e:
                    st.warning(f"Cluster prediction failed for {column_pair}: {str(e)}")
                    debug_print(f"Predictor type: {type(predictor)}, Data shape: {data.shape}")
                    clusters[column_pair] = 'label_0'
            else:
                missing_cols = [col for col in column_pair if col not in data_point.index]
                st.warning(f"Columns {missing_cols} not found in data point")
                clusters[column_pair] = 'unknown'
        else:
            # No predictor available
            clusters[column_pair] = 'label_0'
    
    return clusters


# ============================================================================
# Legacy function - DEPRECATED (kept for backward compatibility)
# ============================================================================

def predict_cluster(data_point, cluster_models, clustering_config, clustering_2d_config):
    """
    DEPRECATED: This function has a critical bug with DBSCAN.
    Use predict_cluster_with_predictors instead.
    
    This function is kept only for backward compatibility and should NOT be used.
    It will be removed in a future version.
    """
    st.error("⚠️ DEPRECATED FUNCTION CALLED: predict_cluster()")
    st.error("This function has known issues with DBSCAN clustering.")
    st.error("Please use predict_cluster_with_predictors() instead.")
    st.stop()
    
    # Original buggy code commented out to prevent accidental use
    """
    clusters = {}
    
    # 1D Clustering prediction
    for column, config_dict in clustering_config.items():
        method = config_dict['method']
        if method == 'None':
            clusters[column] = 'label_0'
        elif column in cluster_models:
            model = cluster_models[column]
            if isinstance(model, DBSCAN):
                # This will likely fail or return -1 (noise)
                cluster = model.fit_predict(data_point[column].values.reshape(1, -1))
                clusters[column] = f'label_{cluster[0]}'
            elif isinstance(model, KMeans):
                cluster = model.predict(data_point[column].values.reshape(1, -1))
                clusters[column] = f'label_{cluster[0]}'
        else:
            clusters[column] = 'unknown'
    
    # 2D Clustering prediction
    for column_pair, config_dict in clustering_2d_config.items():
        method = config_dict['method']
        if method == 'None':
            clusters[column_pair] = 'label_0'
        elif column_pair in cluster_models:
            model = cluster_models[column_pair]
            data = data_point[list(column_pair)].values.reshape(1, -1)
            if isinstance(model, DBSCAN):
                cluster = model.fit_predict(data)
                clusters[column_pair] = f'label_{cluster[0]}'
            elif isinstance(model, KMeans):
                cluster = model.predict(data)
                clusters[column_pair] = f'label_{cluster[0]}'
        else:
            clusters[column_pair] = 'unknown'
    
    return clusters
    """


def evaluate_predictions(y_true, y_pred):
    """Evaluate predictions and display comprehensive metrics."""
    metrics = calculate_metrics(y_true, y_pred)
    display_metrics(metrics)
    plot_prediction_vs_actual(y_true, y_pred)
    plot_residuals(y_true, y_pred)
    plot_prediction_distribution(y_true, y_pred)
    plot_error_distribution(y_true, y_pred)


def load_saved_models(models_directory):
    """Load all saved models from the specified directory."""
    saved_models = {}
    
    if not os.path.exists(models_directory):
        st.warning(f"Models directory {models_directory} does not exist.")
        return saved_models
    
    for filename in os.listdir(models_directory):
        if filename.endswith('.joblib') and not filename.startswith('preprocessor'):
            model_path = os.path.join(models_directory, filename)
            model_name = filename.replace('.joblib', '')
            try:
                saved_models[model_name] = joblib.load(model_path)
            except Exception as e:
                st.warning(f"Failed to load {filename}: {str(e)}")
    
    return saved_models