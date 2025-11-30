import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

from _0config import config, CLUSTERS_PATH, MODELS_DIRECTORY, CHART_HEIGHT, CHART_WIDTH
from _2misc_utils import debug_print

# ============================================================================
# SOLUTION 1: Picklable Predictor Classes (Module-Level)
# ============================================================================

class NearestCentroidPredictor:
    """
    Picklable predictor using nearest centroid method.
    
    This class can be serialized with joblib/pickle because it's defined at module level
    and doesn't use closures.
    """
    def __init__(self, centroids):
        """
        Initialize predictor with cluster centroids.
        
        Args:
            centroids: Dictionary mapping cluster labels to centroid coordinates
        """
        self.centroids = centroids
    
    def predict(self, X_new):
        """
        Predict cluster labels for new data points.
        
        Args:
            X_new: Array-like of shape (n_samples, n_features)
        
        Returns:
            Array of predicted cluster labels
        """
        predictions = []
        for row in X_new:
            # Calculate distances to all centroids
            distances = {
                label: np.linalg.norm(row - centroid) 
                for label, centroid in self.centroids.items() 
                if centroid is not None
            }
            
            if distances:
                # Assign to nearest centroid
                predictions.append(min(distances, key=distances.get))
            else:
                # Fallback to cluster 0
                predictions.append(0)
        
        return np.array(predictions)
    
    def __call__(self, X_new):
        """Allow predictor to be called like a function."""
        return self.predict(X_new)
    
    def __repr__(self):
        return f"NearestCentroidPredictor(n_centroids={len(self.centroids)})"


class KNNClusterPredictor:
    """
    Picklable predictor using K-Nearest Neighbors method.
    
    Uses trained KNN classifier to assign new points based on proximity
    to training points in each cluster.
    """
    def __init__(self, knn_model):
        """
        Initialize predictor with trained KNN model.
        
        Args:
            knn_model: Fitted KNeighborsClassifier instance
        """
        self.knn_model = knn_model
    
    def predict(self, X_new):
        """
        Predict cluster labels using KNN.
        
        Args:
            X_new: Array-like of shape (n_samples, n_features)
        
        Returns:
            Array of predicted cluster labels
        """
        return self.knn_model.predict(X_new)
    
    def __call__(self, X_new):
        """Allow predictor to be called like a function."""
        return self.predict(X_new)
    
    def __repr__(self):
        return f"KNNClusterPredictor(n_neighbors={self.knn_model.n_neighbors})"


class ModelBasedPredictor:
    """
    Picklable predictor using clustering model's built-in predict method.
    
    Works with KMeans and other clustering algorithms that have a predict() method.
    """
    def __init__(self, model):
        """
        Initialize predictor with clustering model.
        
        Args:
            model: Fitted clustering model with predict() method (e.g., KMeans)
        """
        self.model = model
    
    def predict(self, X_new):
        """
        Predict cluster labels using model's built-in predict.
        
        Args:
            X_new: Array-like of shape (n_samples, n_features)
        
        Returns:
            Array of predicted cluster labels
        """
        return self.model.predict(X_new)
    
    def __call__(self, X_new):
        """Allow predictor to be called like a function."""
        return self.predict(X_new)
    
    def __repr__(self):
        return f"ModelBasedPredictor(model={type(self.model).__name__})"


# ============================================================================
# Updated create_cluster_predictor - Returns Class Instances
# ============================================================================

def create_cluster_predictor(cluster_model, X_train, cluster_labels, method='nearest_centroid'):
    """
    Create a picklable predictor for assigning new points to existing clusters.
    
    CRITICAL FIX: Returns class instances instead of closures, making them picklable.
    
    Args:
        cluster_model: Fitted clustering model (DBSCAN or KMeans)
        X_train: Training data used to create clusters (numpy array)
        cluster_labels: Cluster labels from training (numpy array)
        method: Prediction method - 'nearest_centroid', 'knn', or 'model_based'
    
    Returns:
        Picklable predictor object (instance of NearestCentroidPredictor, 
        KNNClusterPredictor, or ModelBasedPredictor)
    """
    
    # Method 1: Use model's built-in predict (only for KMeans)
    if method == 'model_based' and hasattr(cluster_model, 'predict'):
        return ModelBasedPredictor(cluster_model)
    
    # Method 2: Nearest Centroid (works for all clustering methods)
    elif method == 'nearest_centroid':
        # Calculate centroids for each cluster (excluding noise points)
        unique_labels = np.unique(cluster_labels[cluster_labels != -1])
        centroids = {}
        
        for label in unique_labels:
            mask = cluster_labels == label
            if mask.sum() > 0:  # Ensure cluster has points
                centroids[label] = X_train[mask].mean(axis=0)
        
        # Handle noise points (label -1 from DBSCAN)
        if -1 in cluster_labels:
            noise_mask = cluster_labels == -1
            if noise_mask.sum() > 0:
                centroids[-1] = X_train[noise_mask].mean(axis=0)
            else:
                centroids[-1] = None
        
        # Ensure we have at least one centroid
        if not centroids:
            st.warning("No valid clusters found. Creating default centroid.")
            centroids[0] = np.zeros(X_train.shape[1])
        
        return NearestCentroidPredictor(centroids)
    
    # Method 3: K-Nearest Neighbors (most accurate for DBSCAN)
    elif method == 'knn':
        # Remove noise points for training
        mask = cluster_labels != -1
        X_train_clean = X_train[mask]
        labels_clean = cluster_labels[mask]
        
        if len(labels_clean) > 0:
            # Determine appropriate number of neighbors
            n_neighbors = min(5, len(labels_clean))
            
            # Train KNN classifier
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn.fit(X_train_clean, labels_clean)
            
            return KNNClusterPredictor(knn)
        else:
            # Fallback if no valid training points
            st.warning("No valid training points for KNN. Falling back to nearest centroid.")
            centroids = {0: np.zeros(X_train.shape[1])}
            return NearestCentroidPredictor(centroids)
    
    # Default fallback
    else:
        st.warning(f"Unknown prediction method '{method}'. Using nearest centroid.")
        centroids = {0: X_train.mean(axis=0)}
        return NearestCentroidPredictor(centroids)


# ============================================================================
# Main create_clusters Function (Updated)
# ============================================================================

def create_clusters(preprocessed_data, clustering_config, clustering_2d_config, 
                    target_column, prediction_method='auto'):
    """
    Create clusters with prediction capability.
    
    Args:
        preprocessed_data: Preprocessed DataFrame
        clustering_config: 1D clustering configuration
        clustering_2d_config: 2D clustering configuration  
        target_column: Target column name
        prediction_method: 'auto', 'nearest_centroid', 'knn', or 'model_based'
    
    Returns:
        clustered_data: Dictionary of cluster assignments
    """
    st.write("Creating clusters...")
    debug_print("Entering create_clusters function.")
    
    clustered_data = {}
    cluster_models = {}
    cluster_predictors = {}  # Now stores class instances instead of closures
    
    all_columns = [col for col in preprocessed_data.columns if col != target_column]
    
    # User control for prediction method
    if prediction_method == 'auto':
        st.subheader("Cluster Prediction Method")
        prediction_method = st.selectbox(
            "How should new data be assigned to clusters?",
            options=['nearest_centroid', 'knn', 'model_based'],
            format_func=lambda x: {
                'nearest_centroid': 'Nearest Centroid (Fast, works for any clustering)',
                'knn': 'K-Nearest Neighbors (More accurate, slower)',
                'model_based': 'Model\'s built-in (Only for KMeans)'
            }[x],
            help="""
            DBSCAN doesn't have a predict() method, so we need an alternative:
            
            • Nearest Centroid: Assign to cluster with closest center (fast)
            • KNN: Use K-nearest neighbors from training data (more accurate)
            • Model-based: Use KMeans' built-in predict (only works for KMeans)
            """
        )
    
    # 1D Clustering
    for column in all_columns:
        method = clustering_config.get(column, {}).get('method', 'None')
        params = clustering_config.get(column, {}).get('params', {})
        
        if method == 'None':
            clustered_data[column] = {'label_0': preprocessed_data.index.tolist()}
            cluster_predictors[column] = None
        elif method in ['DBSCAN', 'KMeans']:
            data_for_clustering = preprocessed_data[column].values.reshape(-1, 1)
            
            clustered_data[column], cluster_models[column] = perform_clustering_with_options(
                preprocessed_data[column], 
                method,
                params,
                data_name=column
            )
            
            # Create predictor for this cluster (NOW RETURNS CLASS INSTANCE)
            predictor_method = prediction_method if method == 'DBSCAN' else 'model_based'
            
            # Get cluster labels
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(data_for_clustering)
            cluster_labels = cluster_models[column].labels_
            
            predictor = create_cluster_predictor(
                cluster_models[column],
                X_scaled,
                cluster_labels,
                method=predictor_method
            )
            cluster_predictors[column] = predictor
            
        visualize_1d_clusters(preprocessed_data[column], clustered_data[column], method, column)
    
    # 2D Clustering
    for column_pair, config_dict in clustering_2d_config.items():
        if set(column_pair).issubset(set(preprocessed_data.columns)):
            method = config_dict['method']
            params = config_dict['params']
            
            if method == 'None':
                clustered_data[column_pair] = {'label_0': preprocessed_data.index.tolist()}
                cluster_predictors[column_pair] = None
            else:
                data_for_clustering = preprocessed_data[list(column_pair)].values
                
                clustered_data[column_pair], cluster_models[column_pair] = perform_clustering_with_options(
                    preprocessed_data[list(column_pair)],
                    method,
                    params,
                    data_name=f"{column_pair[0]}_{column_pair[1]}"
                )
                
                # Create predictor (NOW RETURNS CLASS INSTANCE)
                predictor_method = prediction_method if method == 'DBSCAN' else 'model_based'
                
                # Get cluster labels
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(data_for_clustering)
                cluster_labels = cluster_models[column_pair].labels_
                
                predictor = create_cluster_predictor(
                    cluster_models[column_pair],
                    X_scaled,
                    cluster_labels,
                    method=predictor_method
                )
                cluster_predictors[column_pair] = predictor
            
            visualize_2d_clusters(preprocessed_data[list(column_pair)], 
                               clustered_data[column_pair], 
                               method, 
                               column_pair)
        else:
            st.warning(f"Skipping 2D clustering for {column_pair} as columns are not present in the data.")
    
    # Save cluster models AND predictors
    save_clustering_models(cluster_models, MODELS_DIRECTORY)
    save_clustering_predictors(cluster_predictors, MODELS_DIRECTORY)
    
    return clustered_data


def perform_clustering_with_options(data, method, base_params, data_name):
    """Perform clustering with enhanced options and parameters."""
    st.write(f"Performing {method} clustering for {data_name}")
    
    if method == 'DBSCAN':
        # Get base parameters
        eps = base_params.get('eps', 0.5)
        min_samples = base_params.get('min_samples', 5)
        
        # Additional options
        col1, col2, col3 = st.columns(3)
        with col1:
            distance_metric = st.selectbox(
                "Distance metric",
                ['euclidean', 'manhattan', 'cosine', 'chebyshev'],
                key=f"distance_{data_name}"
            )
        with col2:
            leaf_size = st.slider(
                "Leaf size",
                min_value=10,
                max_value=100,
                value=30,
                key=f"leaf_{data_name}"
            )
        with col3:
            algorithm = st.selectbox(
                "Algorithm",
                ['auto', 'ball_tree', 'kd_tree', 'brute'],
                key=f"algo_{data_name}"
            )
        
        # Create DBSCAN model with all parameters
        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=distance_metric,
            leaf_size=leaf_size,
            algorithm=algorithm
        )
        
    elif method == 'KMeans':
        # Get base parameters
        n_clusters = base_params.get('n_clusters', 5)
        
        # Additional options
        col1, col2, col3 = st.columns(3)
        with col1:
            init_method = st.selectbox(
                "Initialization method",
                ['k-means++', 'random'],
                key=f"init_{data_name}"
            )
        with col2:
            n_init = st.slider(
                "Number of initializations",
                min_value=1,
                max_value=20,
                value=10,
                key=f"ninit_{data_name}"
            )
        with col3:
            max_iter = st.slider(
                "Maximum iterations",
                min_value=100,
                max_value=1000,
                value=300,
                key=f"maxiter_{data_name}"
            )
        
        # Create KMeans model with all parameters
        model = KMeans(
            n_clusters=n_clusters,
            init=init_method,
            n_init=n_init,
            max_iter=max_iter,
            random_state=config.RANDOM_STATE
        )
    
    # Reshape data for clustering if needed
    X = data.values.reshape(-1, 1) if len(data.shape) == 1 else data.values
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit model and get cluster labels
    cluster_labels = model.fit_predict(X_scaled)
    
    # Create clusters dictionary
    unique_labels = np.unique(cluster_labels)
    clusters = {f'label_{label}': data.index[cluster_labels == label].tolist() 
               for label in unique_labels}
    
    # Display clustering results
    st.write(f"Number of clusters found: {len(unique_labels)}")
    for label, indices in clusters.items():
        st.write(f"- {label}: {len(indices)} points")
    
    return clusters, model


def visualize_1d_clusters(data, clusters, method, column):
    """Enhanced visualization of 1D clustering results."""
    st.write(f"Visualization of 1D clustering for {column}")
    
    # Allow user to select visualization type
    viz_type = st.selectbox(
        "Select visualization type",
        ["Histogram", "Box Plot", "Violin Plot"],
        key=f"viz_1d_{column}"
    )
    
    df = pd.DataFrame({column: data, 'Cluster': pd.Series(dtype='str')})
    for label, indices in clusters.items():
        df.loc[indices, 'Cluster'] = label
    
    if viz_type == "Histogram":
        fig = px.histogram(
            df, 
            x=column, 
            color='Cluster',
            title=f"{method} Clustering for {column}",
            barmode='overlay',
            opacity=0.7
        )
    elif viz_type == "Box Plot":
        fig = px.box(
            df,
            x='Cluster',
            y=column,
            title=f"{method} Clustering for {column}",
            points="all"
        )
    else:  # Violin Plot
        fig = px.violin(
            df,
            x='Cluster',
            y=column,
            title=f"{method} Clustering for {column}",
            box=True,
            points="all"
        )
    
    fig.update_layout(
        height=CHART_HEIGHT,
        width=CHART_WIDTH
    )
    st.plotly_chart(fig)


def visualize_2d_clusters(data, clusters, method, column_pair):
    """Enhanced visualization of 2D clustering results."""
    st.write(f"Visualization of 2D clustering for {column_pair}")
    
    # Allow user to select visualization type
    viz_type = st.selectbox(
        "Select visualization type",
        ["Scatter", "Density Contour", "Density Heatmap"],
        key=f"viz_2d_{column_pair[0]}_{column_pair[1]}"
    )
    
    df = pd.DataFrame(data)
    df['Cluster'] = pd.Series(dtype='str')
    for label, indices in clusters.items():
        df.loc[indices, 'Cluster'] = label
    
    if viz_type == "Scatter":
        fig = px.scatter(
            df, 
            x=column_pair[0], 
            y=column_pair[1],
            color='Cluster',
            title=f"{method} Clustering for {column_pair[0]} vs {column_pair[1]}"
        )
    elif viz_type == "Density Contour":
        fig = px.density_contour(
            df,
            x=column_pair[0],
            y=column_pair[1],
            color='Cluster',
            title=f"{method} Clustering for {column_pair[0]} vs {column_pair[1]}"
        )
    else:  # Density Heatmap
        fig = px.density_heatmap(
            df,
            x=column_pair[0],
            y=column_pair[1],
            title=f"{method} Clustering for {column_pair[0]} vs {column_pair[1]}",
            marginal_x="histogram",
            marginal_y="histogram"
        )
    
    fig.update_layout(
        height=CHART_HEIGHT,
        width=CHART_WIDTH
    )
    st.plotly_chart(fig)


def generate_2d_cluster_filename(column_pair, method):
    """Generate filename for 2D clustering model."""
    col1, col2 = column_pair
    return f"2D_{col1[:3]}_{col2[:3]}_{method}.joblib"


def save_clustering_models(cluster_models, save_path):
    """Save clustering models to the specified directory."""
    os.makedirs(save_path, exist_ok=True)
    for column, model in cluster_models.items():
        if isinstance(column, tuple):  # 2D clustering
            filename = generate_2d_cluster_filename(column, model.__class__.__name__)
        else:  # 1D clustering
            filename = f"cluster_model_{column}.joblib"
        model_path = os.path.join(save_path, filename)
        joblib.dump(model, model_path)
    st.success("Clustering models saved successfully.")


# ============================================================================
# SOLUTION 2: Save/Load Functions (Updated)
# ============================================================================

def save_clustering_predictors(cluster_predictors, save_path):
    """
    Save cluster predictors to disk.
    
    Now works because predictors are picklable class instances (Solution 1).
    
    Args:
        cluster_predictors: Dictionary of predictor objects (class instances)
        save_path: Directory to save predictors
    """
    os.makedirs(save_path, exist_ok=True)
    
    predictor_path = os.path.join(save_path, 'cluster_predictors.joblib')
    
    try:
        # This now works because we're saving class instances, not closures
        joblib.dump(cluster_predictors, predictor_path)
        st.success(f"✅ Cluster predictors saved successfully to: {predictor_path}")
        
        # Log what was saved for debugging
        predictor_types = {
            key: type(pred).__name__ if pred is not None else 'None'
            for key, pred in cluster_predictors.items()
        }
        debug_print(f"Saved predictors: {predictor_types}")
        
    except Exception as e:
        st.error(f"❌ Failed to save cluster predictors: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        
        # Attempt to save diagnostic information
        try:
            diagnostic_info = {
                'error': str(e),
                'predictor_keys': list(cluster_predictors.keys()),
                'predictor_types': {
                    key: type(pred).__name__ if pred is not None else 'None'
                    for key, pred in cluster_predictors.items()
                }
            }
            diagnostic_path = os.path.join(save_path, 'cluster_predictors_diagnostic.joblib')
            joblib.dump(diagnostic_info, diagnostic_path)
            st.warning(f"Diagnostic information saved to: {diagnostic_path}")
        except:
            pass


def load_clustering_predictors(models_directory):
    """
    Load cluster predictors from disk.
    
    Args:
        models_directory: Directory containing saved predictors
    
    Returns:
        Dictionary of predictor objects, or empty dict if loading fails
    """
    predictor_path = os.path.join(models_directory, 'cluster_predictors.joblib')
    
    if not os.path.exists(predictor_path):
        st.warning("⚠️ No cluster predictors found. Prediction may fail for DBSCAN clusters.")
        st.info("Predictors are created during training when clustering is enabled.")
        return {}
    
    try:
        predictors = joblib.load(predictor_path)
        st.success(f"✅ Cluster predictors loaded successfully from: {predictor_path}")
        
        # Log what was loaded for debugging
        predictor_types = {
            key: type(pred).__name__ if pred is not None else 'None'
            for key, pred in predictors.items()
        }
        debug_print(f"Loaded predictors: {predictor_types}")
        
        return predictors
        
    except Exception as e:
        st.error(f"❌ Failed to load cluster predictors: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        st.warning("Prediction mode will have limited functionality.")
        
        # Try to load diagnostic information if available
        try:
            diagnostic_path = os.path.join(models_directory, 'cluster_predictors_diagnostic.joblib')
            if os.path.exists(diagnostic_path):
                diagnostic_info = joblib.load(diagnostic_path)
                st.info("Diagnostic information found:")
                st.json(diagnostic_info)
        except:
            pass
        
        return {}


def load_clustering_models(models_directory):
    """Load clustering models from the specified directory."""
    cluster_models = {}
    if not os.path.exists(models_directory):
        return cluster_models
        
    for filename in os.listdir(models_directory):
        if filename.startswith('cluster_model_') or filename.startswith('2D_'):
            model_path = os.path.join(models_directory, filename)
            if filename.startswith('2D_'):
                # Extract column names from 2D cluster filename
                parts = filename.split('_')
                col1, col2 = parts[1], parts[2]
                column = (col1, col2)
            else:
                column = filename.replace('cluster_model_', '').replace('.joblib', '')
            cluster_models[column] = joblib.load(model_path)
    return cluster_models