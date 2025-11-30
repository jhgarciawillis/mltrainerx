import os
import uuid
import json
import copy
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths and names
DATA_PATH = os.path.join(BASE_DIR, "Data.xlsx")
PREDICTIONS_PATH = os.path.join(BASE_DIR, "Predictions.xlsx")
MODELS_DIRECTORY = os.path.join(BASE_DIR, "Trained")
CLUSTERS_PATH = os.path.join(BASE_DIR, "Clusters.joblib")
GLOBAL_STATS_PATH = os.path.join(BASE_DIR, "global_stats.joblib")

# Sheet name configuration
SHEET_NAME_MAX_LENGTH = 31
TRUNCATE_SHEET_NAME_REPLACEMENT = "_cluster_db"

# Data processing parameters
OUTLIER_THRESHOLD = 3
RANDOM_STATE = 42

# Clustering parameters
AVAILABLE_CLUSTERING_METHODS = ['None', 'DBSCAN', 'KMeans']
DEFAULT_CLUSTERING_METHOD = 'None'
DBSCAN_PARAMETERS = {
    'eps': 0.5,
    'min_samples': 5
}
KMEANS_PARAMETERS = {
    'n_clusters': 5
}

# Feature engineering parameters
STATISTICAL_AGG_FUNCTIONS = ['mean', 'median', 'std']
TOP_K_FEATURES = 20
MAX_INTERACTION_DEGREE = 2
POLYNOMIAL_DEGREE = 2
FEATURE_SELECTION_SCORE_FUNC = 'f_regression'

# Model configurations
MODEL_CLASSES = {
    'rf': RandomForestRegressor,
    'xgb': XGBRegressor,
    'lgbm': LGBMRegressor,
    'ada': AdaBoostRegressor,
    'catboost': CatBoostRegressor,
    'knn': KNeighborsRegressor
}

# Hyperparameter grids
HYPERPARAMETER_GRIDS = {
    'rf': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    },
    'xgb': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 9]
    },
    'lgbm': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 62, 124]
    },
    'ada': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    },
    'catboost': {
        'iterations': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'depth': [4, 6, 8]
    },
    'knn': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }
}

# Model training parameters
MODEL_CV_SPLITS = 5
RANDOMIZED_SEARCH_ITERATIONS = 10
ENSEMBLE_CV_SPLITS = 10
ENSEMBLE_CV_SHUFFLE = True

# Streamlit configurations
STREAMLIT_THEME = {
    'primaryColor': '#FF4B4B',
    'backgroundColor': '#FFFFFF',
    'secondaryBackgroundColor': '#F0F2F6',
    'textColor': '#262730',
    'font': 'sans serif'
}

# Streamlit configurations
STREAMLIT_APP_NAME = 'ML Algo Trainer'
STREAMLIT_APP_ICON = '🧠'

# File upload configurations
ALLOWED_EXTENSIONS = ['csv', 'xlsx']
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB

# Visualization configurations
MAX_ROWS_TO_DISPLAY = 100
CHART_HEIGHT = 400
CHART_WIDTH = 600

# Tooltips for UI elements
TOOLTIPS = {
    "file_upload": "Upload your dataset in CSV or Excel format.",
    "sheet_selection": "Select the sheet containing your data (for Excel files).",
    "train_test_split": "Set the proportion of data to use for training. The rest will be used for testing.",
    "use_clustering": "Enable clustering to group similar data points before training models.",
    "models_to_use": "Select one or more machine learning models to train on your data.",
    "tuning_method": "Choose a method for optimizing model hyperparameters.",
    "clustering_method": "Select a clustering algorithm to apply to this column.",
    "dbscan_eps": "Set the maximum distance between two samples for them to be considered as in the same neighborhood.",
    "dbscan_min_samples": "Set the number of samples in a neighborhood for a point to be considered as a core point.",
    "kmeans_n_clusters": "Set the number of clusters to form and centroids to generate.",
    "2d_clustering": "Select pairs of columns for two-dimensional clustering.",
    "use_saved_models": "Choose whether to use previously saved models or upload new ones.",
    "upload_models": "Upload trained model files (in joblib format).",
    "upload_preprocessor": "Upload the preprocessor used for data transformation.",
    "new_data_file": "Upload a CSV file containing new data for prediction.",
    "auto_detect_column_types": "Automatically detect the column types in the dataset.",
    "manual_column_selection": "Manually select the column types for your dataset.",
    "handle_missing_values": "Choose a strategy to handle missing values in your dataset.",
    "outlier_removal": "Identify and remove data points that significantly differ from other observations.",
    "polynomial_features": "Generate polynomial features to capture non-linear relationships.",
    "interaction_terms": "Create interaction features between numerical and categorical columns.",
    "statistical_features": "Calculate statistical features like mean, median, and standard deviation.",
    "random_state": "Set the random state for reproducibility of the machine learning models.",
    "cv_folds": "Specify the number of cross-validation folds to use during model training.",
    "model_metrics": "Display various performance metrics for the trained model.",
    "feature_correlations": "Show correlations between features and the target variable.",
    "new_data_predictions": "Make predictions on new data using the trained models.",
    "hyperparameter_tuning": "Optimize model hyperparameters to improve performance.",
    "feature_selection": "Select the most relevant features for model training.",
    "randomize_rows": "Shuffle the order of rows to prevent potential ordering bias in your dataset",
    "randomization_strategy": """Choose how to randomize your data:
    
    • Together (Recommended): Safest method - shuffles features and target together
    • Stratified: Preserves class distribution - use for imbalanced datasets
    • None in split: Uses sklearn's built-in shuffle - also safe
    
    Why randomize?
    - Prevents temporal bias (if data is sorted by time)
    - Ensures train and test sets are representative
    - Improves model generalization
    
    When NOT to randomize:
    - Time series data where order matters
    - Sequential experiments where order is meaningful
    - Pre-randomized datasets""",
    "random_seed": """Random seed controls reproducibility:
    
    • Same seed = identical train/test split every time
    • Different seed = different splits
    • Use 42 (common convention) or your lucky number!
    
    Why it matters:
    - Allows you to compare model changes fairly
    - Required for scientific reproducibility
    - Helps debug issues by keeping data constant""",
    "categorical_encoding": """Different strategies for encoding categorical variables:
    
    • OneHot: Creates binary column for each unique value
      - Pros: Standard, interpretable, works well with most models
      - Cons: Explodes with high cardinality (many unique values)
      - Use when: <50 unique values
    
    • Target Encoding: Replaces values with mean of target
      - Pros: Reduces dimensions dramatically
      - Cons: Risk of overfitting, requires target variable
      - Use when: High cardinality + tree-based models
    
    • Frequency Encoding: Replaces values with their frequency
      - Pros: Simple, no overfitting risk, preserves cardinality info
      - Cons: Loses category identity
      - Use when: Frequency matters more than identity
    
    • Hash Encoding: Hashes values into fixed dimensions
      - Pros: Fixed size regardless of cardinality
      - Cons: Loses interpretability, potential collisions
      - Use when: Very high cardinality (>100 values)
    
    • Drop: Excludes the column
      - Use when: Column has no predictive value or too messy""",
    "high_cardinality": """High cardinality = many unique values in a categorical column.
    
    Examples:
    - ZIP codes (>40,000 values)
    - Product IDs (could be millions)
    - User IDs (unlimited)
    
    Why it's a problem:
    - OneHot encoding ZIP codes = 40,000+ new columns
    - Slows down training dramatically
    - Increases memory usage
    - Can cause overfitting
    
    Solutions:
    1. Use alternative encoding (target, frequency, hash)
    2. Group rare categories into "Other"
    3. Extract meaningful features (first 3 digits of ZIP)
    4. Drop the column if not predictive"""
}

# Detailed information for UI sections
INFO_TEXTS = {
    "data_input": "This section allows you to upload your dataset and configure initial settings. You can upload CSV or Excel files and select specific sheets for Excel files.",
    "data_preprocessing": "Data preprocessing involves cleaning and transforming raw data into a format suitable for machine learning models. This includes handling missing values, encoding categorical variables, and scaling numerical features.",
    "feature_engineering": "Feature engineering is the process of using domain knowledge to create new features or transform existing ones. This can improve model performance by providing more relevant information to the algorithms.",
    "clustering_configuration": "Clustering is an unsupervised learning technique that groups similar data points together. It can be used to segment your data before applying regression models, potentially improving overall performance.",
    "model_selection_training": "In this section, you can choose which machine learning models to train on your data. You can also select a method for tuning the hyperparameters of these models to optimize their performance.",
    "advanced_options": "Advanced options allow you to fine-tune various aspects of the machine learning pipeline, such as the random state for reproducibility and the number of cross-validation folds.",
    "outlier_removal": "Outlier removal is the process of identifying and removing data points that significantly differ from other observations. This can improve model performance by reducing the impact of anomalous data.",
    "load_saved_models": "Loading saved models allows you to use previously trained models for making predictions on new data without having to retrain them.",
    "upload_prediction_data": "Upload new data on which you want to make predictions using your trained models.",
    "make_predictions": "Use your trained models to generate predictions for the new data you've uploaded.",
    "manual_column_selection": "Manually select the column types for your dataset. This allows you to specify which columns should be treated as numerical, categorical, target, or unused.",
    "handle_missing_values": "Choose a strategy to handle missing values in your dataset. This can include dropping rows with missing values, filling with the mean/mode, or filling with the median.",
    "polynomial_features": "Generating polynomial features can capture non-linear relationships in your data, potentially improving model performance.",
    "interaction_terms": "Creating interaction features between numerical and categorical columns can help the model learn more complex patterns in the data.",
    "statistical_features": "Calculating statistical features like mean, median, and standard deviation can provide additional information to the machine learning models.",
    "random_state": "Setting the random state ensures reproducibility of the machine learning models, so that the results can be replicated.",
    "cv_folds": "The number of cross-validation folds determines how the dataset is split during model training and validation, which can affect the model's generalization performance.",
    "data_information": "This section provides an overview of your dataset, including the number of rows and columns, and a preview of the data.",
    "column_selection": "In this step, you can manually assign roles to each column in your dataset, such as numerical features, categorical features, target variable, or unused columns.",
    "feature_generation": "This step involves creating new features based on existing ones to capture more complex patterns in the data.",
    "combine_features": "Combine the original features with the newly generated ones to create a richer dataset for model training.",
    "prediction_features": "Generate features for the prediction dataset using the same methods applied during training.",
    "feature_info": "Display information about the features used in the model, including their names and importance.",
    "feature_correlations": "Visualize the correlations between features and the target variable to understand their relationships.",
    "feature_distributions": "View the distributions of individual features to understand their characteristics and potential outliers.",
    "feature_interactions": "Explore how different features interact with each other and their impact on the target variable.",
    "model_training_validation": "Train multiple machine learning models on your data and validate their performance using cross-validation.",
    "flattened_data_training": "Train models on the entire dataset without clustering to compare performance with cluster-specific models.",
    "ensemble_model": "Create an ensemble model that combines predictions from multiple individual models to improve overall performance.",
    "model_performance": "Evaluate the performance of trained models using various metrics and visualizations.",
    "model_evaluation": "Assess the model's performance using metrics, residual plots, and feature importance analysis.",
    "model_comparison": "Compare the performance of different models side by side to determine the best one for your data.",
    "cluster_metrics": "View performance metrics for each cluster to understand how well the models perform on different segments of your data.",
    "predictions_file": "Generate a file containing predictions from all trained models, including individual model predictions and ensemble predictions.",
    "hyperparameter_tuning": "Optimize model hyperparameters to improve performance using techniques like grid search or random search.",
    "model_metrics": "View various performance metrics for the trained models to assess their accuracy and generalization ability."
}


class Config:
    """Enhanced configuration class with session management and validation."""
    
    def __init__(self):
        # Session management
        self.session_id = str(uuid.uuid4())
        self._created_at = pd.Timestamp.now()
        
        # File paths
        self.file_path = None
        self.sheet_name = None
        
        # Column configuration
        self.target_column = None
        self.numerical_columns = []
        self.categorical_columns = []
        self.unused_columns = []
        self.all_columns = []
        
        # Clustering configuration
        self.use_clustering = False
        self.clustering_config = {}
        self.clustering_2d_config = {}
        self.clustering_2d_columns = []
        
        # Training configuration
        self.train_size = 0.8
        self.models_to_use = []
        self.tuning_method = 'None'
        
        # Feature engineering
        self.use_polynomial_features = True
        self.use_interaction_terms = True
        self.use_statistical_features = True
        
        # Prediction configuration
        self.use_saved_models = 'Yes'
        self.uploaded_models = None
        self.uploaded_preprocessor = None
        self.new_data_file = None
        
        # Outlier removal configuration
        self.outlier_removal_columns = []
        self.remove_target_outliers = False
        self.target_outlier_method = 'none'
        self.OUTLIER_THRESHOLD = 3.0
        
        # System configuration
        self.STREAMLIT_APP_NAME = STREAMLIT_APP_NAME
        self.MODELS_DIRECTORY = MODELS_DIRECTORY
        self.RANDOM_STATE = RANDOM_STATE
        self.MODEL_CV_SPLITS = MODEL_CV_SPLITS
        self.MODEL_CLASSES = MODEL_CLASSES

    def update(self, **kwargs):
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Update all_columns to include all columns
        self.all_columns = list(set(
            self.numerical_columns + 
            self.categorical_columns + 
            ([self.target_column] if self.target_column else []) + 
            self.unused_columns
        ))

    def set_column_types(self, numerical, categorical, unused, target):
        """Set column types and track all columns."""
        all_columns = numerical + categorical + unused + [target]
        
        self.numerical_columns = numerical
        self.categorical_columns = categorical
        self.unused_columns = unused
        self.target_column = target
        self.all_columns = list(set(all_columns))

    def set_2d_clustering(self, column_pairs, method, params):
        """Configure 2D clustering for column pairs."""
        for pair in column_pairs:
            self.clustering_2d_config[pair] = {'method': method, 'params': params}

    def set_2d_clustering_columns(self, columns):
        """Set columns selected for 2D clustering."""
        self.clustering_2d_columns = columns

    def update_outlier_removal_columns(self, columns):
        """Update the list of columns to remove outliers from."""
        self.outlier_removal_columns = columns
    
    def copy(self):
        """Create a deep copy of the configuration."""
        return copy.deepcopy(self)
    
    def validate(self):
        """Validate configuration before training."""
        errors = []
        
        if not self.target_column:
            errors.append("Target column not set")
        
        if not self.numerical_columns and not self.categorical_columns:
            errors.append("No feature columns selected")
        
        if not self.models_to_use:
            errors.append("No models selected for training")
        
        if self.train_size <= 0 or self.train_size >= 1:
            errors.append(f"Invalid train_size: {self.train_size} (must be between 0 and 1)")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"- {e}" for e in errors))
        
        return True
    
    def to_dict(self):
        """Convert config to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'created_at': str(self._created_at),
            'file_path': str(self.file_path) if self.file_path else None,
            'sheet_name': self.sheet_name,
            'target_column': self.target_column,
            'numerical_columns': self.numerical_columns,
            'categorical_columns': self.categorical_columns,
            'unused_columns': self.unused_columns,
            'use_clustering': self.use_clustering,
            'clustering_config': self.clustering_config,
            'clustering_2d_config': {str(k): v for k, v in self.clustering_2d_config.items()},
            'train_size': self.train_size,
            'models_to_use': self.models_to_use,
            'tuning_method': self.tuning_method,
            'use_polynomial_features': self.use_polynomial_features,
            'use_interaction_terms': self.use_interaction_terms,
            'use_statistical_features': self.use_statistical_features,
            'outlier_removal_columns': self.outlier_removal_columns,
            'remove_target_outliers': self.remove_target_outliers,
            'target_outlier_method': self.target_outlier_method,
            'OUTLIER_THRESHOLD': self.OUTLIER_THRESHOLD,
            'RANDOM_STATE': self.RANDOM_STATE,
            'MODEL_CV_SPLITS': self.MODEL_CV_SPLITS,
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Restore config from dictionary."""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key) and key not in ['session_id', 'created_at']:
                setattr(config, key, value)
        return config


# Initialize global configuration (for backward compatibility)
<<<<<<< HEAD
config = Config()
=======
config = Config()
>>>>>>> 570c991005b0ff9e859b21570521046c17a61fb8
