import os
import joblib
import pandas as pd
import numpy as np
import traceback
import streamlit as st
import plotly.graph_objects as go

from scipy.stats import zscore
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from _0config import MODELS_DIRECTORY, OUTLIER_THRESHOLD, RANDOM_STATE, CHART_HEIGHT, CHART_WIDTH
from _2misc_utils import debug_print, flatten_clustered_data, identify_high_correlation_features

def load_and_preprocess_data(data, config):
    """Preprocess data with outlier removal."""
    debug_print("Starting data preprocessing...")
    
    # Feature outlier removal
    if config.outlier_removal_columns:
        preprocessed_data = remove_outliers(
            data, 
            config.outlier_removal_columns, 
            config.OUTLIER_THRESHOLD,
            target_column=config.target_column,
            remove_target_outliers=config.remove_target_outliers,
            target_method=config.target_outlier_method
        )
    else:
        preprocessed_data = data.copy()
    
    preprocessed_data = convert_to_numeric(preprocessed_data, config.numerical_columns)
    
    return preprocessed_data

def remove_outliers(data, columns, threshold, target_column=None, 
                   remove_target_outliers=False, target_method='none'):
    """
    Remove outliers based on Z-scores for specified columns, 
    ensuring target column is handled separately.
    
    Args:
    - data: DataFrame
    - columns: List of column names to check for outliers
    - threshold: Z-score threshold for outlier detection
    - target_column: Name of the target column to be handled carefully
    - remove_target_outliers: Whether to remove outliers from target
    - target_method: 'zscore', 'iqr', 'percentile', or 'none'
    
    Returns:
    - Cleaned DataFrame with outlier statistics
    """
    debug_print(f"Removing outliers based on Z-scores in columns: {columns}")
    initial_shape = data.shape
    
    cleaned_data = data.copy()
    outlier_stats = {
        'total_removed': 0,
        'by_column': {},
        'target_outliers_removed': 0
    }
    
    # Separate target column handling
    feature_columns = [col for col in columns if col != target_column]
    
    if not feature_columns and not remove_target_outliers:
        st.warning("No valid columns for outlier removal")
        return cleaned_data
    
    # ========================================================================
    # FEATURE OUTLIER REMOVAL (Standard approach)
    # ========================================================================
    feature_mask = np.ones(len(cleaned_data), dtype=bool)  # Default: keep all
    
    if feature_columns:
        valid_columns = [col for col in feature_columns if col in cleaned_data.columns]
        
        if valid_columns:
            # Calculate Z-scores for features
            z_scores = np.abs(zscore(cleaned_data[valid_columns].astype(float), nan_policy='omit'))
            
            # Create mask for rows that don't exceed threshold in ANY feature
            feature_mask = (z_scores < threshold).all(axis=1)
            
            features_removed = (~feature_mask).sum()
            outlier_stats['total_removed'] += features_removed
            
            for col in valid_columns:
                col_outliers = ((z_scores[col] >= threshold)).sum()
                outlier_stats['by_column'][col] = col_outliers
    
    # ========================================================================
    # TARGET OUTLIER REMOVAL (User controlled)
    # ========================================================================
    target_mask = np.ones(len(cleaned_data), dtype=bool)  # Default: keep all
    
    if target_column and remove_target_outliers and target_method != 'none':
        st.subheader("Target Variable Outlier Removal")
        
        target_data = cleaned_data[target_column].astype(float)
        
        if target_method == 'zscore':
            # Z-score method (assumes normal distribution)
            target_z = np.abs(zscore(target_data, nan_policy='omit'))
            target_mask = target_z < threshold
            
            outliers_removed = (~target_mask).sum()
            outlier_stats['target_outliers_removed'] = outliers_removed
            
            st.info(f"Z-score method: {outliers_removed} target outliers removed (>{threshold} std devs)")
            
        elif target_method == 'iqr':
            # IQR method (robust to non-normal distributions)
            Q1 = target_data.quantile(0.25)
            Q3 = target_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            target_mask = (target_data >= lower_bound) & (target_data <= upper_bound)
            
            outliers_removed = (~target_mask).sum()
            outlier_stats['target_outliers_removed'] = outliers_removed
            
            st.info(f"IQR method: {outliers_removed} target outliers removed (outside [{lower_bound:.2f}, {upper_bound:.2f}])")
            
        elif target_method == 'percentile':
            # Percentile method (remove extreme percentiles)
            lower_percentile = st.slider("Lower percentile cutoff", 0.0, 5.0, 1.0, 0.1)
            upper_percentile = st.slider("Upper percentile cutoff", 95.0, 100.0, 99.0, 0.1)
            
            lower_bound = target_data.quantile(lower_percentile / 100)
            upper_bound = target_data.quantile(upper_percentile / 100)
            
            target_mask = (target_data >= lower_bound) & (target_data <= upper_bound)
            
            outliers_removed = (~target_mask).sum()
            outlier_stats['target_outliers_removed'] = outliers_removed
            
            st.info(f"Percentile method: {outliers_removed} target outliers removed (outside {lower_percentile}%-{upper_percentile}%)")
        
        # Visualize target outliers
        if outliers_removed > 0:
            fig = go.Figure()
            
            # All data
            fig.add_trace(go.Histogram(
                x=target_data,
                name='All data',
                opacity=0.7,
                marker_color='blue'
            ))
            
            # After outlier removal
            fig.add_trace(go.Histogram(
                x=target_data[target_mask],
                name='After outlier removal',
                opacity=0.7,
                marker_color='green'
            ))
            
            fig.update_layout(
                title=f'Target Variable Distribution (Before/After Outlier Removal)',
                xaxis_title=target_column,
                yaxis_title='Frequency',
                barmode='overlay',
                height=CHART_HEIGHT,
                width=CHART_WIDTH
            )
            st.plotly_chart(fig)
    
    # ========================================================================
    # COMBINE MASKS AND APPLY
    # ========================================================================
    combined_mask = feature_mask & target_mask
    cleaned_data = cleaned_data[combined_mask].reset_index(drop=True)
    
    total_removed = (~combined_mask).sum()
    outlier_stats['total_removed'] = total_removed
    
    # Display summary
    st.subheader("Outlier Removal Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Initial Rows", initial_shape[0])
    with col2:
        st.metric("Outliers Removed", total_removed)
    with col3:
        st.metric("Final Rows", cleaned_data.shape[0])
    
    if outlier_stats['by_column']:
        st.write("Outliers by feature:")
        outlier_df = pd.DataFrame([
            {'Column': col, 'Outliers': count} 
            for col, count in outlier_stats['by_column'].items()
        ]).sort_values('Outliers', ascending=False)
        st.dataframe(outlier_df)
    
    debug_print(f"Outliers removed. Data shape changed from {initial_shape} to {cleaned_data.shape}")
    
    return cleaned_data

def convert_to_numeric(df, numerical_columns):
    debug_print(f"Converting numerical columns to numeric type")
    for col in numerical_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    debug_print("Columns converted to numeric types successfully.")
    return df

def create_preprocessor_pipeline(numerical_cols, categorical_cols):
    debug_print(f"Creating preprocessor pipeline with numerical_cols: {numerical_cols} and categorical_cols: {categorical_cols}")
    transformers = []
    
    if numerical_cols:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_transformer, numerical_cols))
    
    if categorical_cols:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False, max_categories=50))
        ])
        transformers.append(('cat', categorical_transformer, categorical_cols))

    if not transformers:
        debug_print("No transformers created, returning None")
        return None

    debug_print(f"Created transformers: {transformers}")
    return ColumnTransformer(transformers=transformers, remainder='passthrough')
  
def split_and_preprocess_data(preprocessed_data, clustered_data, target_column, 
                              train_size, randomize=True, randomization_method='together',
                              config=None):
    """
    Split data with optional randomization - SINGLE SOURCE OF TRUTH for randomization.
    
    Args:
    - preprocessed_data: Original DataFrame
    - clustered_data: Clustered data configuration
    - target_column: Name of the target column
    - train_size: Proportion of data for training
    - randomize: Whether to randomize data before splitting (default True)
    - randomization_method: 'together', 'stratified', or 'none_in_split'
    - config: Configuration object with RANDOM_STATE
    
    Returns:
    - data_splits: Dictionary with train/test splits for each cluster
    """
    debug_print("Splitting and preprocessing data...")
    data_splits = {}
    flattened_clustered_data = flatten_clustered_data(clustered_data)
    
    random_state = config.RANDOM_STATE if config else RANDOM_STATE

    progress_bar = st.progress(0)
    for i, (cluster_name, label, indices) in enumerate(flattened_clustered_data):
        debug_print(f"\n[DEBUG] Processing cluster: {cluster_name}, label: {label}")

        if indices is None or len(indices) == 0:
            debug_print(f"Skipping cluster {cluster_name}, label {label} due to empty indices")
            continue

        try:
            data_subset = preprocessed_data.loc[indices].reset_index(drop=True)
        except Exception as e:
            debug_print(f"Error accessing indices for {cluster_name}, label {label}: {str(e)}")
            continue

        if data_subset.empty:
            debug_print(f"Skipping empty data subset for cluster {cluster_name}, label {label}")
            continue

        X, y = split_features_and_target(data_subset, target_column)

        # ====================================================================
        # RANDOMIZATION LOGIC - Three methods with clear behavior
        # CRITICAL FIX #1: Shuffle X and y TOGETHER to prevent data leakage
        # ====================================================================
        
        if randomization_method == 'together':
            # Method 1: Manual shuffle of both X and y together (SAFEST)
            debug_print(f"Applying 'together' randomization")
            shuffled_indices = np.random.RandomState(random_state).permutation(len(X))
            X = X.iloc[shuffled_indices].reset_index(drop=True)
            y = y.iloc[shuffled_indices].reset_index(drop=True)
            
            # Split WITHOUT additional shuffle
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                train_size=train_size, 
                random_state=random_state,
                shuffle=False  # Already shuffled
            )
            
        elif randomization_method == 'stratified':
            # Method 2: Stratified split (maintains class distribution)
            debug_print(f"Applying 'stratified' randomization")
            
            # Check if stratification makes sense
            if y.dtype == 'object' or y.nunique() < 20:
                # Categorical target or few unique values
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    train_size=train_size, 
                    random_state=random_state,
                    shuffle=True,
                    stratify=y  # Stratify by target
                )
            else:
                # Continuous target - bin it for stratification
                # Create bins for stratification
                y_binned = pd.qcut(y, q=min(5, len(y)//10), labels=False, duplicates='drop')
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    train_size=train_size, 
                    random_state=random_state,
                    shuffle=True,
                    stratify=y_binned
                )
        
        elif randomization_method == 'none_in_split':
            # Method 3: Let train_test_split handle shuffling
            debug_print(f"Using train_test_split's built-in shuffle")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                train_size=train_size, 
                random_state=random_state,
                shuffle=True  # Use sklearn's shuffle
            )
        
        else:
            # No randomization
            debug_print(f"No randomization applied")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                train_size=train_size, 
                random_state=random_state,
                shuffle=False
            )
        
        # ====================================================================
        # End randomization logic
        # ====================================================================

        debug_print(f"Data subset for cluster {cluster_name}, label {label} created with shape: {data_subset.shape}")
        debug_print(f"X shape: {X.shape}, y shape: {y.shape}")
        debug_print(f"Split data for cluster {cluster_name}, label {label}: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
        # Get preprocessing columns from config
        numerical_columns = config.numerical_columns if config else []
        categorical_columns = config.categorical_columns if config else []
        
        preprocessor = create_preprocessor_pipeline(numerical_columns, categorical_columns)

        try:
            if preprocessor is None:
                X_train_prepared, X_test_prepared = X_train, X_test
                feature_names = X_train.columns.tolist()
                debug_print("No preprocessing applied")
            else:
                debug_print("Fitting preprocessor...")
                preprocessor.fit(X_train)
                debug_print("Transforming X_train...")
                X_train_prepared = preprocessor.transform(X_train)
                debug_print("Transforming X_test...")
                X_test_prepared = preprocessor.transform(X_test)

                feature_names = get_feature_names(preprocessor)
                debug_print(f"Feature names after preprocessing: {feature_names}")
                
                # Convert to DataFrame and ensure correct shape
                X_train_prepared = pd.DataFrame(X_train_prepared, columns=feature_names, index=X_train.index)
                X_test_prepared = pd.DataFrame(X_test_prepared, columns=feature_names, index=X_test.index)

            debug_print(f"Columns after preprocessing for cluster {cluster_name}, label {label}: {X_train_prepared.columns.tolist()}")
            debug_print(f"Number of columns after preprocessing for cluster {cluster_name}, label {label}: {len(X_train_prepared.columns)}")

            save_preprocessor(preprocessor, cluster_name, label)
        except Exception as e:
            debug_print(f"Error occurred during preprocessor fitting or transformation: {str(e)}")
            debug_print(f"Error type: {type(e).__name__}")
            debug_print(f"Error traceback: {traceback.format_exc()}")
            debug_print(f"Skipping preprocessing for cluster {cluster_name}, label {label} due to the error.")
            X_train_prepared, X_test_prepared = X_train, X_test
            feature_names = X_train.columns.tolist()
            preprocessor = None

        data_splits[f"{cluster_name}_{label}"] = {
            'X_train': X_train_prepared,
            'X_test': X_test_prepared,
            'y_train': y_train,
            'y_test': y_test,
            'preprocessor': preprocessor,
            'feature_names': feature_names
        }
        debug_print(f"Data split and preprocessing completed for cluster {cluster_name}, label {label}.")
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(flattened_clustered_data))

    debug_print("\n[DEBUG] Finished processing all clusters and labels")
    return data_splits

def split_features_and_target(data_subset, target_column):
    X = data_subset.drop(columns=[target_column])
    y = data_subset[target_column]
    return X, y

def get_feature_names(preprocessor):
    if preprocessor is None:
        return []
    
    feature_names = []
    
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(columns)
        elif name == 'cat':
            if hasattr(transformer, 'get_feature_names_out'):
                if isinstance(transformer, Pipeline):
                    transformer_feature_names = transformer.steps[-1][1].get_feature_names_out(columns)
                else:
                    transformer_feature_names = transformer.get_feature_names_out(columns)
                feature_names.extend(transformer_feature_names)
            else:
                feature_names.extend(columns)
    
    return feature_names

def save_preprocessor(preprocessor, cluster_name, label):
    if preprocessor is not None:
        preprocessor_filename = os.path.join(MODELS_DIRECTORY, f'preprocessor_{cluster_name}_{label}.joblib')
        joblib.dump(preprocessor, preprocessor_filename)
        
        # Save individual components
        numeric_transformer = preprocessor.named_transformers_.get('num')
        if numeric_transformer:
            imputer = numeric_transformer.named_steps['imputer']
            scaler = numeric_transformer.named_steps['scaler']
            
            joblib.dump(imputer, os.path.join(MODELS_DIRECTORY, f'imputer_{cluster_name}_{label}.joblib'))
            joblib.dump(scaler, os.path.join(MODELS_DIRECTORY, f'scaler_{cluster_name}_{label}.joblib'))
        
        categorical_transformer = preprocessor.named_transformers_.get('cat')
        if categorical_transformer:
            onehot_encoder = categorical_transformer.named_steps['onehot']
            joblib.dump(onehot_encoder, os.path.join(MODELS_DIRECTORY, f'onehot_encoder_{cluster_name}_{label}.joblib'))
        
        debug_print(f"Preprocessing components saved for cluster {cluster_name}, label {label}")
    else:
        debug_print(f"No preprocessor saved for cluster {cluster_name}, label {label} as it encountered an error.")

def create_global_preprocessor(data, config=None):
    debug_print("Creating global preprocessor...")
    
    from _0config import config as global_config
    cfg = config if config is not None else global_config
    
    numerical_cols = cfg.numerical_columns
    categorical_cols = cfg.categorical_columns
    all_cols = numerical_cols + categorical_cols
    
    debug_print(f"Numerical columns: {numerical_cols}")
    debug_print(f"Categorical columns: {categorical_cols}")
    
    transformers = []
    
    if numerical_cols:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_transformer, numerical_cols))
    
    if categorical_cols:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False, max_categories=50))
        ])
        transformers.append(('cat', categorical_transformer, categorical_cols))
    
    if not transformers:
        debug_print("No transformers created, returning None")
        return None
    
    debug_print(f"Created transformers: {transformers}")
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    
    # Fit the preprocessor
    preprocessor.fit(data[all_cols])
    
    return preprocessor
  
def save_global_preprocessor(preprocessor, save_path):
    if preprocessor is not None:
        joblib.dump(preprocessor, os.path.join(save_path, 'global_preprocessor.joblib'))
        debug_print("Global preprocessor saved successfully.")
    else:
        debug_print("No global preprocessor to save (preprocessor is None).")

def load_global_preprocessor(save_path):
    try:
        preprocessor = joblib.load(os.path.join(save_path, 'global_preprocessor.joblib'))
        debug_print("Global preprocessor loaded successfully.")
        return preprocessor
    except FileNotFoundError:
        debug_print("Global preprocessor file not found.")
        return None