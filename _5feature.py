import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from _0config import config, MAX_INTERACTION_DEGREE, STATISTICAL_AGG_FUNCTIONS, TOP_K_FEATURES, POLYNOMIAL_DEGREE, TOOLTIPS, INFO_TEXTS, CHART_HEIGHT, CHART_WIDTH
from _2misc_utils import debug_print, plot_feature_importance, flatten_clustered_data
from _2ui_utils import create_tooltip, create_info_button

def generate_polynomial_features(X, degree=None, config=None):
    """
    Generate polynomial features from numerical columns.
    
    Args:
    - X: DataFrame with features
    - degree: Polynomial degree (optional, defaults to config value)
    - config: Configuration object (optional, uses global if not provided)
    
    Returns:
    - DataFrame with polynomial features
    """
    # Use provided config or fall back to global
    if config is None:
        from _0config import config as global_config
        config = global_config
    
    if degree is None:
        degree = POLYNOMIAL_DEGREE
    
    numerical_cols = config.numerical_columns
    
    # Filter to only include numerical columns that exist in X
    numerical_cols = [col for col in numerical_cols if col in X.columns]
    
    if not numerical_cols:
        st.warning("No numerical columns found. Skipping polynomial feature generation.")
        return pd.DataFrame(index=X.index)
    
    X_numeric = X[numerical_cols]
    if X_numeric.empty:
        st.warning("No numerical columns found. Skipping polynomial feature generation.")
        return pd.DataFrame(index=X.index)

    st.write(f"Generating polynomial features with degree {degree}")
    create_tooltip(TOOLTIPS["polynomial_features"])
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_numeric)
    feature_names = poly.get_feature_names_out(X_numeric.columns)
    new_features = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
    st.write(f"Generated {new_features.shape[1]} polynomial features")
    return new_features

def generate_interaction_terms(X, max_degree=None, config=None):
    """
    Generate interaction terms between features.
    
    Args:
    - X: DataFrame with features
    - max_degree: Maximum degree for interactions (optional)
    - config: Configuration object (optional, uses global if not provided)
    
    Returns:
    - DataFrame with interaction features
    """
    # Use provided config or fall back to global
    if config is None:
        from _0config import config as global_config
        config = global_config
    
    if max_degree is None:
        max_degree = MAX_INTERACTION_DEGREE
    
    numerical_cols = config.numerical_columns
    categorical_cols = config.categorical_columns
    
    # Filter to only include columns that exist in X
    numerical_cols = [col for col in numerical_cols if col in X.columns]
    categorical_cols = [col for col in categorical_cols if col in X.columns]
    
    X_numeric = X[numerical_cols] if numerical_cols else pd.DataFrame()
    X_categorical = X[categorical_cols] if categorical_cols else pd.DataFrame()
    
    if X_numeric.empty and X_categorical.empty:
        st.warning("No numerical or categorical columns found. Skipping interaction feature generation.")
        return pd.DataFrame(index=X.index)

    st.write(f"Generating interaction terms with max degree {max_degree}")
    create_tooltip(TOOLTIPS["interaction_terms"])
    interaction_columns = []
    
    # Numeric-Numeric interactions
    if not X_numeric.empty and len(X_numeric.columns) >= 2:
        for i in range(2, min(max_degree + 1, len(X_numeric.columns) + 1)):
            for combo in combinations(X_numeric.columns, i):
                col_name = '_X_'.join(combo)
                interaction_columns.append(pd.Series(X_numeric[list(combo)].product(axis=1), name=col_name))
    
    # Numeric-Categorical interactions
    if not X_numeric.empty and not X_categorical.empty:
        for num_col in X_numeric.columns:
            for cat_col in X_categorical.columns:
                col_name = f"{num_col}_X_{cat_col}"
                interaction_columns.append(pd.Series(X_numeric[num_col] * pd.factorize(X[cat_col])[0], name=col_name))

    if interaction_columns:
        new_features = pd.concat(interaction_columns, axis=1)
        st.write(f"Generated {new_features.shape[1]} interaction features")
    else:
        new_features = pd.DataFrame(index=X.index)
        st.warning("No interaction features generated")

    return new_features

def generate_statistical_features(X, config=None):
    """
    Generate statistical features from numerical columns.
    
    Args:
    - X: DataFrame with features
    - config: Configuration object (optional, uses global if not provided)
    
    Returns:
    - DataFrame with statistical features
    """
    # Use provided config or fall back to global
    if config is None:
        from _0config import config as global_config
        config = global_config
    
    numerical_cols = config.numerical_columns
    
    # Filter to only include numerical columns that exist in X
    numerical_cols = [col for col in numerical_cols if col in X.columns]
    
    if not numerical_cols:
        st.warning("No numerical columns found. Skipping statistical feature generation.")
        return pd.DataFrame(index=X.index)
    
    X_numeric = X[numerical_cols]

    if X_numeric.empty:
        st.warning("No numerical columns found. Skipping statistical feature generation.")
        return pd.DataFrame(index=X.index)

    st.write(f"Generating statistical features using functions: {STATISTICAL_AGG_FUNCTIONS}")
    create_tooltip(TOOLTIPS["statistical_features"])
    new_features = pd.DataFrame(index=X.index)
    for func in STATISTICAL_AGG_FUNCTIONS:
        new_features[f"{func}_all"] = X_numeric.agg(func, axis=1)

    st.write(f"Generated {new_features.shape[1]} statistical features")
    return new_features

def apply_feature_generation(data_splits, feature_generation_functions, config=None):
    """
    Apply feature generation functions to all data splits.
    
    Args:
    - data_splits: Dictionary of train/test splits for each cluster
    - feature_generation_functions: List of feature generation functions
    - config: Configuration object (optional)
    
    Returns:
    - Tuple of (clustered_X_train_combined, clustered_X_test_combined)
    """
    st.subheader("Applying Feature Generation")
    create_info_button("feature_generation")
    clustered_X_train_combined = {}
    clustered_X_test_combined = {}

    progress_bar = st.progress(0)
    
    for i, (cluster_key, split_data) in enumerate(data_splits.items()):
        X_train, y_train = split_data['X_train'], split_data['y_train']
        X_test = split_data['X_test']

        st.write(f"Applying feature generation for cluster: {cluster_key}")
        st.write(f"Initial X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

        new_features_train = pd.DataFrame(index=X_train.index)
        new_features_test = pd.DataFrame(index=X_test.index)

        for feature_gen_func in feature_generation_functions:
            try:
                # Call function - it already has config bound via partial in _1main.py
                new_features_train_func = feature_gen_func(X_train)
                new_features_test_func = feature_gen_func(X_test)
                
                # Ensure unique column names
                new_features_train_func = new_features_train_func.add_prefix(f"{feature_gen_func.func.__name__}_")
                new_features_test_func = new_features_test_func.add_prefix(f"{feature_gen_func.func.__name__}_")
                
                new_features_train = pd.concat([new_features_train, new_features_train_func], axis=1)
                new_features_test = pd.concat([new_features_test, new_features_test_func], axis=1)
            except Exception as e:
                st.error(f"Error in feature generation for cluster {cluster_key}: {str(e)}")
                continue

        X_train_combined = pd.concat([X_train, new_features_train], axis=1)
        X_test_combined = pd.concat([X_test, new_features_test], axis=1)

        # Remove duplicate columns
        X_train_combined = X_train_combined.loc[:, ~X_train_combined.columns.duplicated()]
        X_test_combined = X_test_combined.loc[:, ~X_test_combined.columns.duplicated()]

        # Apply feature selection
        X_train_selected = select_top_features(X_train_combined, y_train)
        X_test_selected = X_test_combined[X_train_selected.columns]

        clustered_X_train_combined[cluster_key] = X_train_selected
        clustered_X_test_combined[cluster_key] = X_test_selected

        st.write(f"Final X_train shape for cluster {cluster_key}: {X_train_selected.shape}")
        st.write(f"Final X_test shape for cluster {cluster_key}: {X_test_selected.shape}")
        
        progress_bar.progress((i + 1) / len(data_splits))

    st.success("Feature generation completed for all clusters.")
    return clustered_X_train_combined, clustered_X_test_combined
  
def select_top_features(X, y, k=TOP_K_FEATURES):
    """Select top K features based on F-score and mutual information."""
    X_numeric = X.select_dtypes(include=['number'])
    if X_numeric.shape[1] == 0:
        st.warning("No numeric features found for selection. Returning original features.")
        return X
    
    st.write(f"Selecting top {k} features")
    create_tooltip(TOOLTIPS["feature_selection"])
    
    # Ensure k doesn't exceed number of features
    k = min(k, X_numeric.shape[1])
    
    # F-regression for linear relationships
    f_selector = SelectKBest(score_func=f_regression, k=k)
    f_selector.fit(X_numeric, y)
    f_scores = f_selector.scores_
    
    # Mutual information for non-linear relationships
    mi_selector = SelectKBest(score_func=mutual_info_regression, k=k)
    mi_selector.fit(X_numeric, y)
    mi_scores = mi_selector.scores_
    
    # Combine scores
    combined_scores = f_scores + mi_scores
    selected_feature_mask = np.argsort(combined_scores)[-k:]
    selected_numeric_features = X_numeric.columns[selected_feature_mask]
    
    # Include all non-numeric columns
    non_numeric_features = X.select_dtypes(exclude=['number']).columns
    selected_features = list(selected_numeric_features) + list(non_numeric_features)
    
    st.write(f"Selected {len(selected_features)} features")
    
    # Plot feature importance
    feature_importance = pd.DataFrame({
        'feature': selected_numeric_features,
        'f_score': f_scores[selected_feature_mask],
        'mi_score': mi_scores[selected_feature_mask],
        'combined_score': combined_scores[selected_feature_mask]
    }).sort_values('combined_score', ascending=False)
    
    plot_feature_importance(feature_importance)
    
    return X[selected_features]

def combine_feature_engineered_data(data_splits, clustered_X_train_combined, clustered_X_test_combined):
    """Combine feature engineered data across clusters."""
    st.subheader("Combining Feature Engineered Data")
    create_info_button("combine_features")
    
    for cluster_key in clustered_X_train_combined.keys():
        X_train = clustered_X_train_combined[cluster_key]
        X_test = clustered_X_test_combined[cluster_key]

        # Remove duplicate columns
        X_train = X_train.loc[:, ~X_train.columns.duplicated()]
        X_test = X_test.loc[:, ~X_test.columns.duplicated()]

        # Get the union of columns from both train and test
        all_columns = X_train.columns.union(X_test.columns)

        # Reindex both dataframes with the union of columns
        X_train = X_train.reindex(columns=all_columns, fill_value=0)
        X_test = X_test.reindex(columns=all_columns, fill_value=0)

        # Sort the columns alphabetically
        X_train = X_train.reindex(sorted(X_train.columns), axis=1)
        X_test = X_test.reindex(sorted(X_test.columns), axis=1)

        clustered_X_train_combined[cluster_key] = X_train
        clustered_X_test_combined[cluster_key] = X_test

        st.write(f"Combined data for cluster {cluster_key}:")
        st.write(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    st.success("Feature engineered data combined for all clusters.")
    return clustered_X_train_combined, clustered_X_test_combined

def generate_features_for_prediction(X, feature_generation_functions, config=None):
    """
    Generate features for prediction using the same functions as training.
    
    Args:
    - X: DataFrame with features
    - feature_generation_functions: List of feature generation functions
    - config: Configuration object (optional)
    
    Returns:
    - DataFrame with generated features
    """
    st.subheader("Generating Features for Prediction")
    create_info_button("prediction_features")
    new_features = pd.DataFrame(index=X.index)

    for feature_gen_func in feature_generation_functions:
        try:
            # Call function - it may be a partial with config already bound
            new_features_func = feature_gen_func(X)
            
            # Handle both partial and regular functions
            func_name = feature_gen_func.func.__name__ if hasattr(feature_gen_func, 'func') else feature_gen_func.__name__
            new_features_func = new_features_func.add_prefix(f"{func_name}_")
            
            new_features = pd.concat([new_features, new_features_func], axis=1)
        except Exception as e:
            st.error(f"Error in feature generation for prediction: {str(e)}")
            continue

    X_combined = pd.concat([X, new_features], axis=1)
    X_combined = X_combined.loc[:, ~X_combined.columns.duplicated()]

    st.write(f"Generated features for prediction. Final shape: {X_combined.shape}")
    return X_combined

def display_feature_info(clustered_X_train_combined, clustered_X_test_combined):
    """Display information about features in each cluster."""
    st.subheader("Feature Information")
    create_info_button("feature_info")
    for cluster_key in clustered_X_train_combined.keys():
        st.write(f"Cluster: {cluster_key}")
        st.write(f"Number of features: {clustered_X_train_combined[cluster_key].shape[1]}")
        st.write("Top 10 features:")
        st.write(clustered_X_train_combined[cluster_key].columns[:10].tolist())
        st.write("---")

def plot_feature_correlations(X, y):
    """Plot correlations between features and target."""
    st.subheader("Feature Correlations with Target")
    create_info_button("feature_correlations")
    correlations = X.apply(lambda x: x.corr(y) if x.dtype in ['int64', 'float64'] else 0)
    correlations = correlations.sort_values(ascending=False)
    
    fig = px.bar(x=correlations.index, y=correlations.values, 
                 labels={'x': 'Features', 'y': 'Correlation with Target'},
                 title='Feature Correlations with Target Variable',
                 height=CHART_HEIGHT, width=CHART_WIDTH)
    st.plotly_chart(fig)

def visualize_feature_distributions(X):
    """Visualize distributions of numerical features."""
    st.subheader("Feature Distributions")
    create_info_button("feature_distributions")
    
    numeric_cols = X.select_dtypes(include=['number']).columns
    
    for col in numeric_cols[:10]:  # Limit to first 10 to avoid overwhelming
        fig = px.histogram(X, x=col, title=f'Distribution of {col}',
                           height=CHART_HEIGHT, width=CHART_WIDTH)
        st.plotly_chart(fig)

def analyze_feature_interactions(X, y, top_n=5):
    """Analyze interactions between top features."""
    st.subheader("Feature Interactions")
    create_info_button("feature_interactions")
    
    numeric_cols = X.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) < 2:
        st.warning("Not enough numeric features for interaction analysis.")
        return
    
    # Get top N features based on correlation with target
    top_features = X[numeric_cols].corrwith(y).abs().nlargest(min(top_n, len(numeric_cols))).index
    
    for i in range(len(top_features)):
        for j in range(i+1, len(top_features)):
            fig = px.scatter(x=X[top_features[i]], y=X[top_features[j]], color=y,
                             title=f'Interaction: {top_features[i]} vs {top_features[j]}',
                             labels={'x': top_features[i], 'y': top_features[j], 'color': 'Target'},
                             height=CHART_HEIGHT, width=CHART_WIDTH)
            st.plotly_chart(fig)