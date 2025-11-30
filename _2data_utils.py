import os
import pandas as pd
import streamlit as st

from _0config import config, SHEET_NAME_MAX_LENGTH, TRUNCATE_SHEET_NAME_REPLACEMENT, TOOLTIPS
from _2ui_utils import create_tooltip, create_info_button

def debug_print(*args):
    """Displays debugging information in Streamlit."""
    message = ' '.join(map(str, args))
    st.text(f"DEBUG: {message}")

def display_sheet_selection(uploaded_file):
    """Select sheet for Excel files."""
    create_tooltip(TOOLTIPS["sheet_selection"])
    xls = pd.ExcelFile(uploaded_file)
    
    # If only one sheet, return it
    if len(xls.sheet_names) == 1:
        return xls.sheet_names[0]
    
    # If multiple sheets, allow user to select
    sheet_name = st.selectbox("Select sheet", xls.sheet_names)
    
    # Truncate sheet name if too long
    if len(sheet_name) > SHEET_NAME_MAX_LENGTH:
        sheet_name = sheet_name[:SHEET_NAME_MAX_LENGTH - len(TRUNCATE_SHEET_NAME_REPLACEMENT)] + TRUNCATE_SHEET_NAME_REPLACEMENT
    
    return sheet_name

def load_data(file_path, sheet_name=None):
    """Load data from Excel or CSV file."""
    try:
        # Check if file_path is a Streamlit UploadedFile or a path
        if hasattr(file_path, 'name'):
            # Streamlit UploadedFile
            if file_path.name.endswith('.xlsx'):
                if sheet_name is None:
                    # If no sheet name provided, try to get the first sheet
                    xls = pd.ExcelFile(file_path)
                    sheet_name = xls.sheet_names[0]
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                except Exception as e:
                    st.error(f"Error reading Excel file: {str(e)}")
                    return None
            elif file_path.name.endswith('.csv'):
                try:
                    df = pd.read_csv(file_path)
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
                    return None
            else:
                st.error("Unsupported file format.")
                return None
        else:
            # Assume it's a file path string
            if str(file_path).endswith('.xlsx'):
                if sheet_name is None:
                    # If no sheet name provided, try to get the first sheet
                    xls = pd.ExcelFile(file_path)
                    sheet_name = xls.sheet_names[0]
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                except Exception as e:
                    st.error(f"Error reading Excel file: {str(e)}")
                    return None
            elif str(file_path).endswith('.csv'):
                try:
                    df = pd.read_csv(file_path)
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
                    return None
            else:
                st.error("Unsupported file format.")
                return None
        
        # Validate DataFrame
        if df is None or df.empty:
            st.error("The loaded dataframe is empty.")
            return None
        
        # Reset index to ensure consistent indexing
        df = df.reset_index(drop=True)
        
        return df
    
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the data: {str(e)}")
        return None

def display_data_info(df):
    """Display information about the loaded data."""
    st.subheader("Data Information")
    create_info_button("data_information")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    st.write(f"Columns: {', '.join(df.columns)}")
    
    display_dataframe(df.head(), "Data Preview")

def handle_missing_values(df):
    """Handle missing values in the DataFrame."""
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        st.warning("Missing values detected in the following columns:")
        st.write(missing_values[missing_values > 0])
        
        strategy = st.selectbox("Choose a strategy to handle missing values:", 
                               ["Drop rows", "Fill with mean/mode", "Fill with median"])
        
        create_tooltip(TOOLTIPS["handle_missing_values"])
        
        if strategy == "Drop rows":
            df = df.dropna()
            st.success("Rows with missing values have been dropped.")
        elif strategy == "Fill with mean/mode":
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
            st.success("Missing values have been filled with mean/mode.")
        elif strategy == "Fill with median":
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
            st.success("Missing values have been filled with median.")
    
    return df

def auto_detect_column_types(data):
    """Automatically detect column types."""
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = data.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    target_column = data.columns[-1]
    
    # Remove target column from numeric or categorical lists
    if target_column in numeric_columns:
        numeric_columns.remove(target_column)
    elif target_column in categorical_columns:
        categorical_columns.remove(target_column)
    
    return {
        'numerical': numeric_columns,
        'categorical': categorical_columns,
        'target': target_column,
        'unused': []
    }

def display_column_selection(columns, initial_types):
    """Display interface for manual column selection."""
    st.subheader("Column Selection")
    create_info_button("column_selection")
    
    column_types = {}
    for col in columns:
        if col == initial_types['target']:
            column_types[col] = st.selectbox(f"Select type for {col}", 
                                             ['target', 'numerical', 'categorical', 'unused'],
                                             index=0)
        else:
            column_types[col] = st.selectbox(f"Select type for {col}", 
                                             ['numerical', 'categorical', 'unused', 'target'],
                                             index=['numerical', 'categorical', 'unused', 'target'].index(
                                                 'numerical' if col in initial_types['numerical'] 
                                                 else 'categorical' if col in initial_types['categorical']
                                                 else 'unused'
                                             ))
    
    # Ensure we have a target column
    if 'target' not in column_types.values():
        st.error("Please select a target column")
        return None
    
    return {
        'numerical': [col for col, type in column_types.items() if type == 'numerical'],
        'categorical': [col for col, type in column_types.items() if type == 'categorical'],
        'target': next(col for col, type in column_types.items() if type == 'target'),
        'unused': [col for col, type in column_types.items() if type == 'unused']
    }

def save_unused_data(unused_data, file_path):
    """Save unused data as CSV."""
    if not unused_data.empty:
        unused_data.to_csv(file_path, index=False)
        debug_print(f"Unused data saved to {file_path}")

def display_dataframe(df, title="DataFrame"):
    """Display a DataFrame in Streamlit with pagination."""
    st.subheader(title)
    page_size = 10
    total_pages = (len(df) - 1) // page_size + 1
    page_number = st.number_input(f"Page (1-{total_pages})", min_value=1, max_value=total_pages, value=1)
    start_idx = (page_number - 1) * page_size
    end_idx = min(start_idx + page_size, len(df))
    st.dataframe(df.iloc[start_idx:end_idx])
    st.info(f"Showing rows {start_idx+1} to {end_idx} of {len(df)}")

def check_and_remove_duplicate_columns(df):
    """Check and remove duplicate columns from a DataFrame."""
    duplicate_columns = df.columns[df.columns.duplicated()]
    if len(duplicate_columns) > 0:
        st.warning(f"Duplicate columns found and removed: {', '.join(duplicate_columns)}")
        df = df.loc[:, ~df.columns.duplicated()]
    return df

def check_and_reset_indices(df):
    """Check and reset indices if they are not unique or continuous."""
    if not df.index.is_unique or not df.index.is_monotonic_increasing:
        st.info("Indices are not unique or continuous. Resetting index.")
        df = df.reset_index(drop=True)
    return df