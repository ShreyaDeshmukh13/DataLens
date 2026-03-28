"""
Data Cleaning Module
Handles duplicate removal, missing value imputation, and data type fixing.
"""
import pandas as pd
import numpy as np


def remove_duplicates(df):
    """Remove duplicate rows and return cleaned df + count removed."""
    initial_count = len(df)
    df_cleaned = df.drop_duplicates()
    removed_count = initial_count - len(df_cleaned)
    return df_cleaned, removed_count


def handle_missing_values(df):
    """
    Fill missing values:
    - Numeric columns → fill with column mean
    - Categorical columns → fill with 'Unknown'
    Returns cleaned df + dict of changes per column.
    """
    changes = {}
    df_cleaned = df.copy()

    for col in df_cleaned.columns:
        missing_count = df_cleaned[col].isnull().sum()
        if missing_count > 0:
            if df_cleaned[col].dtype in ['int64', 'float64']:
                fill_value = round(df_cleaned[col].mean(), 2)
                df_cleaned[col] = df_cleaned[col].fillna(fill_value)
                changes[col] = {
                    'count': int(missing_count),
                    'strategy': f'Filled with mean ({fill_value})'
                }
            else:
                df_cleaned[col] = df_cleaned[col].fillna('Unknown')
                changes[col] = {
                    'count': int(missing_count),
                    'strategy': 'Filled with "Unknown"'
                }

    return df_cleaned, changes


def fix_data_types(df):
    """
    Auto-detect and fix incorrect data types.
    E.g., columns stored as strings that should be numeric.
    Returns cleaned df + list of type changes.
    """
    type_changes = []
    df_cleaned = df.copy()

    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object':
            # Try converting to numeric
            try:
                converted = pd.to_numeric(df_cleaned[col], errors='coerce')
                # If more than 50% of non-null values converted successfully, apply
                non_null = df_cleaned[col].dropna()
                if len(non_null) > 0:
                    success_rate = converted.notna().sum() / len(non_null)
                    if success_rate > 0.5:
                        df_cleaned[col] = converted
                        type_changes.append({
                            'column': col,
                            'from': 'object (text)',
                            'to': str(converted.dtype),
                            'note': f'{int(success_rate * 100)}% values converted'
                        })
            except Exception:
                pass

    return df_cleaned, type_changes


def clean_data(df):
    """
    Run the full cleaning pipeline:
    1. Remove duplicates
    2. Fix data types
    3. Handle missing values
    Returns cleaned df + comprehensive summary report.
    """
    report = {}

    # Step 1: Remove duplicates
    df, duplicates_removed = remove_duplicates(df)
    report['duplicates_removed'] = duplicates_removed

    # Step 2: Fix data types (before filling missing, so mean works on numeric)
    df, type_changes = fix_data_types(df)
    report['type_changes'] = type_changes

    # Step 3: Handle missing values
    df, missing_changes = handle_missing_values(df)
    report['missing_changes'] = missing_changes

    report['final_shape'] = {'rows': len(df), 'columns': len(df.columns)}

    return df, report
