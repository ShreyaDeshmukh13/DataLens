"""
Exploratory Data Analysis (EDA) Module
Provides dataset preview, summary statistics, column info, and missing values analysis.
"""
import pandas as pd
import numpy as np


def get_dataset_warnings(df):
    """
    Analyze dataset and return a list of warnings for data issues.
    Checks: High missing (>30%), Duplicate rows, Skewed numeric, Imbalanced categorical.
    """
    warnings = []
    total_rows = len(df)
    if total_rows == 0:
        return ["Dataset is empty."]

    # Duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        warnings.append({'type': 'duplicates', 'message': f"Found {dup_count} duplicate rows."})

    # High Missing Values
    for col in df.columns:
        missing_pct = df[col].isnull().sum() / total_rows
        if missing_pct > 0.3:
            warnings.append({'type': 'missing', 'message': f"Column '{col}' has {missing_pct:.1%} missing values."})

    # Skewed Numeric Columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 2:
            skewness = col_data.skew()
            if abs(skewness) > 2:
                direction = "right" if skewness > 0 else "left"
                warnings.append({'type': 'skewed', 'message': f"Numeric column '{col}' is highly {direction}-skewed."})

    # Imbalanced Categorical
    cat_cols = df.select_dtypes(exclude=['number']).columns
    for col in cat_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            top_freq = col_data.value_counts(normalize=True).iloc[0]
            if top_freq > 0.9 and col_data.nunique() > 1:
                warnings.append({'type': 'imbalanced',
                                 'message': f"Categorical column '{col}' is highly imbalanced (>90% single value)."})

    return warnings


def get_analysis_suggestion(df):
    """
    Suggest an analysis type based on the dataset shape and col types.
    """
    num_cols = len(df.select_dtypes(include=['number']).columns)
    cat_cols = len(df.select_dtypes(exclude=['number']).columns)

    if 'date' in map(str.lower, df.columns) or 'time' in map(str.lower, df.columns) or 'year' in map(str.lower,
                                                                                                     df.columns):
        return "Time Series Analysis (Focus on trends over time)"

    if num_cols >= 2 and cat_cols >= 1:
        return "Comparative Analysis (Compare numeric metrics across categories)"

    if num_cols >= 3:
        return "Correlation Analysis (Explore relationships between numeric variables)"

    if cat_cols >= 2:
        return "Categorical Distribution Analysis (Analyze frequencies and compositions)"

    return "General Exploratory Data Analysis"


def get_preview(df, n=5):
    """Return first n rows as an HTML table."""
    return df.head(n).to_html(
        classes='data-table',
        index=False,
        border=0,
        na_rep='—'
    )


def get_summary_stats(df):
    """Return df.describe() as an HTML table (numeric columns)."""
    stats = df.describe(include='all').round(2)
    return stats.to_html(
        classes='data-table',
        border=0,
        na_rep='—'
    )


def get_column_info(df):
    """
    Return column info as a list of dicts:
    Each dict has: name, dtype, non_null_count, null_count, sample_values
    """
    info = []
    for col in df.columns:
        info.append({
            'name': col,
            'dtype': str(df[col].dtype),
            'non_null': int(df[col].notna().sum()),
            'null': int(df[col].isnull().sum()),
            'unique': int(df[col].nunique()),
            'sample': str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else '—'
        })
    return info


def get_missing_values(df):
    """
    Return missing value info per column as a list of dicts.
    Each dict has: column, missing_count, missing_percent
    """
    missing = []
    total = len(df)
    for col in df.columns:
        count = int(df[col].isnull().sum())
        percent = round((count / total) * 100, 1) if total > 0 else 0
        missing.append({
            'column': col,
            'count': count,
            'percent': percent
        })
    return missing


def get_shape_info(df):
    """Return shape and memory usage of the dataframe."""
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'memory_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        'numeric_cols': len(df.select_dtypes(include=['int64', 'float64']).columns),
        'categorical_cols': len(df.select_dtypes(exclude=['int64', 'float64']).columns)
    }
