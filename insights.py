"""
Auto Insights Module
Generates simple analytical insights from the dataset.
"""
import pandas as pd
import numpy as np


def get_correlation_insights(df, top_n=5):
    """
    Find the top correlated pairs of numeric columns.
    Returns list of dicts with col1, col2, correlation.
    """
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if len(numeric_df.columns) < 2:
        return []

    corr = numeric_df.corr()
    pairs = []

    for i, col1 in enumerate(corr.columns):
        for col2 in corr.columns[i + 1:]:
            val = round(corr.loc[col1, col2], 3)
            if not np.isnan(val):
                pairs.append({
                    'col1': col1,
                    'col2': col2,
                    'correlation': val,
                    'strength': _correlation_strength(abs(val))
                })

    # Sort by absolute correlation, descending
    pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
    return pairs[:top_n]


def _correlation_strength(val):
    """Classify correlation strength."""
    if val >= 0.8:
        return 'Very Strong'
    elif val >= 0.6:
        return 'Strong'
    elif val >= 0.4:
        return 'Moderate'
    elif val >= 0.2:
        return 'Weak'
    else:
        return 'Very Weak'


def get_extreme_values(df):
    """
    Find highest and lowest values per numeric column.
    Returns list of dicts with column, min, max, range.
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    extremes = []

    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            min_val = round(float(col_data.min()), 2)
            max_val = round(float(col_data.max()), 2)
            extremes.append({
                'column': col,
                'min': min_val,
                'max': max_val,
                'range': round(max_val - min_val, 2),
                'mean': round(float(col_data.mean()), 2),
                'median': round(float(col_data.median()), 2)
            })

    return extremes


def get_frequency_insights(df, top_n=3):
    """
    Find most frequent values in categorical columns.
    Returns list of dicts with column, top values, and counts.
    """
    categorical_cols = df.select_dtypes(exclude=['int64', 'float64']).columns
    insights = []

    for col in categorical_cols:
        value_counts = df[col].value_counts().head(top_n)
        if len(value_counts) > 0:
            top_values = []
            for val, count in value_counts.items():
                percent = round((count / len(df)) * 100, 1)
                top_values.append({
                    'value': str(val),
                    'count': int(count),
                    'percent': percent
                })
            insights.append({
                'column': col,
                'unique_count': int(df[col].nunique()),
                'top_values': top_values
            })

    return insights


def get_trend_summary(df):
    """
    Generate basic trend/descriptive summary for numeric columns.
    Returns list of dicts with column, stats, and skewness indication.
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    trends = []

    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            mean = float(col_data.mean())
            median = float(col_data.median())
            std = float(col_data.std()) if len(col_data) > 1 else 0

            # Determine skewness direction
            if mean > median * 1.05:
                skew = 'Right-skewed (higher values pull mean up)'
            elif mean < median * 0.95:
                skew = 'Left-skewed (lower values pull mean down)'
            else:
                skew = 'Roughly symmetric'

            trends.append({
                'column': col,
                'mean': round(mean, 2),
                'median': round(median, 2),
                'std_dev': round(std, 2),
                'skewness': skew,
                'cv': round((std / mean) * 100, 1) if mean != 0 else 0  # coefficient of variation
            })

    return trends


def generate_all_insights(df):
    """
    Orchestrate all insight functions into a single structured dict.
    """
    return {
        'correlations': get_correlation_insights(df),
        'extremes': get_extreme_values(df),
        'frequencies': get_frequency_insights(df),
        'trends': get_trend_summary(df),
        'row_count': len(df),
        'col_count': len(df.columns)
    }
