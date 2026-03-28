"""
Visualization Module
Chart generation using Plotly with smart suggestion engine.
"""
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np


# ─── Shared Layout Theme ─────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    template='plotly_dark',
    plot_bgcolor='#050505',
    paper_bgcolor='#0F0F0F',
    font=dict(family="Inter, -apple-system, sans-serif", color="#EDEDED", size=13),
    title_font=dict(size=18, color="#EDEDED"),
    xaxis=dict(gridcolor='#1A1A1A', zerolinecolor='#1A1A1A'),
    yaxis=dict(gridcolor='#1A1A1A', zerolinecolor='#1A1A1A'),
    margin=dict(l=50, r=30, t=60, b=50),
    legend=dict(font=dict(color="#888888"), bgcolor='rgba(0,0,0,0)'),
    colorway=['#22C55E', '#A855F7', '#EC4899', '#F59E0B', '#22D3EE',
              '#818CF8', '#FB923C', '#2DD4BF', '#C084FC', '#F87171']
)


# ─── Chart Functions ──────────────────────────────────────────────────────────

def bar_chart(df, x, y, title="Bar Chart"):
    fig = px.bar(df, x=x, y=y, title=title, color_discrete_sequence=['#6c5ce7'])
    fig.update_layout(**CHART_LAYOUT)
    return pio.to_html(fig, full_html=False)


def pie_chart(df, x, y, title="Pie Chart"):
    fig = px.pie(df, values=y, names=x, title=title)
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0F0F0F',
        plot_bgcolor='#050505',
        font=dict(family="Inter, -apple-system, sans-serif", color="#EDEDED", size=13),
        title_font=dict(size=18, color="#EDEDED"),
        legend=dict(font=dict(color="#888888"), bgcolor='rgba(0,0,0,0)'),
        colorway=['#22C55E', '#A855F7', '#EC4899', '#F59E0B', '#22D3EE']
    )
    return pio.to_html(fig, full_html=False)


def histogram(df, x, y=None, title="Histogram"):
    fig = px.histogram(df, x=x, y=y, title=title, color_discrete_sequence=['#00cec9'])
    fig.update_layout(**CHART_LAYOUT)
    return pio.to_html(fig, full_html=False)


def scatter_plot(df, x, y, title="Scatter Plot"):
    fig = px.scatter(df, x=x, y=y, title=title, color_discrete_sequence=['#22C55E'])
    fig.update_layout(**CHART_LAYOUT)
    return pio.to_html(fig, full_html=False)


def correlation_heatmap(df, title="Correlation Matrix"):
    """Generate a correlation heatmap for numeric columns."""
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if len(numeric_df.columns) < 2:
        return None

    corr = numeric_df.corr().round(2)

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale='Viridis',
        text=corr.values,
        texttemplate='%{text}',
        textfont=dict(size=12, color='white'),
        hovertemplate='%{x} vs %{y}: %{z}<extra></extra>'
    ))
    fig.update_layout(
        title=title,
        template='plotly_dark',
        plot_bgcolor='#050505',
        paper_bgcolor='#0F0F0F',
        font=dict(family="Inter, -apple-system, sans-serif", color="#EDEDED", size=13),
        title_font=dict(size=18, color="#EDEDED"),
        margin=dict(l=80, r=30, t=60, b=80),
        xaxis=dict(tickangle=-45),
    )
    return pio.to_html(fig, full_html=False)


# ─── Smart Suggestion Engine ─────────────────────────────────────────────────

def suggest_charts(df):
    """
    Analyze column types and suggest appropriate chart types.
    Returns a list of suggestion dicts.
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

    suggestions = []

    # 1 numeric → Histogram
    for col in numeric_cols:
        suggestions.append({
            'type': 'Histogram',
            'icon': '📊',
            'description': f'Distribution of {col}',
            'x': col,
            'y': None,
            'chart_code': 'HG'
        })

    # 2 numeric → Scatter
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i + 1:]:
            suggestions.append({
                'type': 'Scatter Plot',
                'icon': '🔵',
                'description': f'{col1} vs {col2}',
                'x': col1,
                'y': col2,
                'chart_code': 'SC'
            })

    # Categorical + Numeric → Bar chart
    for cat_col in categorical_cols:
        for num_col in numeric_cols:
            # Only suggest if categorical has reasonable cardinality
            if df[cat_col].nunique() <= 20:
                suggestions.append({
                    'type': 'Bar Chart',
                    'icon': '📈',
                    'description': f'{num_col} by {cat_col}',
                    'x': cat_col,
                    'y': num_col,
                    'chart_code': 'BC'
                })

    # Only categorical → Pie chart
    for cat_col in categorical_cols:
        if df[cat_col].nunique() <= 10:
            suggestions.append({
                'type': 'Pie Chart',
                'icon': '🥧',
                'description': f'Distribution of {cat_col}',
                'x': cat_col,
                'y': None,
                'chart_code': 'PC'
            })

    # Limit to top 8 suggestions to avoid overwhelming the user
    return suggestions[:8]


def generate_suggested_chart(df, suggestion):
    """Generate a chart based on a suggestion dict."""
    chart_code = suggestion['chart_code']
    x = suggestion['x']
    y = suggestion['y']
    desc = suggestion['description']

    if chart_code == 'HG':
        return histogram(df, x, title=f"Histogram: {desc}")
    elif chart_code == 'SC':
        return scatter_plot(df, x, y, title=f"Scatter: {desc}")
    elif chart_code == 'BC':
        return bar_chart(df, x, y, title=f"Bar: {desc}")
    elif chart_code == 'PC':
        # For pie chart without a y-axis, use value_counts
        counts = df[x].value_counts()
        fig = px.pie(values=counts.values, names=counts.index, title=f"Pie: {desc}")
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0F0F0F',
            plot_bgcolor='#050505',
            font=dict(family="Inter, -apple-system, sans-serif", color="#EDEDED", size=13),
            title_font=dict(size=18, color="#EDEDED"),
            legend=dict(font=dict(color="#888888"), bgcolor='rgba(0,0,0,0)'),
            colorway=['#22C55E', '#A855F7', '#EC4899', '#F59E0B', '#22D3EE']
        )
        return pio.to_html(fig, full_html=False)
    return None
