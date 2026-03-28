import pandas as pd
import plotly.express as px
import plotly.io as pio


def parse_and_execute_query(df, query):
    """
    Basic NLP simulation using keyword matching for tabular data.
    Returns (html_result, message, format_type)
    format_type can be 'html_table' or 'chart'
    """
    query = query.lower()
    cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()

    # Detect requested columns in query
    found_cols = [c for c in cols if c.lower() in query.replace('_', ' ')]

    # Defaults
    format_type = 'html_table'
    message = "Query executed successfully."
    html_result = ""

    if not found_cols:
        return "", "Could not identify any matching columns in the query. Please mention column names exactly.", "error"

    # Intent detection
    is_plot = any(k in query for k in ['plot', 'chart', 'graph', 'visualize'])
    is_highest = any(k in query for k in ['highest', 'max', 'top', 'largest'])
    is_lowest = any(k in query for k in ['lowest', 'min', 'bottom', 'smallest'])
    is_average = any(k in query for k in ['average', 'mean'])
    is_groupby = 'by' in query or 'groupby' in query or 'per' in query

    # Layout for charts
    layout_opts = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        title_font=dict(size=18, color="#ffffff"),
    )

    try:
        # Case 1: Group by
        if is_groupby and len(found_cols) >= 2:
            # Assumes first text column is category, first numeric is value
            cat_col = next((c for c in found_cols if c in categorical_cols), found_cols[0])
            num_col = next((c for c in found_cols if c in numeric_cols), found_cols[1])

            if is_average:
                res = df.groupby(cat_col)[num_col].mean().reset_index()
                title = f'Average {num_col} by {cat_col}'
            elif is_highest or is_lowest:
                res = df.groupby(cat_col)[num_col].max() if is_highest else df.groupby(cat_col)[num_col].min()
                res = res.reset_index()
                title = f'{"Highest" if is_highest else "Lowest"} {num_col} by {cat_col}'
            else:
                res = df.groupby(cat_col)[num_col].sum().reset_index()
                title = f'Total {num_col} by {cat_col}'

            if is_plot:
                fig = px.bar(res, x=cat_col, y=num_col, title=title, template='plotly_dark')
                fig.update_layout(**layout_opts)
                html_result = pio.to_html(fig, full_html=False)
                format_type = 'chart'
            else:
                html_result = res.to_html(classes='data-table bg-gray-900', index=False, border=0)

            return html_result, f"Showing {title}", format_type

        # Case 2: Highest / Lowest over a regular list
        if is_highest or is_lowest:
            num_col = next((c for c in found_cols if c in numeric_cols), None)
            if num_col:
                res = df.nlargest(5, num_col) if is_highest else df.nsmallest(5, num_col)
                # Keep target col and maybe a categorical col for context
                disp_cols = [c for c in [next((x for x in categorical_cols), None), num_col] if c]
                res = res[disp_cols]
                html_result = res.to_html(classes='data-table bg-gray-900', index=False, border=0)
                return html_result, f"Showing top 5 rows for {'highest' if is_highest else 'lowest'} {num_col}", format_type

        # Case 3: Simple Plotting
        if is_plot:
            if len(found_cols) == 1:
                col = found_cols[0]
                if col in numeric_cols:
                    fig = px.histogram(df, x=col, title=f'Distribution of {col}', template='plotly_dark')
                else:
                    fig = px.pie(df, names=col, title=f'Distribution of {col}', template='plotly_dark')
            elif len(found_cols) >= 2:
                # Scatter if two num, Bar if 1 cat 1 num
                c1, c2 = found_cols[0], found_cols[1]
                if c1 in numeric_cols and c2 in numeric_cols:
                    fig = px.scatter(df, x=c1, y=c2, title=f'{c1} vs {c2}', template='plotly_dark')
                else:
                    cat_col = c1 if c1 in categorical_cols else c2
                    num_col = c2 if c2 in numeric_cols else c1
                    if cat_col and num_col:
                        # Pre-aggregate sum for smooth bar plotting
                        agg = df.groupby(cat_col)[num_col].sum().reset_index().nlargest(20, num_col)
                        fig = px.bar(agg, x=cat_col, y=num_col, title=f'{num_col} by {cat_col}', template='plotly_dark')
                    else:
                        return "", "Could not determine plot type. Valid pairs: (Numeric, Numeric) or (Category, Numeric).", "error"

            fig.update_layout(**layout_opts)
            html_result = pio.to_html(fig, full_html=False)
            format_type = 'chart'
            return html_result, message, format_type

        # Default fallback: Just show the columns
        res = df[found_cols].head(10)
        html_result = res.to_html(classes='data-table bg-gray-900', index=False, border=0)
        return html_result, f"Showing sample data for columns: {', '.join(found_cols)}", format_type

    except Exception as e:
        return "", f"Error processing query: {str(e)}", "error"
