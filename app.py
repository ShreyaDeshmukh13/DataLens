"""
DataLens — Automated Data Analysis Web Interface
Main Flask application with routes for upload, clean, analyze, visualize, and insights.
"""
import pandas as pd
import os
import json
from flask import Flask, render_template, request, session, redirect, url_for, send_file

from auth import init_db, register_user, verify_user
from data_cleaning import clean_data
from eda import get_preview, get_summary_stats, get_column_info, get_missing_values, get_shape_info, \
    get_dataset_warnings, get_analysis_suggestion
from visualization import (
    bar_chart, pie_chart, histogram, scatter_plot,
    correlation_heatmap, suggest_charts, generate_suggested_chart
)
from insights import generate_all_insights
from nlp_query import parse_and_execute_query
from state_manager import save_dashboard_state, load_dashboard_state
import shutil



base_dir = os.path.abspath(os.path.dirname(__file__))
template_dir = os.path.join(base_dir, 'templates')
static_dir = os.path.join(base_dir, 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = 'datalens-secret-key'
app.config['UPLOAD_FOLDER'] = os.path.join(base_dir, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Authentication DB
init_db()


@app.before_request
def require_login():
    """Protect all routes except login, signup, and static files."""
    allowed_routes = ['login_page', 'signup_page', 'static', 'load_sample']
    if request.endpoint not in allowed_routes and 'user' not in session:
        return redirect(url_for('login_page'))


# ─── Auth Routes ─────────────────────────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if verify_user(username, password):
            session['user'] = username
            return redirect(url_for('dashboard_page'))
        else:
            return render_template('login.html', message='⚠️ Invalid username or password.')
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup_page():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        success, msg = register_user(username, password)
        if success:
            session['user'] = username
            return redirect(url_for('dashboard_page'))
        else:
            return render_template('signup.html', message=f'⚠️ {msg}')
    return render_template('signup.html')


# ─── Helper ──────────────────────────────────────────────────────────────────

DATA_CACHE = {}


def get_dataframe():
    """Load dataframe from session filepath. Returns (df, filepath, filename) or (None, None, None)."""
    filepath = session.get('filepath')
    filename = session.get('filename')
    if filepath and os.path.exists(filepath):
        try:
            mtime = os.path.getmtime(filepath)
            if filepath in DATA_CACHE and DATA_CACHE[filepath]['mtime'] == mtime:
                return DATA_CACHE[filepath]['df'], filepath, filename

            df = pd.read_csv(filepath)
            DATA_CACHE[filepath] = {'df': df, 'mtime': mtime}
            return df, filepath, filename
        except Exception:
            pass
    return None, None, None


def common_context():
    """Return common template variables."""
    df, filepath, filename = get_dataframe()
    has_data = df is not None
    shape = get_shape_info(df) if has_data else None
    warnings_list = get_dataset_warnings(df) if has_data else []
    suggestion = get_analysis_suggestion(df) if has_data else ""
    return {
        'has_data': has_data,
        'filename': filename or '',
        'shape': shape,
        'warnings_list': warnings_list,
        'suggestion': suggestion
    }


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def upload_page():
    """Upload Data page."""
    # Clear previous dataset session data without destroying the auth token
    if 'filepath' in session:
        try:
            os.remove(session['filepath'])
        except Exception:
            pass
    session.pop('filepath', None)
    session.pop('filename', None)
    session.pop('suggestions', None)
    return render_template('upload.html',
                           active_tab='upload',
                           has_data=False,
                           filename='',
                           shape=None,
                           preview_html=None,
                           warnings_list=[],
                           suggestion='',
                           message='')


@app.route('/upload', methods=['POST'])
def handle_upload():
    """Handle CSV file upload."""
    file = request.files.get('file')

    if not file or file.filename == '':
        return render_template('upload.html',
                               active_tab='upload',
                               has_data=False,
                               filename='',
                               shape=None,
                               preview_html=None,
                               warnings_list=[],
                               suggestion='',
                               message='⚠️ Please select a CSV file.')

    from werkzeug.utils import secure_filename
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        os.remove(filepath)
        return render_template('upload.html',
                               active_tab='upload',
                               has_data=False,
                               filename='',
                               shape=None,
                               preview_html=None,
                               warnings_list=[],
                               suggestion='',
                               message=f'⚠️ Error reading CSV: {str(e)}')

    session['filepath'] = filepath
    session['filename'] = filename

    preview_html = get_preview(df)
    shape = get_shape_info(df)

    return render_template('upload.html',
                           active_tab='upload',
                           has_data=True,
                           filename=filename,
                           shape=shape,
                           preview_html=preview_html,
                           warnings_list=get_dataset_warnings(df),
                           suggestion=get_analysis_suggestion(df),
                           message='✅ File uploaded successfully! Explore the tabs above.')


@app.route('/clean', methods=['GET'])
def clean_page():
    """Show data cleaning page with detected issues."""
    ctx = common_context()
    df, filepath, filename = get_dataframe()

    if df is None:
        return render_template('clean.html', active_tab='clean', **ctx,
                               preview_html=None, issues=None, report=None, cleaned=False)

    preview_html = get_preview(df)

    # Detect issues before cleaning
    issues = {
        'duplicates': int(df.duplicated().sum()),
        'total_missing': int(df.isnull().sum().sum()),
        'type_issues': sum(1 for col in df.select_dtypes(include=['object']).columns
                           if pd.to_numeric(df[col], errors='coerce').notna().sum() >
                           len(df[col].dropna()) * 0.5)
    }

    return render_template('clean.html', active_tab='clean', **ctx,
                           preview_html=preview_html, issues=issues, report=None, cleaned=False)


@app.route('/clean', methods=['POST'])
def run_cleaning():
    """Execute data cleaning pipeline."""
    ctx = common_context()
    df, filepath, filename = get_dataframe()

    if df is None or not filepath:
        return redirect(url_for('upload_page'))

    # Run cleaning pipeline
    df_cleaned, report = clean_data(df)

    # Save cleaned data back to the same path
    df_cleaned.to_csv(filepath, index=False)

    preview_html = get_preview(df_cleaned)
    ctx['shape'] = get_shape_info(df_cleaned)

    return render_template('clean.html', active_tab='clean', **ctx,
                           preview_html=preview_html, issues=None, report=report, cleaned=True,
                           message='✅ Data cleaned successfully!')


@app.route('/analyze')
def analyze_page():
    """Show EDA page."""
    ctx = common_context()
    df, filepath, filename = get_dataframe()

    if df is None:
        return render_template('analyze.html', active_tab='analyze', **ctx,
                               preview_html=None, stats_html=None,
                               column_info=[], missing_values=[])

    preview_html = get_preview(df)
    stats_html = get_summary_stats(df)
    column_info = get_column_info(df)
    missing_values = get_missing_values(df)

    return render_template('analyze.html', active_tab='analyze', **ctx,
                           preview_html=preview_html,
                           stats_html=stats_html,
                           column_info=column_info,
                           missing_values=missing_values)


@app.route('/visualize', methods=['GET'])
def visualize_page():
    """Show visualization page with smart suggestions."""
    ctx = common_context()
    df, filepath, filename = get_dataframe()

    if df is None:
        return render_template('visualize.html', active_tab='visualize', **ctx,
                               suggestions=[], all_columns=[], numeric_cols=[],
                               categorical_cols=[], chart_html=None)

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    all_columns = df.columns.tolist()
    suggestions = suggest_charts(df)

    # Store suggestions in session as JSON for later retrieval
    session['suggestions'] = json.dumps(suggestions)

    return render_template('visualize.html', active_tab='visualize', **ctx,
                           suggestions=suggestions,
                           all_columns=all_columns,
                           numeric_cols=numeric_cols,
                           categorical_cols=categorical_cols,
                           chart_html=None)


@app.route('/visualize', methods=['POST'])
def generate_chart():
    """Generate a chart (from suggestion or manual)."""
    ctx = common_context()
    df, filepath, filename = get_dataframe()

    if df is None:
        return redirect(url_for('upload_page'))

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    all_columns = df.columns.tolist()

    # Reload suggestions
    suggestions = json.loads(session.get('suggestions', '[]'))

    chart_html = None
    action = request.form.get('action')

    if action == 'suggestion':
        idx = int(request.form.get('suggestion_index', 0))
        if 0 <= idx < len(suggestions):
            chart_html = generate_suggested_chart(df, suggestions[idx])
    elif action == 'manual':
        chart_type = request.form.get('chart_type')
        x = request.form.get('x')
        y = request.form.get('y') or None

        if chart_type == 'BC' and x and y:
            chart_html = bar_chart(df, x, y)
        elif chart_type == 'PC' and x:
            if y:
                chart_html = pie_chart(df, x, y)
            else:
                # Use value_counts for pie without y
                import plotly.express as px
                import plotly.io as pio
                counts = df[x].value_counts()
                fig = px.pie(values=counts.values, names=counts.index, title=f"Pie: Distribution of {x}")
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Poppins, sans-serif", color="#e0e0e0"),
                    title_font=dict(size=18, color="#ffffff"),
                    legend=dict(font=dict(color="#e0e0e0"))
                )
                chart_html = pio.to_html(fig, full_html=False)
        elif chart_type == 'HG' and x:
            chart_html = histogram(df, x, y)
        elif chart_type == 'SC' and x and y:
            chart_html = scatter_plot(df, x, y)

    if not chart_html:
        message = '⚠️ Could not generate chart. Please check your column selections.'
    else:
        message = ''

    return render_template('visualize.html', active_tab='visualize', **ctx,
                           suggestions=suggestions,
                           all_columns=all_columns,
                           numeric_cols=numeric_cols,
                           categorical_cols=categorical_cols,
                           chart_html=chart_html,
                           message=message)


@app.route('/dashboard')
def dashboard_page():
    """Show auto-generated insights main dashboard and filter UI."""
    ctx = common_context()
    df, filepath, filename = get_dataframe()

    if df is None:
        return render_template('dashboard.html', active_tab='dashboard', **ctx,
                               insights={}, charts={}, heatmap_html=None, num_filters=[], cat_filters=[])

    insights = generate_all_insights(df)
    heatmap_html = correlation_heatmap(df)

    charts = {}
    charts['corr'] = heatmap_html

    # Try to generate a main trend chart and a dist chart automatically
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

    try:
        if len(numeric_cols) > 0 and len(categorical_cols) > 0:
            charts['main'] = bar_chart(df, categorical_cols[0], numeric_cols[0])
        elif len(numeric_cols) > 1:
            charts['main'] = scatter_plot(df, numeric_cols[0], numeric_cols[1])

        if len(numeric_cols) > 0:
            charts['dist'] = histogram(df, numeric_cols[0], None)
    except Exception:
        pass

    # Prepare filter metadata
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

    num_filters = []
    for c in numeric_cols:
        try:
            cmin, cmax = float(df[c].min()), float(df[c].max())
            num_filters.append({'name': c, 'min': cmin, 'max': cmax})
        except:
            pass

    cat_filters = []
    for c in categorical_cols:
        try:
            uniq = df[c].dropna().unique().tolist()
            if len(uniq) < 50:
                cat_filters.append({'name': c, 'unique': uniq})
        except:
            pass

    return render_template('dashboard.html', active_tab='dashboard', **ctx,
                           insights=insights, charts=charts, heatmap_html=heatmap_html,
                           num_filters=num_filters, cat_filters=cat_filters)


@app.route('/filter', methods=['POST'])
def filter_dataset():
    """Apply filters and save dataset as <filename>_filtered.csv."""
    df, filepath, filename = get_dataframe()
    if df is None or not filepath:
        return redirect(url_for('upload_page'))

    original_rows = len(df)
    for key, value in request.form.items():
        if not value:
            continue
        try:
            if key.startswith('__min_'):
                col = key.replace('__min_', '')
                if col in df.columns: df = df[df[col] >= float(value)]
            elif key.startswith('__max_'):
                col = key.replace('__max_', '')
                if col in df.columns: df = df[df[col] <= float(value)]
            elif key.startswith('__cat_'):
                col = key.replace('__cat_', '')
                if col in df.columns: df = df[df[col].astype(str) == str(value)]
        except Exception:
            pass

    import uuid
    new_filename = f"filtered_{uuid.uuid4().hex[:8]}_{session.get('filename', 'data.csv')}"
    new_filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
    df.to_csv(new_filepath, index=False)

    session['filepath'] = new_filepath
    session['filename'] = new_filename

    return redirect(url_for('dashboard_page'))


@app.route('/download')
def download_data():
    """Export the current session dataset as CSV."""
    df, filepath, filename = get_dataframe()
    if filepath and os.path.exists(filepath):
        # We rename the outbound file to prefix 'cleaned_' or 'datalens_'
        out_name = f"datalens_{filename}"
        return send_file(filepath, as_attachment=True, download_name=out_name)
    return redirect(url_for('upload_page'))


@app.route('/reset')
def reset_data():
    """Clear dataset data and return to upload."""
    filepath = session.get('filepath')
    if filepath and os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception:
            pass

    session.pop('filepath', None)
    session.pop('filename', None)
    session.pop('suggestions', None)
    return redirect(url_for('upload_page'))


@app.route('/export')
def export_data():
    """Export the current session dataset as CSV for the profile dropdown."""
    df, filepath, filename = get_dataframe()
    if filepath and os.path.exists(filepath):
        out_name = f"export_{filename}"
        return send_file(filepath, as_attachment=True, download_name=out_name)
    return "No data available to export", 400


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    """Simulate logout behavior."""
    filepath = session.get('filepath')
    if filepath and os.path.exists(filepath):
        try:
            os.remove(filepath)
        except:
            pass
    session.clear()

    if request.method == 'POST':
        return {'message': 'Logged out successfully'}
    return redirect(url_for('login_page'))


@app.route('/load_dashboard_simulate', methods=['POST'])
def load_dashboard_simulate():
    """Load a saved dashboard state."""
    state_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"state_{session.get('user', 'guest')}.json")
    if os.path.exists(state_filepath):
        state_data = load_dashboard_state(state_filepath)
        if state_data is not None:
            return {'message': 'Saved dashboard loaded! Applying filters...', 'state': state_data}
    return {'error': 'No saved dashboard found. Save your current metrics in the Dashboard tab first!'}, 404


@app.route('/query', methods=['POST'])
def query_data():
    """Handle natural language query text."""
    df, filepath, filename = get_dataframe()
    if df is None:
        return {'error': 'No dataset loaded'}, 400

    query = request.json.get('query', '')
    if not query:
        return {'error': 'Empty query'}, 400

    html_result, message, format_type = parse_and_execute_query(df, query)
    return {'html_result': html_result, 'message': message, 'format_type': format_type}


@app.route('/save_state', methods=['POST'])
def save_state():
    """Save dashboard configuration logic to JSON."""
    df, filepath, filename = get_dataframe()
    if not filename:
        return {'error': 'No dataset loaded'}, 400

    state_data = request.json
    state_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"state_{session.get('user', 'guest')}.json")

    if save_dashboard_state(state_filepath, state_data):
        return {'message': 'Dashboard state saved securely.'}
    return {'error': 'Failed to save state'}, 500


@app.route('/load_state', methods=['POST'])
def load_state():
    """Load dashboard configuration from uploaded JSON."""
    file = request.files.get('file')
    if not file or not file.filename.endswith('.json'):
        return {'error': 'Invalid state file'}, 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_state.json')
    file.save(filepath)

    state_data = load_dashboard_state(filepath)
    if state_data:
        return {'message': 'State loaded', 'state': state_data}
    return {'error': 'Could not parse state file'}, 400


@app.route('/load_sample/<dataset_name>')
def load_sample(dataset_name):
    """Load a built-in sample dataset."""
    sample_path = os.path.join('static', 'datasets', f"{dataset_name}.csv")
    if not os.path.exists(sample_path):
        return redirect(url_for('upload_page'))

    filename = f"sample_{dataset_name}.csv"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    shutil.copy(sample_path, filepath)

    session['filepath'] = filepath
    session['filename'] = filename
    return redirect(url_for('dashboard_page'))


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Clean up old uploads on startup
    try:
        for f in os.listdir(app.config['UPLOAD_FOLDER']):
            p = os.path.join(app.config['UPLOAD_FOLDER'], f)
            os.remove(p)
    except Exception:
        pass

    app.run(debug=True)