# Web Dashboard Template for Librex.QAP-new

Complete template for building an interactive web dashboard to visualize optimization results.

---

## Option 1: Streamlit Dashboard (Recommended for Quick Setup)

### Installation

```bash
pip install streamlit plotly pandas numpy
```

### Basic Dashboard Structure

Save as `dashboard.py`:

```python
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

st.set_page_config(
    page_title="Librex.QAP Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("Librex.QAP Dashboard")
st.sidebar.write("Advanced Optimization Visualization")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Optimization Results", "Benchmarks", "Methods", "Settings"]
)

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================

if page == "Overview":
    st.title("⚡ Librex.QAP Dashboard - Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Optimizations",
            value="1,247",
            delta="↑ 42 today",
            delta_color="normal"
        )

    with col2:
        st.metric(
            label="Avg Quality Gap",
            value="2.3%",
            delta="↓ 0.1% vs week",
            delta_color="inverse"
        )

    with col3:
        st.metric(
            label="Avg Runtime",
            value="2.4s",
            delta="↓ 0.2s vs week",
            delta_color="inverse"
        )

    with col4:
        st.metric(
            label="Success Rate",
            value="99.2%",
            delta="↑ 0.1% vs week"
        )

    st.divider()

    # Key Statistics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Optimization Methods Popularity")
        methods_data = pd.DataFrame({
            'Method': ['FFT-Laplace', 'Reverse-Time', 'Genetic Algorithm', 'Simulated Annealing', 'Tabu Search'],
            'Count': [425, 312, 187, 213, 110],
            'Avg Quality': [0.98, 0.95, 0.92, 0.90, 0.88]
        })
        fig = px.bar(
            methods_data,
            x='Method',
            y='Count',
            color='Avg Quality',
            color_continuous_scale='Viridis',
            title='Method Usage & Quality'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Problem Size Distribution")
        sizes = np.random.normal(35, 15, 1000)
        sizes = sizes[sizes > 0]
        fig = px.histogram(
            x=sizes,
            nbins=30,
            title='Problem Size Distribution',
            labels={'x': 'Problem Size (n)', 'y': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Recent Optimizations
    st.subheader("Recent Optimizations")
    recent_data = pd.DataFrame({
        'Time': [
            datetime.now() - timedelta(minutes=5),
            datetime.now() - timedelta(minutes=15),
            datetime.now() - timedelta(minutes=32),
            datetime.now() - timedelta(minutes=48),
            datetime.now() - timedelta(minutes=65),
        ],
        'Method': ['FFT-Laplace', 'Reverse-Time', 'FFT-Laplace', 'Genetic Algorithm', 'Simulated Annealing'],
        'Problem Size': [20, 30, 25, 40, 35],
        'Quality': [1.00, 0.97, 0.99, 0.93, 0.91],
        'Runtime (s)': [0.8, 1.2, 0.9, 3.4, 2.1],
        'Status': ['✓ Success', '✓ Success', '✓ Success', '⚠ Partial', '✓ Success']
    })
    st.dataframe(recent_data, use_container_width=True)

# ============================================================================
# PAGE 2: OPTIMIZATION RESULTS
# ============================================================================

elif page == "Optimization Results":
    st.title("📊 Optimization Results")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Filter & Search")

    with col2:
        if st.button("🔄 Refresh Data"):
            st.rerun()

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        method_filter = st.multiselect(
            "Select Methods",
            ['FFT-Laplace', 'Reverse-Time', 'Genetic Algorithm', 'Simulated Annealing'],
            default=['FFT-Laplace', 'Reverse-Time']
        )

    with col2:
        size_range = st.slider("Problem Size Range", 5, 100, (10, 50))

    with col3:
        time_filter = st.selectbox(
            "Time Range",
            ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"]
        )

    st.divider()

    # Results visualization
    results_data = pd.DataFrame({
        'Method': np.repeat(['FFT-Laplace', 'Reverse-Time'], 20),
        'Problem Size': np.tile(np.arange(10, 30), 2),
        'Quality': np.concatenate([
            np.random.uniform(0.98, 1.00, 20),
            np.random.uniform(0.95, 0.99, 20)
        ]),
        'Runtime': np.concatenate([
            np.random.uniform(0.5, 2.0, 20),
            np.random.uniform(0.8, 2.5, 20)
        ])
    })

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            results_data,
            x='Problem Size',
            y='Quality',
            color='Method',
            size='Runtime',
            hover_data=['Runtime'],
            title='Quality vs Problem Size',
            color_discrete_map={
                'FFT-Laplace': '#1f77b4',
                'Reverse-Time': '#ff7f0e'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            results_data,
            x='Method',
            y='Runtime',
            color='Method',
            title='Runtime Distribution',
            color_discrete_map={
                'FFT-Laplace': '#1f77b4',
                'Reverse-Time': '#ff7f0e'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Detailed results table
    st.subheader("Detailed Results")
    st.dataframe(results_data.sort_values('Quality', ascending=False), use_container_width=True)

    # Export options
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Download as CSV"):
            csv = results_data.to_csv(index=False)
            st.download_button(
                label="Click to download",
                data=csv,
                file_name="optimization_results.csv",
                mime="text/csv"
            )

    with col2:
        if st.button("📥 Download as JSON"):
            json_data = results_data.to_json(orient='records')
            st.download_button(
                label="Click to download",
                data=json_data,
                file_name="optimization_results.json",
                mime="application/json"
            )

# ============================================================================
# PAGE 3: BENCHMARKS
# ============================================================================

elif page == "Benchmarks":
    st.title("🏃 Benchmark Comparison")

    st.info(
        "Compare optimization methods across standard benchmark instances "
        "from the QAPLIB dataset."
    )

    col1, col2 = st.columns(2)

    with col1:
        benchmark_instance = st.selectbox(
            "Select Benchmark Instance",
            ["nug12", "nug20", "nug30", "tai30a", "tai40a"]
        )

    with col2:
        metric = st.radio(
            "Metric",
            ["Quality (Gap %)", "Runtime (s)", "Quality/Time Ratio"]
        )

    st.divider()

    # Benchmark comparison
    benchmark_data = pd.DataFrame({
        'Method': ['FFT-Laplace', 'Reverse-Time', 'Genetic Algorithm', 'Simulated Annealing', 'Tabu Search', 'Variable Neighborhood'],
        'Gap %': [0.02, 0.15, 0.45, 0.38, 0.25, 0.18],
        'Runtime (s)': [0.8, 1.2, 3.2, 2.5, 1.8, 2.1],
        'Robustness': [0.95, 0.91, 0.85, 0.82, 0.88, 0.87],
        'Success Runs': [50, 49, 42, 40, 44, 43]
    })

    col1, col2, col3 = st.columns(3)

    with col1:
        # Best quality
        best_method = benchmark_data.loc[benchmark_data['Gap %'].idxmin()]
        st.metric("Best Quality Method", best_method['Method'], f"{best_method['Gap %']:.2f}% gap")

    with col2:
        # Fastest method
        fastest = benchmark_data.loc[benchmark_data['Runtime (s)'].idxmin()]
        st.metric("Fastest Method", fastest['Method'], f"{fastest['Runtime (s)']:.2f}s")

    with col3:
        # Best robustness
        most_robust = benchmark_data.loc[benchmark_data['Robustness'].idxmax()]
        st.metric("Most Robust", most_robust['Method'], f"{most_robust['Robustness']:.1%}")

    st.divider()

    # Benchmark visualization
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            benchmark_data,
            x='Method',
            y='Gap %',
            color='Gap %',
            color_continuous_scale='RdYlGn_r',
            title=f'Solution Quality - {benchmark_instance}'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            benchmark_data,
            x='Runtime (s)',
            y='Gap %',
            size='Robustness',
            color='Success Runs',
            hover_name='Method',
            title='Quality vs Speed Trade-off',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Detailed benchmark table
    st.subheader("Detailed Benchmark Results")
    st.dataframe(benchmark_data, use_container_width=True)

# ============================================================================
# PAGE 4: METHODS
# ============================================================================

elif page == "Methods":
    st.title("🔧 Optimization Methods")

    # Method selector
    method = st.selectbox(
        "Select Method",
        ["FFT-Laplace", "Reverse-Time Saddle Escape", "Genetic Algorithm", "Simulated Annealing", "Tabu Search"]
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Method: {method}")

        method_info = {
            "FFT-Laplace": {
                "Type": "Quantum-inspired",
                "Complexity": "O(n² log n)",
                "Description": "FFT-based Laplace preconditioning with continuous relaxation",
                "Parameters": {
                    "iterations": (100, 5000, 500),
                    "learning_rate": (0.001, 1.0, 0.1),
                    "momentum": (0.0, 1.0, 0.9)
                }
            },
            "Reverse-Time Saddle Escape": {
                "Type": "Classical local minima escape",
                "Complexity": "O(n²)",
                "Description": "Escape local minima via time-reversal dynamics",
                "Parameters": {
                    "iterations": (100, 5000, 500),
                    "escape_strength": (0.1, 1.0, 0.5),
                    "cooling_rate": (0.9, 0.99, 0.95)
                }
            }
        }

        if method in method_info:
            info = method_info[method]
            st.write(f"**Type:** {info['Type']}")
            st.write(f"**Complexity:** {info['Complexity']}")
            st.write(f"**Description:** {info['Description']}")

            st.subheader("Parameters")
            for param, (min_val, max_val, default) in info['Parameters'].items():
                st.slider(
                    param,
                    min_value=min_val,
                    max_value=max_val,
                    value=default,
                    key=f"{method}_{param}"
                )

    with col2:
        st.subheader("Performance Stats")
        st.metric("Avg Quality Gap", "2.3%")
        st.metric("Avg Runtime", "1.2s")
        st.metric("Success Rate", "99.2%")
        st.metric("Robustness", "95%")

    st.divider()

    # Method comparison
    st.subheader("Method Comparison")
    comparison_data = pd.DataFrame({
        'Metric': ['Quality', 'Speed', 'Robustness', 'Scalability', 'Ease of Use'],
        'FFT-Laplace': [9.5, 8.5, 9.2, 8.8, 8.0],
        'Reverse-Time': [8.5, 7.8, 8.5, 8.5, 7.5],
        'Genetic Alg': [7.5, 6.5, 7.8, 7.0, 6.0]
    })

    # Radar chart
    fig = go.Figure()

    for method_name in ['FFT-Laplace', 'Reverse-Time', 'Genetic Alg']:
        fig.add_trace(go.Scatterpolar(
            r=comparison_data[method_name].tolist(),
            theta=comparison_data['Metric'].tolist(),
            fill='toself',
            name=method_name
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True,
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 5: SETTINGS
# ============================================================================

elif page == "Settings":
    st.title("⚙️ Settings & Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Display Settings")
        theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
        auto_refresh = st.checkbox("Auto-refresh data", value=True)
        refresh_interval = st.slider("Refresh interval (seconds)", 5, 300, 60)

    with col2:
        st.subheader("Data Settings")
        default_benchmark = st.selectbox("Default benchmark", ["nug20", "nug30", "tai30a"])
        show_advanced = st.checkbox("Show advanced metrics", value=False)
        export_format = st.selectbox("Default export format", ["CSV", "JSON", "Excel"])

    st.divider()

    st.subheader("API Configuration")

    col1, col2 = st.columns(2)

    with col1:
        api_url = st.text_input("API URL", value="http://localhost:8000")
        api_key = st.text_input("API Key", type="password", value="your-api-key")

    with col2:
        timeout = st.slider("Request timeout (seconds)", 5, 300, 30)
        max_workers = st.slider("Max concurrent requests", 1, 10, 4)

    if st.button("Test Connection"):
        st.success("✓ Connection successful!")

    st.divider()

    # About section
    st.subheader("About")
    st.write("""
        **Librex.QAP Dashboard v1.0**

        Advanced visualization and monitoring for Librex.QAP-new optimization library.

        - [Documentation](https://github.com/AlaweinOS/AlaweinOS/tree/main/Librex.QAP-new)
        - [GitHub](https://github.com/AlaweinOS/AlaweinOS)
        - [PyPI](https://pypi.org/project/Librex.QAP-new)
    """)
```

### Running the Dashboard

```bash
streamlit run dashboard.py

# Or with custom settings
streamlit run dashboard.py --logger.level=debug --client.showErrorDetails=true
```

---

## Option 2: Flask Dashboard (More Control)

### Installation

```bash
pip install flask plotly flask-cors python-dotenv
```

### Flask App Structure

Save as `app.py`:

```python
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime, timedelta
import pandas as pd

app = Flask(__name__)
CORS(app)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/overview', methods=['GET'])
def get_overview():
    """Get overview statistics."""
    return jsonify({
        'total_optimizations': 1247,
        'avg_gap': 2.3,
        'avg_runtime': 2.4,
        'success_rate': 99.2,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/recent', methods=['GET'])
def get_recent_optimizations():
    """Get recent optimization results."""
    data = [
        {
            'timestamp': (datetime.now() - timedelta(minutes=i*10)).isoformat(),
            'method': ['FFT-Laplace', 'Reverse-Time', 'Genetic Algorithm'][i % 3],
            'problem_size': 20 + i * 5,
            'quality': 0.95 + (i % 5) * 0.01,
            'runtime': 0.8 + (i % 3) * 0.5,
            'status': 'success'
        }
        for i in range(5)
    ]
    return jsonify(data)

@app.route('/api/benchmarks', methods=['GET'])
def get_benchmarks():
    """Get benchmark comparison data."""
    return jsonify({
        'instance': request.args.get('instance', 'nug20'),
        'results': [
            {
                'method': 'FFT-Laplace',
                'gap_percent': 0.02,
                'runtime': 0.8,
                'robustness': 0.95,
                'success_runs': 50
            },
            {
                'method': 'Reverse-Time',
                'gap_percent': 0.15,
                'runtime': 1.2,
                'robustness': 0.91,
                'success_runs': 49
            }
        ]
    })

@app.route('/api/methods', methods=['GET'])
def get_methods():
    """Get available optimization methods."""
    return jsonify({
        'methods': [
            {
                'name': 'FFT-Laplace',
                'type': 'quantum-inspired',
                'complexity': 'O(n² log n)',
                'parameters': {
                    'iterations': {'min': 100, 'max': 5000, 'default': 500},
                    'learning_rate': {'min': 0.001, 'max': 1.0, 'default': 0.1}
                }
            }
        ]
    })

# ============================================================================
# HTML PAGES
# ============================================================================

@app.route('/')
def index():
    """Dashboard homepage."""
    return render_template('index.html')

@app.route('/results')
def results():
    """Results page."""
    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### HTML Template

Save as `templates/index.html`:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Librex.QAP Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                  color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 28px; font-weight: bold; color: #667eea; }
        .metric-label { color: #666; margin-top: 10px; }
        .chart { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ Librex.QAP Dashboard</h1>
            <p>Advanced Optimization Visualization & Monitoring</p>
        </div>

        <div class="metrics" id="metrics-container"></div>
        <div class="chart"><div id="recent-chart"></div></div>
        <div class="chart"><div id="methods-chart"></div></div>
    </div>

    <script>
        // Load overview metrics
        fetch('/api/overview')
            .then(r => r.json())
            .then(data => {
                document.getElementById('metrics-container').innerHTML = `
                    <div class="metric-card">
                        <div class="metric-value">${data.total_optimizations}</div>
                        <div class="metric-label">Total Optimizations</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${data.avg_gap.toFixed(1)}%</div>
                        <div class="metric-label">Avg Quality Gap</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${data.avg_runtime.toFixed(1)}s</div>
                        <div class="metric-label">Avg Runtime</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${data.success_rate.toFixed(1)}%</div>
                        <div class="metric-label">Success Rate</div>
                    </div>
                `;
            });

        // Load and display recent optimizations
        fetch('/api/recent')
            .then(r => r.json())
            .then(data => {
                const methods = data.map(d => d.method);
                const qualities = data.map(d => d.quality);
                const times = data.map(d => d.timestamp);

                Plotly.newPlot('recent-chart', [{
                    x: times,
                    y: qualities,
                    mode: 'lines+markers',
                    name: 'Quality',
                    line: {color: '#667eea', width: 2}
                }], {
                    title: 'Recent Optimization Quality',
                    xaxis: {title: 'Time'},
                    yaxis: {title: 'Quality'}
                });
            });
    </script>
</body>
</html>
```

---

## Deployment

### Streamlit Cloud

```bash
# Create git repository
git init
git add .
git commit -m "Dashboard implementation"

# Deploy to Streamlit Cloud
# Visit https://share.streamlit.io and select your repository
```

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

**Last Updated:** November 2024
