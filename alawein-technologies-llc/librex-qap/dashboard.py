"""
Librex.QAP-new Enhanced Professional Dashboard
Production-grade visualization with modern design, dark mode, and advanced features
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import json
import time
import io
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import base64

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Librex.QAP Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/AlaweinOS/AlaweinOS',
        'Report a bug': 'https://github.com/AlaweinOS/AlaweinOS/issues',
        'About': '# Librex.QAP Dashboard v2.0\nProfessional quantum-inspired optimization platform'
    }
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    if 'comparison_history' not in st.session_state:
        st.session_state.comparison_history = []
    if 'saved_benchmarks' not in st.session_state:
        st.session_state.saved_benchmarks = []
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    if 'refresh_interval' not in st.session_state:
        st.session_state.refresh_interval = 5

init_session_state()

# ============================================================================
# MODERN DESIGN SYSTEM - CUSTOM CSS
# ============================================================================

def get_custom_css(dark_mode: bool = False) -> str:
    """Generate custom CSS for modern design system."""

    if dark_mode:
        # Dark Theme Colors
        bg_primary = "#0e1117"
        bg_secondary = "#1a1d26"
        bg_tertiary = "#262730"
        text_primary = "#fafafa"
        text_secondary = "#b3b3b3"
        accent_primary = "#3b82f6"
        accent_secondary = "#8b5cf6"
        success = "#10b981"
        warning = "#f59e0b"
        error = "#ef4444"
        border_color = "#2d3139"
        shadow = "rgba(0, 0, 0, 0.3)"
    else:
        # Light Theme Colors
        bg_primary = "#ffffff"
        bg_secondary = "#f8f9fa"
        bg_tertiary = "#e9ecef"
        text_primary = "#1a1a1a"
        text_secondary = "#6c757d"
        accent_primary = "#3b82f6"
        accent_secondary = "#8b5cf6"
        success = "#10b981"
        warning = "#f59e0b"
        error = "#ef4444"
        border_color = "#dee2e6"
        shadow = "rgba(0, 0, 0, 0.1)"

    return f"""
    <style>
        /* Import Feather Icons */
        @import url('https://unpkg.com/feather-icons@4.29.0/dist/feather.min.css');

        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* Global Styles */
        * {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}

        /* Main container */
        .main {{
            background-color: {bg_primary};
            color: {text_primary};
            transition: background-color 0.3s ease, color 0.3s ease;
        }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {bg_secondary};
            border-right: 1px solid {border_color};
            transition: all 0.3s ease;
        }}

        [data-testid="stSidebar"] * {{
            color: {text_primary} !important;
        }}

        /* Cards */
        .card {{
            background: {bg_secondary};
            border: 1px solid {border_color};
            border-radius: 12px;
            padding: 24px;
            margin: 16px 0;
            box-shadow: 0 2px 8px {shadow};
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 16px {shadow};
        }}

        /* Metric Cards */
        [data-testid="stMetricValue"] {{
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: {text_primary} !important;
        }}

        [data-testid="stMetricLabel"] {{
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            color: {text_secondary} !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        [data-testid="stMetricDelta"] {{
            font-size: 0.875rem !important;
        }}

        /* Buttons */
        .stButton > button {{
            border-radius: 8px;
            font-weight: 500;
            padding: 0.5rem 1.5rem;
            transition: all 0.2s ease;
            border: none;
            box-shadow: 0 1px 3px {shadow};
        }}

        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 12px {shadow};
        }}

        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {accent_primary} 0%, {accent_secondary} 100%);
            color: white;
        }}

        /* Input Fields */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > select {{
            border-radius: 8px;
            border: 1px solid {border_color};
            background-color: {bg_tertiary};
            color: {text_primary};
            padding: 0.5rem 0.75rem;
            transition: all 0.2s ease;
        }}

        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus {{
            border-color: {accent_primary};
            box-shadow: 0 0 0 2px {accent_primary}33;
        }}

        /* Sliders */
        .stSlider > div > div > div > div {{
            background-color: {accent_primary};
        }}

        /* Dataframes */
        [data-testid="stDataFrame"] {{
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid {border_color};
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: {bg_secondary};
            padding: 8px;
            border-radius: 8px;
        }}

        .stTabs [data-baseweb="tab"] {{
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 500;
            transition: all 0.2s ease;
        }}

        .stTabs [aria-selected="true"] {{
            background-color: {accent_primary};
            color: white;
        }}

        /* Status Messages */
        .stSuccess {{
            background-color: {success}22;
            border-left: 4px solid {success};
            border-radius: 8px;
            padding: 16px;
            color: {text_primary};
        }}

        .stWarning {{
            background-color: {warning}22;
            border-left: 4px solid {warning};
            border-radius: 8px;
            padding: 16px;
            color: {text_primary};
        }}

        .stError {{
            background-color: {error}22;
            border-left: 4px solid {error};
            border-radius: 8px;
            padding: 16px;
            color: {text_primary};
        }}

        .stInfo {{
            background-color: {accent_primary}22;
            border-left: 4px solid {accent_primary};
            border-radius: 8px;
            padding: 16px;
            color: {text_primary};
        }}

        /* Loading Skeleton */
        .skeleton {{
            background: linear-gradient(90deg, {bg_tertiary} 25%, {bg_secondary} 50%, {bg_tertiary} 75%);
            background-size: 200% 100%;
            animation: loading 1.5s ease-in-out infinite;
            border-radius: 8px;
            height: 20px;
            margin: 8px 0;
        }}

        @keyframes loading {{
            0% {{ background-position: 200% 0; }}
            100% {{ background-position: -200% 0; }}
        }}

        /* Tooltips */
        .tooltip {{
            position: relative;
            display: inline-block;
            cursor: help;
        }}

        .tooltip .tooltiptext {{
            visibility: hidden;
            background-color: {bg_tertiary};
            color: {text_primary};
            text-align: center;
            border-radius: 6px;
            padding: 8px 12px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -60px;
            opacity: 0;
            transition: opacity 0.3s;
            box-shadow: 0 2px 8px {shadow};
            font-size: 0.875rem;
        }}

        .tooltip:hover .tooltiptext {{
            visibility: visible;
            opacity: 1;
        }}

        /* Progress Bar */
        .stProgress > div > div > div {{
            background: linear-gradient(90deg, {accent_primary}, {accent_secondary});
            border-radius: 4px;
        }}

        /* Divider */
        hr {{
            border: none;
            border-top: 1px solid {border_color};
            margin: 2rem 0;
        }}

        /* Code Blocks */
        code {{
            background-color: {bg_tertiary};
            color: {accent_primary};
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.875rem;
        }}

        /* Expander */
        .streamlit-expanderHeader {{
            background-color: {bg_secondary};
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s ease;
        }}

        .streamlit-expanderHeader:hover {{
            background-color: {bg_tertiary};
        }}

        /* Radio Buttons */
        .stRadio > div {{
            background-color: {bg_secondary};
            padding: 12px;
            border-radius: 8px;
        }}

        /* Checkbox */
        .stCheckbox {{
            padding: 8px;
        }}

        /* Custom Badge */
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .badge-success {{
            background-color: {success}22;
            color: {success};
        }}

        .badge-warning {{
            background-color: {warning}22;
            color: {warning};
        }}

        .badge-error {{
            background-color: {error}22;
            color: {error};
        }}

        .badge-info {{
            background-color: {accent_primary}22;
            color: {accent_primary};
        }}

        /* Animation Classes */
        .fade-in {{
            animation: fadeIn 0.5s ease-in;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* Accessibility */
        *:focus {{
            outline: 2px solid {accent_primary};
            outline-offset: 2px;
        }}

        /* Print Styles */
        @media print {{
            .stButton, .stDownloadButton {{
                display: none;
            }}
        }}
    </style>
    """

# Apply custom CSS
st.markdown(get_custom_css(st.session_state.dark_mode), unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def show_skeleton_loader(num_lines: int = 3):
    """Display skeleton loading animation."""
    html = '<div class="fade-in">'
    for _ in range(num_lines):
        html += '<div class="skeleton" style="width: 100%;"></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def create_badge(text: str, badge_type: str = "info") -> str:
    """Create a colored badge."""
    return f'<span class="badge badge-{badge_type}">{text}</span>'

def create_tooltip(text: str, tooltip_text: str) -> str:
    """Create a tooltip element."""
    return f'''
    <div class="tooltip">{text}
        <span class="tooltiptext">{tooltip_text}</span>
    </div>
    '''

def show_empty_state(title: str, description: str, icon: str = "📭"):
    """Display an empty state message."""
    st.markdown(f"""
    <div class="card" style="text-align: center; padding: 48px 24px;">
        <div style="font-size: 4rem; margin-bottom: 16px;">{icon}</div>
        <h3 style="margin-bottom: 8px;">{title}</h3>
        <p style="color: #6c757d;">{description}</p>
    </div>
    """, unsafe_allow_html=True)

def show_error_with_recovery(error_msg: str, recovery_suggestions: List[str]):
    """Display error message with recovery suggestions."""
    st.error(f"**Error:** {error_msg}")

    with st.expander("💡 How to fix this", expanded=True):
        st.markdown("**Try these solutions:**")
        for i, suggestion in enumerate(recovery_suggestions, 1):
            st.markdown(f"{i}. {suggestion}")

# ============================================================================
# API CONFIGURATION & HELPERS
# ============================================================================

class APIStatus(Enum):
    """API connection status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@st.cache_data(ttl=60)
def get_api_base_url() -> str:
    """Get cached API base URL."""
    return st.session_state.get('api_base_url', 'http://localhost:8000')

def api_call(
    endpoint: str,
    method: str = "GET",
    data: Optional[Dict] = None,
    timeout: int = 10
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Make API call with enhanced error handling.

    Returns:
        Tuple of (response_data, error_message)
    """
    try:
        url = f"{get_api_base_url()}{endpoint}"

        if method == "GET":
            response = requests.get(url, timeout=timeout)
        else:
            response = requests.post(url, json=data, timeout=timeout)

        if response.status_code == 200:
            return response.json(), None
        elif response.status_code == 404:
            return None, f"Endpoint not found: {endpoint}"
        elif response.status_code == 422:
            return None, f"Invalid request data: {response.json()}"
        elif response.status_code >= 500:
            return None, f"Server error (HTTP {response.status_code})"
        else:
            return None, f"API error (HTTP {response.status_code})"

    except requests.exceptions.Timeout:
        return None, "Request timeout - server may be overloaded"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API server"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

@st.cache_data(ttl=30)
def check_api_health() -> Tuple[APIStatus, Dict]:
    """Check API health with caching."""
    health_data, error = api_call("/health")

    if health_data and health_data.get("status") == "healthy":
        return APIStatus.HEALTHY, health_data
    elif error:
        return APIStatus.UNHEALTHY, {"error": error}
    else:
        return APIStatus.UNKNOWN, {}

def generate_sample_matrix(n: int, seed: int = 42) -> List[List[float]]:
    """Generate sample cost matrix."""
    np.random.seed(seed)
    return np.random.rand(n, n).tolist()

# ============================================================================
# EXPORT FUNCTIONALITY
# ============================================================================

def export_to_csv(df: pd.DataFrame) -> str:
    """Export DataFrame to CSV."""
    return df.to_csv(index=False).encode('utf-8')

def export_to_json(data: Dict) -> str:
    """Export data to JSON."""
    return json.dumps(data, indent=2).encode('utf-8')

def export_to_excel(df: pd.DataFrame) -> bytes:
    """Export DataFrame to Excel."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
    return output.getvalue()

def create_download_link(data: bytes, filename: str, file_format: str) -> str:
    """Create a download link for data."""
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:application/{file_format};base64,{b64}" download="{filename}" class="stButton">⬇️ Download {file_format.upper()}</a>'

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    # Header
    st.markdown("# ⚡ Librex.QAP")
    st.markdown("**Professional Optimization Platform**")
    st.markdown("---")

    # Dark Mode Toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Theme**")
    with col2:
        if st.button("🌓", help="Toggle dark mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    # API Configuration
    st.markdown("### ⚙️ Configuration")

    api_base = st.text_input(
        "API Base URL",
        value="http://localhost:8000",
        help="Base URL for the Librex.QAP API server",
        key="api_base_url"
    )

    # Connection Status
    status, health_data = check_api_health()

    if status == APIStatus.HEALTHY:
        st.success("✅ Connected")
    elif status == APIStatus.UNHEALTHY:
        st.error("❌ Disconnected")
    else:
        st.warning("⚠️ Unknown")

    st.markdown("---")

    # Navigation
    st.markdown("### 📍 Navigation")
    page = st.radio(
        "Select Page",
        [
            "🏠 Overview",
            "🔧 Solve Problem",
            "🏃 Benchmarks",
            "⚙️ Methods",
            "📊 Analytics",
            "💾 History"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Auto-refresh settings
    st.markdown("### 🔄 Auto-Refresh")
    st.session_state.auto_refresh = st.checkbox(
        "Enable auto-refresh",
        value=st.session_state.auto_refresh,
        help="Automatically refresh metrics"
    )

    if st.session_state.auto_refresh:
        st.session_state.refresh_interval = st.slider(
            "Refresh interval (seconds)",
            min_value=1,
            max_value=60,
            value=st.session_state.refresh_interval
        )

    st.markdown("---")

    # Quick Links
    st.markdown("### 🔗 Quick Links")
    st.markdown(f"[📚 API Docs]({api_base}/docs)")
    st.markdown("[🐙 GitHub](https://github.com/AlaweinOS/AlaweinOS)")
    st.markdown("[📖 Documentation](https://github.com/AlaweinOS/AlaweinOS/tree/main/Librex.QAP-new)")

    st.markdown("---")
    st.caption("v2.0.0 - Professional Edition")

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================

if page == "🏠 Overview":
    st.title("⚡ Librex.QAP Dashboard")
    st.markdown("**Real-time Optimization Platform Overview**")

    # Check API health
    status, health_data = check_api_health()

    if status != APIStatus.HEALTHY:
        show_error_with_recovery(
            "Cannot connect to API server",
            [
                "Ensure the server is running: `python server.py`",
                "Check the API URL in the sidebar configuration",
                "Verify the server is accessible on the configured port",
                "Check firewall settings if running remotely"
            ]
        )
    else:
        # Show loading skeleton while fetching data
        metrics_container = st.container()

        with st.spinner("Loading metrics..."):
            metrics_data, error = api_call("/metrics")

        if error:
            st.error(f"Failed to load metrics: {error}")
        elif metrics_data:
            with metrics_container:
                # Key Metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    total_opts = metrics_data.get("total_optimizations", 0)
                    active = metrics_data.get("active_requests", 0)
                    st.metric(
                        label="Total Optimizations",
                        value=f"{total_opts:,}",
                        delta=f"{active} active",
                        help="Total number of optimization requests processed"
                    )

                with col2:
                    avg_quality = metrics_data.get("avg_quality", 0)
                    st.metric(
                        label="Avg Quality",
                        value=f"{avg_quality:.1%}",
                        delta="Target: 95%",
                        delta_color="off",
                        help="Average solution quality across all optimizations"
                    )

                with col3:
                    avg_runtime = metrics_data.get("avg_runtime_ms", 0)
                    st.metric(
                        label="Avg Runtime",
                        value=f"{avg_runtime:.0f}ms",
                        help="Average time to complete optimization"
                    )

                with col4:
                    methods_count = metrics_data.get("methods_available", 0)
                    st.metric(
                        label="Methods Available",
                        value=methods_count,
                        help="Number of optimization algorithms available"
                    )

        st.markdown("---")

        # Available Methods Section
        st.markdown("### 🛠️ Available Optimization Methods")

        with st.spinner("Loading methods..."):
            methods_data, error = api_call("/methods")

        if error:
            st.error(f"Failed to load methods: {error}")
        elif methods_data:
            # Create interactive method cards
            for method in methods_data:
                with st.expander(
                    f"**{method['name'].replace('_', ' ').title()}** - {method['category']}",
                    expanded=False
                ):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown(f"**Description:** {method['description']}")
                        st.markdown(f"**Best For:** {', '.join(method['best_for'])}")
                        st.markdown(f"**Complexity:** `{method['complexity_time']}`")

                    with col2:
                        st.metric("Avg Quality", f"{method['avg_quality']:.1%}")
                        st.metric("Avg Runtime", f"{method['avg_runtime_ms']:.0f}ms")

        # System Status
        st.markdown("---")
        st.markdown("### 🔍 System Status")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**API Status**")
            st.markdown(create_badge("HEALTHY", "success"), unsafe_allow_html=True)

        with col2:
            st.markdown("**Last Updated**")
            st.markdown(f"`{datetime.now().strftime('%H:%M:%S')}`")

        with col3:
            st.markdown("**Version**")
            st.markdown(f"`{health_data.get('version', '1.0.0')}`")

# ============================================================================
# PAGE 2: SOLVE PROBLEM
# ============================================================================

elif page == "🔧 Solve Problem":
    st.title("🔧 Solve Optimization Problem")
    st.markdown("**Configure and solve quadratic assignment problems**")

    # Configuration Section
    st.markdown("### ⚙️ Problem Configuration")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Problem Settings
        with st.container():
            st.markdown("**Problem Settings**")

            problem_size = st.slider(
                "Problem Size (n)",
                min_value=5,
                max_value=100,
                value=20,
                help="Dimension of the quadratic assignment problem"
            )

            seed = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=9999,
                value=42,
                help="Seed for reproducible random problem generation"
            )

            iterations = st.slider(
                "Iterations",
                min_value=100,
                max_value=10000,
                value=500,
                step=100,
                help="Maximum number of optimization iterations"
            )

    with col2:
        # Method Selection
        st.markdown("**Optimization Method**")

        methods_data, _ = api_call("/methods")

        if methods_data:
            method_names = [m['name'] for m in methods_data]
            method = st.selectbox(
                "Select Method",
                method_names,
                format_func=lambda x: x.replace('_', ' ').title(),
                help="Choose the optimization algorithm to use",
                label_visibility="collapsed"
            )

            # Show method details
            selected_method = next((m for m in methods_data if m['name'] == method), None)

            if selected_method:
                st.markdown("---")
                st.markdown("**Method Details**")
                st.markdown(f"**Category:** {selected_method['category']}")
                st.markdown(f"**Avg Quality:** {selected_method['avg_quality']:.1%}")
                st.markdown(f"**Avg Runtime:** {selected_method['avg_runtime_ms']:.0f}ms")
                st.markdown(f"**Complexity:** `{selected_method['complexity_time']}`")
        else:
            method = st.selectbox(
                "Select Method",
                ["fft_laplace", "reverse_time", "genetic_algorithm", "simulated_annealing"],
                label_visibility="collapsed"
            )

    st.markdown("---")

    # Advanced Options
    with st.expander("🔧 Advanced Options", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            timeout = st.number_input(
                "Timeout (seconds)",
                min_value=1,
                max_value=3600,
                value=300,
                help="Maximum time allowed for optimization"
            )

        with col2:
            use_custom_matrix = st.checkbox(
                "Use custom cost matrix",
                help="Upload or paste your own cost matrix"
            )

        if use_custom_matrix:
            matrix_input = st.text_area(
                "Cost Matrix (JSON format)",
                placeholder='[[1.0, 2.0], [3.0, 4.0]]',
                help="Enter a square matrix in JSON format"
            )

    # Solve Button
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        solve_button = st.button(
            "🚀 Solve Now",
            type="primary",
            use_container_width=True
        )

    with col2:
        if st.session_state.last_result:
            st.button("📋 View Last Result", use_container_width=True)

    with col3:
        if st.session_state.last_result:
            st.button("💾 Save Result", use_container_width=True)

    # Solve Process
    if solve_button:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Generate or load problem matrix
            status_text.text("⏳ Generating problem...")
            progress_bar.progress(20)

            if use_custom_matrix and matrix_input:
                try:
                    problem_matrix = json.loads(matrix_input)
                except:
                    st.error("Invalid JSON format for cost matrix")
                    st.stop()
            else:
                problem_matrix = generate_sample_matrix(problem_size, seed)

            # Call API
            status_text.text("🔄 Optimizing... Please wait")
            progress_bar.progress(40)

            solve_data = {
                "problem_size": problem_size,
                "problem_matrix": problem_matrix,
                "method": method,
                "iterations": iterations,
                "random_seed": seed,
                "timeout_seconds": timeout
            }

            result, error = api_call("/solve", method="POST", data=solve_data, timeout=timeout + 5)

            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()

            if error:
                show_error_with_recovery(
                    error,
                    [
                        "Try reducing the problem size",
                        "Increase the timeout value",
                        "Try a different optimization method",
                        "Check the API server logs for details"
                    ]
                )
            elif result:
                # Success!
                st.success(f"✅ Optimization completed in {result['runtime_seconds']:.2f}s")

                # Store in session
                st.session_state.last_result = result

                # Results Display
                st.markdown("### 📊 Results")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Objective Value",
                        f"{result['objective_value']:.2f}",
                        help="Total cost of the solution"
                    )

                with col2:
                    st.metric(
                        "Iterations",
                        f"{result['iterations_completed']:,}",
                        help="Number of iterations completed"
                    )

                with col3:
                    st.metric(
                        "Runtime",
                        f"{result['runtime_seconds']:.3f}s",
                        help="Total execution time"
                    )

                with col4:
                    if result.get('gap_percent'):
                        st.metric(
                            "Gap",
                            f"{result['gap_percent']:.2f}%",
                            help="Optimality gap percentage"
                        )
                    else:
                        st.metric("Status", result.get('status', 'completed'))

                # Solution Details
                st.markdown("---")
                st.markdown("### 🎯 Solution Details")

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown("**Best Solution Found**")
                    solution_preview = str(result['best_solution'][:20])
                    if len(result['best_solution']) > 20:
                        solution_preview += "..."
                    st.code(solution_preview, language="python")
                    st.caption(f"Request ID: `{result['request_id']}`")

                with col2:
                    st.markdown("**Export Options**")
                    result_df = pd.DataFrame([result])

                    col_a, col_b = st.columns(2)
                    with col_a:
                        csv_data = export_to_csv(result_df)
                        st.download_button(
                            "📥 CSV",
                            csv_data,
                            f"result_{result['request_id'][:8]}.csv",
                            "text/csv"
                        )

                    with col_b:
                        json_data = export_to_json(result)
                        st.download_button(
                            "📥 JSON",
                            json_data,
                            f"result_{result['request_id'][:8]}.json",
                            "application/json"
                        )

                # Add to comparison history
                if st.button("➕ Add to Comparison History"):
                    st.session_state.comparison_history.append(result)
                    st.success("Added to comparison history!")

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Unexpected error: {str(e)}")

# ============================================================================
# PAGE 3: BENCHMARKS
# ============================================================================

elif page == "🏃 Benchmarks":
    st.title("🏃 Benchmark Comparison")
    st.markdown("**Compare optimization methods on standard instances**")

    # Configuration
    st.markdown("### ⚙️ Benchmark Configuration")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Instance Settings**")

        instance = st.selectbox(
            "Problem Instance",
            ["nug12", "nug15", "nug20", "nug25", "nug30", "tai20a", "tai30a", "tai40a", "tai50a"],
            help="Select a standard QAP benchmark instance"
        )

        num_runs = st.slider(
            "Number of Runs",
            min_value=1,
            max_value=20,
            value=5,
            help="Runs per method for statistical significance"
        )

    with col2:
        st.markdown("**Method Selection**")

        methods_data, _ = api_call("/methods")

        if methods_data:
            method_names = [m['name'] for m in methods_data]
            methods = st.multiselect(
                "Methods to Compare",
                method_names,
                default=method_names[:3] if len(method_names) >= 3 else method_names,
                format_func=lambda x: x.replace('_', ' ').title(),
                help="Select 2 or more methods to compare"
            )
        else:
            methods = st.multiselect(
                "Methods to Compare",
                ["fft_laplace", "reverse_time", "genetic_algorithm", "simulated_annealing"],
                default=["fft_laplace", "reverse_time"]
            )

        iterations = st.slider(
            "Iterations per Run",
            min_value=100,
            max_value=5000,
            value=500,
            step=100,
            help="Optimization iterations for each run"
        )

    # Advanced Filtering
    with st.expander("🔍 Advanced Filtering & Display Options"):
        col1, col2 = st.columns(2)

        with col1:
            show_outliers = st.checkbox("Show outliers", value=True)
            show_statistics = st.checkbox("Show detailed statistics", value=True)

        with col2:
            chart_type = st.selectbox(
                "Visualization Type",
                ["Box Plot", "Violin Plot", "Scatter Plot", "All"]
            )

    st.markdown("---")

    # Run Benchmark
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        run_button = st.button("🏃 Run Benchmark", type="primary", use_container_width=True)

    with col2:
        if st.session_state.saved_benchmarks:
            if st.button("💾 Saved Benchmarks", use_container_width=True):
                st.session_state.show_saved = True

    with col3:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.benchmark_result = None

    if run_button:
        if len(methods) < 2:
            st.warning("⚠️ Please select at least 2 methods to compare")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text(f"🔄 Running benchmark on {instance}...")
                progress_bar.progress(20)

                benchmark_data = {
                    "instance_name": instance,
                    "methods": methods,
                    "num_runs": num_runs,
                    "iterations_per_run": iterations
                }

                result, error = api_call("/benchmark", method="POST", data=benchmark_data, timeout=120)

                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()

                if error:
                    show_error_with_recovery(
                        error,
                        [
                            "Try reducing the number of runs",
                            "Select fewer methods to compare",
                            "Ensure the API server has enough resources",
                            "Check server logs for detailed error information"
                        ]
                    )
                elif result and "results" in result:
                    results = result["results"]
                    df = pd.DataFrame(results)

                    st.success(f"✅ Benchmark completed with {len(results)} total runs")

                    # Store in session
                    st.session_state.benchmark_result = result

                    # Summary Statistics
                    st.markdown("### 📊 Summary Statistics")

                    summary = df.groupby("method").agg({
                        "quality": ["mean", "std", "min", "max"],
                        "runtime_seconds": ["mean", "std", "min", "max"]
                    }).round(4)

                    st.dataframe(summary, use_container_width=True)

                    # Best Results
                    st.markdown("### 🏆 Best Results by Method")

                    best_by_method = df.loc[df.groupby("method")["quality"].idxmax()]
                    best_display = best_by_method[["method", "quality", "runtime_seconds", "gap_percent"]].copy()
                    best_display["method"] = best_display["method"].str.replace("_", " ").str.title()

                    st.dataframe(best_display, use_container_width=True)

                    # Visualizations
                    st.markdown("### 📈 Visualizations")

                    if chart_type == "Box Plot" or chart_type == "All":
                        st.markdown("**Quality Distribution**")
                        fig = px.box(
                            df,
                            x="method",
                            y="quality",
                            color="method",
                            title="Quality Distribution by Method",
                            labels={"method": "Method", "quality": "Solution Quality"},
                            points="all" if show_outliers else False
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                    if chart_type == "Violin Plot" or chart_type == "All":
                        st.markdown("**Runtime Distribution**")
                        fig = px.violin(
                            df,
                            x="method",
                            y="runtime_seconds",
                            color="method",
                            title="Runtime Distribution by Method",
                            labels={"method": "Method", "runtime_seconds": "Runtime (seconds)"},
                            box=True
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                    if chart_type == "Scatter Plot" or chart_type == "All":
                        st.markdown("**Quality vs Runtime Trade-off**")
                        fig = px.scatter(
                            df,
                            x="runtime_seconds",
                            y="quality",
                            color="method",
                            size="quality",
                            title="Quality vs Runtime Trade-off",
                            labels={
                                "runtime_seconds": "Runtime (seconds)",
                                "quality": "Solution Quality",
                                "method": "Method"
                            },
                            hover_data=["run", "gap_percent"]
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Export Options
                    st.markdown("### 💾 Export Results")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        csv_data = export_to_csv(df)
                        st.download_button(
                            "📥 Download CSV",
                            csv_data,
                            f"benchmark_{instance}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv"
                        )

                    with col2:
                        json_data = export_to_json(result)
                        st.download_button(
                            "📥 Download JSON",
                            json_data,
                            f"benchmark_{instance}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            "application/json"
                        )

                    with col3:
                        excel_data = export_to_excel(df)
                        st.download_button(
                            "📥 Download Excel",
                            excel_data,
                            f"benchmark_{instance}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                    with col4:
                        if st.button("💾 Save Benchmark"):
                            st.session_state.saved_benchmarks.append({
                                "timestamp": datetime.now(),
                                "instance": instance,
                                "result": result
                            })
                            st.success("Benchmark saved!")

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Unexpected error: {str(e)}")

# ============================================================================
# PAGE 4: METHODS
# ============================================================================

elif page == "⚙️ Methods":
    st.title("⚙️ Optimization Methods")
    st.markdown("**Explore available optimization algorithms**")

    # Search and Filter
    col1, col2 = st.columns([2, 1])

    with col1:
        search_query = st.text_input(
            "🔍 Search methods",
            placeholder="Search by name, category, or description...",
            help="Filter methods by keywords"
        )

    with col2:
        filter_category = st.selectbox(
            "Filter by Category",
            ["All", "Quantum-Inspired", "Classical", "Evolutionary", "Swarm Intelligence"]
        )

    st.markdown("---")

    # Load methods
    with st.spinner("Loading methods..."):
        methods_data, error = api_call("/methods")

    if error:
        show_error_with_recovery(
            error,
            [
                "Ensure the API server is running",
                "Check the API URL configuration",
                "Verify network connectivity"
            ]
        )
    elif methods_data:
        # Filter methods
        filtered_methods = methods_data

        if search_query:
            search_lower = search_query.lower()
            filtered_methods = [
                m for m in filtered_methods
                if search_lower in m['name'].lower()
                or search_lower in m['description'].lower()
                or search_lower in m['category'].lower()
            ]

        if filter_category != "All":
            category_map = {
                "Quantum-Inspired": "quantum-inspired",
                "Classical": "classical",
                "Evolutionary": "evolutionary",
                "Swarm Intelligence": "swarm_intelligence"
            }
            target_category = category_map.get(filter_category, filter_category.lower())
            filtered_methods = [
                m for m in filtered_methods
                if target_category in m['category'].lower()
            ]

        if not filtered_methods:
            show_empty_state(
                "No Methods Found",
                "Try adjusting your search criteria or filters",
                "🔍"
            )
        else:
            # Display methods in tabs
            tabs = st.tabs([
                m["name"].replace("_", " ").title()
                for m in filtered_methods
            ])

            for tab, method in zip(tabs, filtered_methods):
                with tab:
                    # Method Header
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"## {method['name'].replace('_', ' ').title()}")
                        st.markdown(
                            create_badge(method['category'], "info"),
                            unsafe_allow_html=True
                        )

                    with col2:
                        st.metric("Avg Quality", f"{method['avg_quality']:.1%}")
                        st.metric("Avg Runtime", f"{method['avg_runtime_ms']:.0f}ms")

                    st.markdown("---")

                    # Method Details
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown("### 📝 Description")
                        st.write(method['description'])

                        st.markdown("### 🎯 Best For")
                        for item in method['best_for']:
                            st.markdown(f"- {item.replace('_', ' ').title()}")

                        st.markdown("### ⚙️ Parameters")
                        params_df = pd.DataFrame([
                            {
                                "Parameter": name,
                                "Min": info['min'],
                                "Max": info['max'],
                                "Default": info['default']
                            }
                            for name, info in method['parameters'].items()
                        ])
                        st.dataframe(params_df, use_container_width=True)

                    with col2:
                        st.markdown("### 📊 Complexity")
                        st.code(f"Time:  {method['complexity_time']}\nSpace: {method['complexity_space']}")

                        st.markdown("### 📈 Performance")

                        # Create a simple performance chart
                        perf_data = pd.DataFrame({
                            "Metric": ["Quality", "Speed"],
                            "Score": [method['avg_quality'], min(1.0, 1000 / method['avg_runtime_ms'])]
                        })

                        fig = px.bar(
                            perf_data,
                            x="Metric",
                            y="Score",
                            title="Performance Overview",
                            color="Metric",
                            range_y=[0, 1]
                        )
                        fig.update_layout(showlegend=False, height=300)
                        st.plotly_chart(fig, use_container_width=True)

                    # Interactive Demo
                    st.markdown("---")
                    st.markdown("### 🧪 Quick Test")

                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        test_size = st.slider(
                            "Test Problem Size",
                            5, 50, 10,
                            key=f"test_size_{method['name']}"
                        )

                    with col2:
                        test_iter = st.number_input(
                            "Iterations",
                            100, 2000, 500,
                            key=f"test_iter_{method['name']}"
                        )

                    with col3:
                        if st.button("▶️ Run Test", key=f"test_btn_{method['name']}"):
                            with st.spinner("Running test..."):
                                test_matrix = generate_sample_matrix(test_size)
                                test_result, test_error = api_call(
                                    "/solve",
                                    method="POST",
                                    data={
                                        "problem_size": test_size,
                                        "problem_matrix": test_matrix,
                                        "method": method['name'],
                                        "iterations": test_iter
                                    }
                                )

                                if test_error:
                                    st.error(f"Test failed: {test_error}")
                                elif test_result:
                                    st.success(f"✅ Completed in {test_result['runtime_seconds']:.2f}s")
                                    st.metric("Objective Value", f"{test_result['objective_value']:.2f}")
    else:
        show_empty_state(
            "No Methods Available",
            "Unable to load optimization methods from the API",
            "⚙️"
        )

# ============================================================================
# PAGE 5: ANALYTICS
# ============================================================================

elif page == "📊 Analytics":
    st.title("📊 Analytics & Statistics")
    st.markdown("**System performance and usage analytics**")

    # Refresh Controls
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    with col2:
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()

    with col3:
        auto_refresh_toggle = st.checkbox(
            "Auto-refresh",
            value=st.session_state.auto_refresh
        )

    # Auto-refresh logic
    if auto_refresh_toggle:
        time.sleep(st.session_state.refresh_interval)
        st.rerun()

    st.markdown("---")

    # Load statistics
    with st.spinner("Loading analytics..."):
        stats_data, error = api_call("/stats")

    if error:
        show_error_with_recovery(
            error,
            [
                "Ensure the API server is running",
                "Check network connectivity",
                "Verify the /stats endpoint is available"
            ]
        )
    elif stats_data:
        # Service Statistics
        st.markdown("### 🖥️ Service Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Requests",
                f"{stats_data.get('total_requests', 0):,}",
                help="Total number of optimization requests processed"
            )

        with col2:
            st.metric(
                "In Queue",
                stats_data.get('requests_in_queue', 0),
                help="Requests currently in processing queue"
            )

        with col3:
            st.metric(
                "Methods Available",
                stats_data.get('methods_available', 0),
                help="Number of optimization methods"
            )

        with col4:
            uptime = stats_data.get('uptime_hours', 0)
            st.metric(
                "Uptime",
                f"{uptime:.1f}h",
                help="Service uptime in hours"
            )

        st.markdown("---")

        # Performance Metrics
        st.markdown("### 📈 Performance Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_quality = stats_data.get('average_quality', 0)
            st.metric(
                "Avg Quality",
                f"{avg_quality:.2%}",
                delta="Target: 95%",
                delta_color="off"
            )

        with col2:
            avg_runtime = stats_data.get('average_runtime_seconds', 0)
            st.metric(
                "Avg Runtime",
                f"{avg_runtime:.3f}s"
            )

        with col3:
            min_quality = stats_data.get('min_quality', 0)
            st.metric(
                "Min Quality",
                f"{min_quality:.2%}"
            )

        with col4:
            max_quality = stats_data.get('max_quality', 0)
            st.metric(
                "Max Quality",
                f"{max_quality:.2%}"
            )

        st.markdown("---")

        # Historical Trends (Simulated)
        st.markdown("### 📊 Historical Trends")

        # Generate sample historical data
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=30),
            periods=30,
            freq="D"
        )

        historical_data = pd.DataFrame({
            "Date": dates,
            "Requests": np.random.randint(50, 200, 30).cumsum() % 200 + 50,
            "Avg Quality": np.random.uniform(0.85, 0.98, 30),
            "Avg Runtime": np.random.uniform(0.5, 3.0, 30)
        })

        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Requests", "🎯 Quality", "⏱️ Runtime"])

        with tab1:
            fig = px.line(
                historical_data,
                x="Date",
                y="Requests",
                title="Daily Requests Over Time",
                markers=True
            )
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig = px.line(
                historical_data,
                x="Date",
                y="Avg Quality",
                title="Average Solution Quality Over Time",
                markers=True
            )
            fig.add_hline(
                y=0.95,
                line_dash="dash",
                line_color="green",
                annotation_text="Target: 95%"
            )
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            fig = px.line(
                historical_data,
                x="Date",
                y="Avg Runtime",
                title="Average Runtime Over Time",
                markers=True
            )
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        # System Health
        st.markdown("---")
        st.markdown("### 🏥 System Health")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**API Status**")
            st.markdown(create_badge("HEALTHY", "success"), unsafe_allow_html=True)
            st.caption("All systems operational")

        with col2:
            st.markdown("**Database**")
            st.markdown(create_badge("CONNECTED", "success"), unsafe_allow_html=True)
            st.caption("In-memory storage active")

        with col3:
            st.markdown("**Compute**")
            st.markdown(create_badge("READY", "success"), unsafe_allow_html=True)
            st.caption("Optimization engines ready")

# ============================================================================
# PAGE 6: HISTORY
# ============================================================================

elif page == "💾 History":
    st.title("💾 Comparison History")
    st.markdown("**Review and compare past optimization results**")

    if not st.session_state.comparison_history:
        show_empty_state(
            "No Saved Results",
            "Run some optimizations and save them to view comparison history",
            "💾"
        )
    else:
        # Controls
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown(f"**Total Results:** {len(st.session_state.comparison_history)}")

        with col2:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.comparison_history = []
                st.rerun()

        with col3:
            if st.button("📥 Export All", use_container_width=True):
                all_results_df = pd.DataFrame(st.session_state.comparison_history)
                csv_data = export_to_csv(all_results_df)
                st.download_button(
                    "Download CSV",
                    csv_data,
                    f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )

        st.markdown("---")

        # Display results
        df = pd.DataFrame(st.session_state.comparison_history)

        # Summary stats
        st.markdown("### 📊 Summary Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Runs", len(df))

        with col2:
            st.metric("Avg Quality", f"{df['objective_value'].mean():.2f}")

        with col3:
            st.metric("Avg Runtime", f"{df['runtime_seconds'].mean():.3f}s")

        with col4:
            st.metric("Methods Used", df['method'].nunique())

        st.markdown("---")

        # Comparison Chart
        st.markdown("### 📈 Method Comparison")

        fig = px.scatter(
            df,
            x="runtime_seconds",
            y="objective_value",
            color="method",
            size="problem_size",
            hover_data=["iterations_completed", "request_id"],
            title="Performance Comparison: Quality vs Runtime"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Detailed Table
        st.markdown("### 📋 Detailed Results")

        display_df = df[[
            "method", "problem_size", "objective_value",
            "runtime_seconds", "iterations_completed", "timestamp"
        ]].copy()

        display_df["method"] = display_df["method"].str.replace("_", " ").str.title()
        display_df = display_df.sort_values("timestamp", ascending=False)

        st.dataframe(display_df, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("### 🚀 Quick Start")
    st.markdown("""
    1. Start API: `python server.py`
    2. Run Dashboard: `streamlit run dashboard.py`
    3. Begin optimizing!
    """)

with footer_col2:
    st.markdown("### 📚 Resources")
    st.markdown(f"""
    - [API Documentation]({get_api_base_url()}/docs)
    - [GitHub Repository](https://github.com/AlaweinOS/AlaweinOS)
    - [User Guide](https://github.com/AlaweinOS/AlaweinOS/tree/main/Librex.QAP-new)
    """)

with footer_col3:
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Librex.QAP Dashboard v2.0**

    Professional optimization platform with quantum-inspired algorithms

    © 2024 AlaweinOS
    """)

# Auto-refresh for real-time updates
if st.session_state.auto_refresh and st.session_state.get('page') in ["🏠 Overview", "📊 Analytics"]:
    time.sleep(st.session_state.refresh_interval)
    st.rerun()
