import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import time
import os
import sys
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Import your agent function
try:
    from agent_at_work import agent_at_work

    AGENT_AVAILABLE = True
except ImportError as e:
    st.error(f"Could not import agent_at_work.py: {e}")
    AGENT_AVAILABLE = False

    # Fallback function if import fails
    def agent_at_work(model, path, user_query):
        return {
            "status": "error",
            "message": "agent_at_work.py not found or import failed",
            "data": {"error": "Please ensure agent_at_work.py is in the same folder"},
        }


# Configure page
st.set_page_config(
    page_title="AI Agent Interface",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for futuristic dark theme
st.markdown(
    """
<style>
    /* Import futuristic font */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    /* Main theme colors */
    :root {
        --primary-bg: #0a0a0a;
        --secondary-bg: #1a1a2e;
        --accent-color: #00d4ff;
        --accent-glow: #00d4ff33;
        --text-primary: #ffffff;
        --text-secondary: #b8bcc8;
        --border-color: #333;
        --success-color: #00ff88;
        --warning-color: #ffaa00;
        --disabled-color: #555;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, var(--primary-bg) 0%, var(--secondary-bg) 100%);
        color: var(--text-primary);
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Custom title styling */
    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, var(--accent-color), var(--success-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
        text-shadow: 0 0 30px var(--accent-glow);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 20px var(--accent-glow)); }
        to { filter: drop-shadow(0 0 30px var(--accent-color)); }
    }
    
    .subtitle {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.5rem;
        font-weight: 400;
        text-align: center;
        color: var(--text-secondary);
        margin-bottom: 3rem;
    }
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(45deg, var(--accent-color), var(--success-color));
        color: var(--primary-bg);
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-family: 'Orbitron', monospace;
        font-weight: 700;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px var(--accent-glow);
        width: 100%;
        height: 60px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px var(--accent-glow);
    }
    
    .stButton > button:disabled {
        background: var(--disabled-color) !important;
        color: var(--text-secondary) !important;
        box-shadow: none !important;
        transform: none !important;
        cursor: not-allowed !important;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed var(--accent-color);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid var(--accent-color);
        border-radius: 10px;
        color: var(--text-primary);
        backdrop-filter: blur(10px);
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid var(--accent-color);
        border-radius: 10px;
        color: var(--text-primary);
        padding: 1rem;
        font-size: 1.1rem;
        backdrop-filter: blur(10px);
    }
    
    .stTextInput > div > div > input:focus {
        box-shadow: 0 0 15px var(--accent-glow);
        border-color: var(--accent-color);
    }
    
    /* DataFrame styling */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        overflow: hidden;
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-color);
    }
    
    /* Custom containers */
    .upload-container {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid var(--accent-color);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .query-container {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid var(--accent-color);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(15px);
    }
    
    .output-container {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--border-color);
        border-radius: 15px;
        padding: 2rem;
        min-height: 400px;
        backdrop-filter: blur(15px);
    }
    
    /* Status indicators */
    .status-success {
        color: var(--success-color);
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        background: rgba(0, 255, 136, 0.1);
        border: 1px solid var(--success-color);
        display: inline-block;
        margin: 0.5rem 0;
    }
    
    .status-processing {
        color: var(--warning-color);
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        background: rgba(255, 170, 0, 0.1);
        border: 1px solid var(--warning-color);
        display: inline-block;
        margin: 0.5rem 0;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Navigation styling */
    .nav-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .nav-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: var(--border-color);
        transition: all 0.3s ease;
    }
    
    .nav-dot.active {
        background: var(--accent-color);
        box-shadow: 0 0 15px var(--accent-glow);
        transform: scale(1.2);
    }
    
    .section-header {
        font-family: 'Orbitron', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--accent-color);
        margin: 2rem 0 1rem 0;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Result type styling */
    .result-type-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1rem;
    }
    
    .result-statistic {
        background: rgba(255, 170, 0, 0.2);
        color: var(--warning-color);
        border: 1px solid var(--warning-color);
    }
    
    .result-table {
        background: rgba(0, 212, 255, 0.2);
        color: var(--accent-color);
        border: 1px solid var(--accent-color);
    }
    
    .result-graph {
        background: rgba(0, 255, 136, 0.2);
        color: var(--success-color);
        border: 1px solid var(--success-color);
    }
</style>
""",
    unsafe_allow_html=True,
)


# Environment variables validation
def check_environment():
    """Check if required environment variables are set"""
    required_vars = []
    missing_vars = []

    # Add your required environment variables here
    # Example: required_vars = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'OTHER_KEY']

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    return missing_vars


def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location and return path"""
    try:
        # Create temp directory if it doesn't exist
        temp_dir = "temp_uploads"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Save file
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None


def display_agent_result(result):
    """Display the result from agent_at_work function in a clean, organized manner"""
    if not result:
        st.error("No result received from the agent")
        return

    # Display result type badge
    result_type = result.get("result_type", "unknown")
    result_type_class = (
        f"result-{result_type}"
        if result_type in ["statistic", "table", "graph"]
        else "result-statistic"
    )

    st.markdown(
        f'<div class="result-type-badge {result_type_class}">📊 {result_type.upper()} RESULT</div>',
        unsafe_allow_html=True,
    )

    if result["status"] == "success":
        # Display success status
        st.markdown(
            '<div class="status-success">✅ Analysis Complete!</div>',
            unsafe_allow_html=True,
        )

        # Display message
        if "message" in result:
            st.markdown(f"**Response:** {result['message']}")

        # Display the actual data based on result type
        data = result.get("data", {})

        if result_type == "statistic":
            # Display scalar/statistical result
            st.subheader("📈 Statistical Result")
            value = data.get("value", "No value found")

            # Create a nice display for the statistic
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 2rem; background: rgba(255,170,0,0.1); 
                                border-radius: 15px; border: 2px solid var(--warning-color); margin: 1rem 0;">
                        <div style="font-size: 3rem; font-weight: bold; color: var(--warning-color);">
                            {value}
                        </div>
                        <div style="font-size: 1rem; color: var(--text-secondary); margin-top: 0.5rem;">
                            {data.get('description', 'Statistical Result')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        elif result_type == "table":
            # Display DataFrame result
            st.subheader("📊 Data Table")
            dataframe = data.get("dataframe")

            if dataframe is not None and hasattr(dataframe, "to_dict"):
                # Show dataframe info
                shape = data.get("shape", dataframe.shape)
                st.markdown(f"**Shape:** {shape[0]} rows × {shape[1]} columns")

                # Display the dataframe
                st.dataframe(dataframe, use_container_width=True, height=400)

                # Show column information
                with st.expander("📋 Column Details"):
                    col_info = pd.DataFrame(
                        {
                            "Column": dataframe.columns,
                            "Data Type": dataframe.dtypes,
                            "Non-Null Count": dataframe.count(),
                            "Unique Values": dataframe.nunique(),
                        }
                    )
                    st.dataframe(col_info, use_container_width=True)

            else:
                st.error("DataFrame not found or invalid format")

        elif result_type == "graph":
            # Display visualization result
            st.subheader("📈 Visualization")
            figure = data.get("figure")

            if figure is not None:
                figure_type = data.get("type", "unknown")

                if figure_type == "matplotlib":
                    # Handle matplotlib figures
                    st.pyplot(figure, use_container_width=True)

                elif figure_type == "plotly":
                    # Handle plotly figures
                    st.plotly_chart(figure, use_container_width=True)

                else:
                    # Try to display as matplotlib by default
                    try:
                        st.pyplot(figure, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not display figure: {str(e)}")
                        st.write(f"Figure type: {type(figure)}")
            else:
                st.error("No visualization found")

        # Show additional description if available
        if data.get("description"):
            st.info(f"ℹ️ {data['description']}")

    elif result["status"] == "error":
        # Display error status
        st.error(f"❌ Analysis failed: {result.get('message', 'Unknown error')}")

        # Show error details if available
        error_data = result.get("data", {})
        if "error" in error_data:
            with st.expander("🔍 Error Details"):
                st.code(error_data["error"])

    else:
        st.warning(f"⚠️ Unexpected response status: {result['status']}")
        st.json(result)


# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = 1
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "df" not in st.session_state:
    st.session_state.df = None
if "file_path" not in st.session_state:
    st.session_state.file_path = None
if "processing" not in st.session_state:
    st.session_state.processing = False
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "query_text" not in st.session_state:
    st.session_state.query_text = ""


# Navigation dots
def show_navigation():
    nav_html = """
    <div class="nav-container">
        <div class="nav-dot {}"></div>
        <div style="width: 50px; height: 2px; background: {}; margin: 0 1rem;"></div>
        <div class="nav-dot {}"></div>
    </div>
    """.format(
        "active" if st.session_state.page == 1 else "",
        "#00d4ff" if st.session_state.page == 2 else "#333",
        "active" if st.session_state.page == 2 else "",
    )
    st.markdown(nav_html, unsafe_allow_html=True)


# PAGE 1: Landing Page
def show_page_1():
    # Main title with environment status
    missing_vars = check_environment()
    if missing_vars and AGENT_AVAILABLE:
        st.warning(f"⚠️ Missing environment variables: {', '.join(missing_vars)}")
    elif not AGENT_AVAILABLE:
        st.error("🚨 agent_at_work.py not found in current directory")

    st.markdown(
        '<h1 class="main-title">🤖 AnalystBud.AI</h1>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="subtitle">Advanced Data Analysis with Intelligent Models</p>',
        unsafe_allow_html=True,
    )

    show_navigation()

    # Main upload container
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            '<h2 class="section-header">📁 Data Upload</h2>', unsafe_allow_html=True
        )

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose your CSV file",
            type=["csv"],
            help="Upload a CSV file to begin analysis",
            label_visibility="collapsed",
        )

        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            try:
                # Read and preview the file
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df

                # Save file to disk for agent_at_work function
                file_path = save_uploaded_file(uploaded_file)
                if file_path:
                    st.session_state.file_path = file_path
                    st.markdown(
                        '<div class="status-success">✅ File uploaded and saved successfully!</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"**File:** {uploaded_file.name} | **Rows:** {len(df)} | **Columns:** {len(df.columns)}"
                    )
                else:
                    st.error("Failed to save uploaded file")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

        st.markdown(
            '<h2 class="section-header">🧠 Model Selection</h2>', unsafe_allow_html=True
        )

        # Model selection
        models = [
            "qwen/qwen3-coder",
            "meta-llama/llama-3.1-405b-instruct:free",
            "openrouter/horizon-beta",
            "openai/gpt-5-nano",
            "openai/gpt-5-chat",
        ]
        selected_model = st.selectbox(
            "Choose AI Model",
            models,
            index=0,
            help="Select the AI model for data analysis",
            label_visibility="collapsed",
        )
        st.session_state.selected_model = selected_model

        st.markdown("</br>", unsafe_allow_html=True)

        # Next button
        next_disabled = not uploaded_file or not AGENT_AVAILABLE
        if st.button("🚀 LAUNCH ANALYSIS", disabled=next_disabled):
            if not AGENT_AVAILABLE:
                st.error("Cannot proceed: agent_at_work.py not available")
            else:
                st.session_state.page = 2
                st.rerun()

    # Feature highlights
    st.markdown("</br></br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    features = [
        ("📊", "Data Visualization", "Interactive charts and graphs"),
        ("📝", "Smart Analysis", "AI-powered insights"),
        ("⚡", "Real-time Results", "Instant query processing"),
        ("🎯", "Custom Queries", "Natural language questions"),
    ]

    for i, (icon, title, desc) in enumerate(features):
        with [col1, col2, col3, col4][i]:
            st.markdown(
                f"""
            <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.05); 
                        border-radius: 15px; border: 1px solid #333; margin: 0.5rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                <div style="font-weight: bold; color: #00d4ff; margin-bottom: 0.5rem;">{title}</div>
                <div style="font-size: 0.9rem; color: #b8bcc8;">{desc}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )


# PAGE 2: Analysis Page
def show_page_2():
    st.markdown(
        '<h1 class="main-title">📊 DATA ANALYSIS HUB</h1>', unsafe_allow_html=True
    )

    show_navigation()

    if st.session_state.df is not None:
        # Data Preview Section
        st.markdown(
            '<h2 class="section-header">📋 Data Preview</h2>', unsafe_allow_html=True
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(
                f"**Model:** {st.session_state.selected_model} | **File:** {st.session_state.uploaded_file.name}"
            )
        with col2:
            if st.button("← Back to Upload", type="secondary"):
                st.session_state.page = 1
                st.rerun()

        # Show top 10 rows
        preview_df = st.session_state.df.head(10)
        st.dataframe(preview_df, use_container_width=True, height=300)

        # Query Section
        st.markdown(
            '<h2 class="section-header">💬 Ask Your Question</h2>',
            unsafe_allow_html=True,
        )

        # Check if there's text in the input
        query_col1, query_col2, query_col3 = st.columns([3, 1, 1])

        with query_col1:
            # Text input that updates immediately
            user_query = st.text_input(
                "Enter your query",
                placeholder="What insights can you provide about this data?",
                key="query_input",
                label_visibility="collapsed",
                disabled=st.session_state.processing,
            )

        # Check if query has content for button states - use the actual input value
        has_query_text = bool(user_query and user_query.strip())

        with query_col2:
            clear_button = st.button(
                "🗑️ Clear",
                disabled=not has_query_text or st.session_state.processing,
                key="clear_btn",
            )

        with query_col3:
            stop_button = st.button(
                "⏹️ Stop", disabled=not st.session_state.processing, key="stop_btn"
            )

        # Handle button clicks
        if clear_button:
            st.session_state.analysis_result = None
            st.rerun()

        if stop_button and st.session_state.processing:
            st.session_state.processing = False
            st.session_state.analysis_result = {
                "status": "error",
                "message": "Analysis was interrupted by user",
                "data": {"error": "User stopped the operation"},
                "result_type": "unknown",
            }
            st.rerun()

        # Analyze button
        analyze_disabled = (
            not has_query_text
            or not AGENT_AVAILABLE
            or not st.session_state.file_path
            or st.session_state.processing
        )

        # Single analyze button
        analyze_button = st.button(
            "🔍 ANALYZE DATA",
            disabled=analyze_disabled,
            key="analyze_btn",
            use_container_width=True,
        )

        # Error messages for disabled states
        if not AGENT_AVAILABLE:
            st.error("🚨 Cannot analyze: agent_at_work.py not available")
        elif not st.session_state.file_path:
            st.error("🚨 File path not available. Please re-upload the file.")

        # Handle analysis trigger
        if analyze_button and has_query_text and not st.session_state.processing:
            # Set processing state
            st.session_state.processing = True
            st.session_state.analysis_result = None
            # Store the query for processing
            st.session_state.current_query = user_query
            st.rerun()

        # Show processing status if processing
        if st.session_state.processing:
            st.markdown(
                '<div class="status-processing">🔄 Processing your query...</div>',
                unsafe_allow_html=True,
            )

            # Perform the actual analysis
            try:
                result = agent_at_work(
                    model=st.session_state.selected_model,
                    user_query=st.session_state.current_query,
                    path=st.session_state.file_path,
                )

                st.session_state.analysis_result = result
                st.session_state.processing = False
                st.rerun()

            except Exception as e:
                st.session_state.analysis_result = {
                    "status": "error",
                    "message": f"Error during analysis: {str(e)}",
                    "data": {"error": str(e)},
                    "result_type": "unknown",
                }
                st.session_state.processing = False
                st.rerun()

        # Display results if available
        if st.session_state.analysis_result and not st.session_state.processing:
            display_agent_result(st.session_state.analysis_result)
        elif not st.session_state.analysis_result and not st.session_state.processing:
            # Show placeholder when no query or results
            st.markdown(
                """
            <div style="text-align: center; padding: 3rem; color: #b8bcc8;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🤔</div>
                <div style="font-size: 1.2rem;">Enter a query above to start the analysis</div>
                <div style="font-size: 1rem; margin-top: 1rem;">Ask questions like:</div>
                <div style="font-size: 0.9rem; margin-top: 0.5rem; font-style: italic;">
                    "Show me the correlation between variables"<br>
                    "What are the key trends in this data?"<br>
                    "Generate insights about the dataset"<br>
                    "Create a bar plot showing distribution"<br>
                    "What is the average of column X?"
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    else:
        st.error("No data found. Please go back and upload a file.")
        if st.button("← Back to Upload"):
            st.session_state.page = 1
            st.rerun()


# Main app logic
def main():
    if st.session_state.page == 1:
        show_page_1()
    else:
        show_page_2()


if __name__ == "__main__":
    main()
