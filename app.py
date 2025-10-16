"""
GPT-2 Token Similarity Graph - Streamlit App
Main application file
"""

import time
from typing import Dict

import numpy as np
import streamlit as st

# Import our modules
from src.gpt2_integration import GPT2StreamlitAdapter
from src.graph_simulator import GraphSimulator
from src.similarity_computer import SimilarityComputer
from src.utils import (
    compute_sequence_stats,
    create_activation_based_colors,
    create_color_gradient,
    create_token_display,
    format_number,
    validate_prompt,
)
from src.visualizer import GraphVisualizer

__version__ = "0.1.0"

# Page configuration
st.set_page_config(
    page_title="GPT-2 Token Similarity Graph",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
   
    /* Ensure main content adjusts */
    .main .block-container {
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B6B;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


def initialize_session_state():
    """Initialize session state variables"""
    if "model_run" not in st.session_state:
        st.session_state.model_run = None
    if "current_layer" not in st.session_state:
        st.session_state.current_layer = 0
    if "playing" not in st.session_state:
        st.session_state.playing = False
    if "similarity_cache" not in st.session_state:
        st.session_state.similarity_cache = {}
    if "graph_cache" not in st.session_state:
        st.session_state.graph_cache = {}
    if "search_token" not in st.session_state:
        st.session_state.search_token = ""
    if "color_scheme" not in st.session_state:
        st.session_state.color_scheme = "Activation-based"


def create_sidebar_controls():
    """Create sidebar controls"""
    with st.sidebar:
        st.markdown(f"**Version:** {__version__}")
        # Initialize prompt in session state if not exists
        if "prompt" not in st.session_state:
            st.session_state.prompt = "The quick brown fox jumps over the lazy dog"

        # Prompt input
        prompt = st.text_area(
            "Prompt",
            value=st.session_state.prompt,
            height=100,
            help="Enter a text prompt to analyze",
            key="prompt_input",
        )

        # Model run button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Run Model", type="primary", width="stretch"):
                run_model(prompt)
                return {"prompt": prompt}
        with col2:
            if st.button("🔄 Reset", width="stretch"):
                reset_session()
                return {"prompt": prompt}

        # Parameters (only show if model has been run)
        if st.session_state.model_run:
            return _additional_sidebar_controls(prompt)
        return {"prompt": prompt}


def _additional_sidebar_controls(prompt):
    st.header("⚙️ Parameters")

    # Similarity metric
    metric = st.segmented_control(
        "Similarity Metric",
        ["cosine", "dot"],
        default="cosine",
        help="Method for computing token similarities",
    )

    # Threshold and k-cap
    col1, col2 = st.columns(2)
    with col1:
        threshold = st.slider(
            "Threshold τ",
            min_value=-1.0,
            max_value=1.0,
            value=0.10,
            step=0.01,
            help="Minimum similarity for edge creation",
        )
    with col2:
        k_cap = st.slider(
            "Max neighbors",
            min_value=2,
            max_value=len(prompt.split(" ")),
            value=8,
            step=1,
            help="Maximum edges per node",
        )

    # Layer controls
    st.header("🎚️ Layer Controls")
    max_layer = st.session_state.model_run["L"]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("⏮️ First"):
            st.session_state.current_layer = 0
            st.rerun()
    with col2:
        if st.button("⬅️ Prev"):
            if st.session_state.current_layer > 0:
                st.session_state.current_layer -= 1
            st.rerun()
    with col3:
        play_text = "⏸️ Pause" if st.session_state.playing else "▶️ Play"
        if st.button(play_text):
            st.session_state.playing = not st.session_state.playing
            st.rerun()
    with col4:
        if st.button("➡️ Next"):
            if st.session_state.current_layer < max_layer:
                st.session_state.current_layer += 1
            st.rerun()
    with col5:
        if st.button("⏭️ Last"):
            st.session_state.current_layer = max_layer
            st.rerun()

    # Layer slider
    layer = st.slider(
        "Layer",
        min_value=0,
        max_value=max_layer,
        value=st.session_state.current_layer,
        step=1,
        help=f"Select layer (0 to {max_layer})",
    )

    # Graph controls
    st.header("🔧 Graph Controls")
    col1, col2 = st.columns(2)
    node_size = col1.slider("Node size", min_value=10, max_value=50, value=35, step=5)
    edge_width = col2.slider(
        "Edge width multiplier", min_value=0.5, max_value=5.0, value=3.0, step=0.5
    )

    # Color scheme options
    color_scheme = st.segmented_control(
        "Node coloring",
        ["Activation-based", "Position-based"],
        default="Activation-based",
        help="""
Activation-based: Colors by activation magnitude (blue=low, red=high)  
Position-based: Colors by token order (blue→magenta)
""",
    )

    # Update session state when color scheme changes
    if color_scheme != st.session_state.color_scheme:
        st.session_state.color_scheme = color_scheme

    # Search and filter
    st.header("🔍 Search & Filter")
    search_token = st.text_input(
        "Search token",
        value=st.session_state.search_token,
        placeholder="Enter token to highlight...",
        help="Search for specific tokens in the graph",
        key="search_input",
    )

    # Update session state when search token changes
    if search_token != st.session_state.search_token:
        st.session_state.search_token = search_token

    # Show search results if there's a search token and model has been run
    if st.session_state.search_token and st.session_state.model_run:
        tokens = st.session_state.model_run["tokens"]
        matching_tokens = [
            token for token in tokens if st.session_state.search_token.lower() in token.lower()
        ]
        if matching_tokens:
            st.success(
                f"🔍 Found {len(matching_tokens)} matching token(s): {', '.join(matching_tokens)}"
            )
        else:
            st.warning(f"🔍 No tokens found matching '{st.session_state.search_token}'")

    return {
        "prompt": prompt,
        "metric": metric,
        "threshold": threshold,
        "k_cap": k_cap,
        "layer": layer,
        "node_size": node_size,
        "edge_width": edge_width,
        "search_token": st.session_state.search_token,
        "color_scheme": st.session_state.color_scheme,
    }


def run_model(prompt: str):
    """Run GPT-2 model on prompt"""
    # Validate prompt
    is_valid, error_msg = validate_prompt(prompt)
    if not is_valid:
        st.error(f"❌ {error_msg}")
        return

    # Create progress container
    progress_container = st.container()
    with progress_container:
        st.info("🔄 Initializing GPT-2 model...")
        progress_bar = st.progress(0)
        status_text = st.empty()

    try:
        # Step 1: Load model
        status_text.text("📥 Loading GPT-2 model...")
        progress_bar.progress(20)

        adapter = GPT2StreamlitAdapter()
        model, tokenizer = adapter.load_model()

        # Step 2: Tokenize input
        status_text.text("🔤 Tokenizing input...")
        progress_bar.progress(40)

        # Step 3: Run forward pass
        status_text.text("🧠 Processing through GPT-2 layers...")
        progress_bar.progress(60)

        model_run = adapter.run_prompt(prompt)

        # Step 4: Store results
        status_text.text("💾 Storing results...")
        progress_bar.progress(80)

        st.session_state.model_run = model_run
        st.session_state.current_layer = 0
        st.session_state.playing = False

        # Step 5: Complete
        progress_bar.progress(100)
        status_text.text("✅ Model run complete!")

        # Clear progress indicators after a short delay
        time.sleep(1)
        progress_container.empty()
        st.rerun()
        st.success("🎉 Visualization ready! Explore the graph below.")

    except Exception as e:
        progress_container.empty()
        st.error(f"❌ Failed to run model: {str(e)}")
        st.error("💡 Try a shorter prompt or check your internet connection.")


def _run_prompt(prompt):
    # Load model
    adapter = GPT2StreamlitAdapter()
    model_run = adapter.run_prompt(prompt)

    # Store in session state
    st.session_state.model_run = model_run
    st.session_state.current_layer = 0
    st.session_state.playing = False

    st.success("✅ Model run complete!")
    st.rerun()


def reset_session():
    """Reset session state"""
    for key in [
        "model_run",
        "current_layer",
        "playing",
        "similarity_cache",
        "graph_cache",
        "search_token",
        "color_scheme",
    ]:
        if key in st.session_state:
            del st.session_state[key]
    # Reset prompt to default
    st.session_state.prompt = "The quick brown fox jumps over the lazy dog"
    st.rerun()


def create_main_visualization(controls: Dict):
    """Create main visualization area"""
    if not st.session_state.model_run:
        st.info("👆 Enter a prompt and click 'Run Model' to start visualization")
        return

    # Get current layer data
    model_run = st.session_state.model_run
    layer_data = model_run["layers"][controls["layer"]]
    activations = layer_data["data"].reshape(layer_data["T"], layer_data["D"])

    # Compute similarities
    computer = SimilarityComputer()
    similarities = computer.compute_similarities_optimized(activations, controls["metric"])

    # Extract edges
    edges = computer.threshold_edges(similarities, controls["threshold"], controls["k_cap"])

    # Create graph
    simulator = GraphSimulator()
    tokens = model_run["tokens"]

    # Choose color scheme based on user selection
    if controls.get("color_scheme") == "Activation-based":
        colors = create_activation_based_colors(activations, controls["metric"])
    else:
        colors = create_color_gradient(len(tokens))

    graph_state = simulator.create_graph(tokens, edges, colors)

    # Visualize
    visualizer = GraphVisualizer()
    fig = visualizer.create_interactive_graph(
        graph_state,
        show_labels=True,  # Always show labels
        metric=controls["metric"],
        search_token=controls.get("search_token"),
        node_size=controls.get("node_size", 35),
        edge_width_multiplier=controls.get("edge_width", 3.0),
        activations=activations,
    )

    # Display graph
    st.plotly_chart(fig, width="stretch")

    # Statistics (always shown)
    create_statistics_section(model_run, graph_state, similarities, computer)


def create_statistics_section(
    model_run: Dict, graph_state: Dict, similarities: np.ndarray, computer: SimilarityComputer
):
    """Create statistics section"""
    st.header("📊 Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Nodes", len(graph_state["nodes"]))

    with col2:
        st.metric("Edges", len(graph_state["edges"]))

    with col3:
        st.metric("Layer", f"{st.session_state.current_layer}/{model_run['L']}")

    with col4:
        density = len(graph_state["edges"]) / (
            len(graph_state["nodes"]) * (len(graph_state["nodes"]) - 1) / 2
        )
        st.metric("Density", format_number(density))

    # Similarity statistics
    sim_stats = computer.get_similarity_stats(similarities)

    with st.expander("📈 Similarity Statistics"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean", format_number(sim_stats["mean"]))
        with col2:
            st.metric("Std Dev", format_number(sim_stats["std"]))
        with col3:
            st.metric(
                "Range", f"{format_number(sim_stats['min'])} to {format_number(sim_stats['max'])}"
            )


def create_token_info_section():
    """Create token information section"""
    if not st.session_state.model_run:
        return

    model_run = st.session_state.model_run
    tokens = model_run["tokens"]

    st.header("🔤 Token Information")

    # Token sequence
    st.subheader("Token Sequence")
    token_display = create_token_display(tokens)
    st.code(token_display, language=None)

    # Token statistics
    stats = compute_sequence_stats(tokens)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sequence Length", stats["length"])
    with col2:
        st.metric("Unique Tokens", stats["unique_tokens"])
    with col3:
        st.metric("Avg Token Length", format_number(stats["avg_token_length"]))


def main():
    """Main application function"""
    initialize_session_state()

    # Header
    st.markdown(
        '<div class="main-header">🧠 GPT-2 Token Similarity Graph</div>',
        unsafe_allow_html=True,
    )
    st.markdown("Visualize token similarity relationships across GPT-2 layers")

    # Sidebar controls
    controls = create_sidebar_controls()

    # Main content
    if st.session_state.model_run and controls is not None:
        # Main visualization
        create_main_visualization(controls)

        # Token information
        create_token_info_section()

        # Auto-play functionality
        if st.session_state.playing:
            if st.session_state.current_layer < st.session_state.model_run["L"]:
                st.session_state.current_layer += 1
            else:
                st.session_state.current_layer = 0
            time.sleep(1.0)  # Small delay for animation
            st.rerun()
    else:
        # Welcome message
        st.markdown(
            """
        <div class="info-box">
        <h3>🎯 Welcome to the GPT-2 Token Similarity Graph!</h3>
        <p>This tool helps you understand how GPT-2 processes text by visualizing the relationships between tokens across different layers.</p>
        
        <h4>🚀 How to use:</h4>
        <ol>
            <li>Enter a text prompt in the sidebar</li>
            <li>Click "Run Model" to process it through GPT-2</li>
            <li>Explore the similarity graph across different layers</li>
            <li>Adjust parameters to see how relationships change</li>
        </ol>
        
        <h4>🔧 Features:</h4>
        <ul>
            <li><strong>Interactive Graphs:</strong> Pan, zoom, and hover for details</li>
            <li><strong>Layer Exploration:</strong> See how token relationships evolve</li>
            <li><strong>Adjustable Parameters:</strong> Threshold and neighbor limits</li>
            <li><strong>Multiple Metrics:</strong> Cosine similarity and dot product</li>
            <li><strong>Export Options:</strong> Download graphs as PNG/SVG</li>
            <li><strong>Search & Filter:</strong> Find specific tokens in the graph</li>
            <li><strong>Color Schemes:</strong> Position-based or activation-based coloring</li>
        </ul>
        
        </div>
        """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
