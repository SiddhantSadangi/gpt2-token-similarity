# GPT-2 Token Similarity Graph

A Streamlit application for visualizing token similarity relationships across GPT-2 layers. This is a Python fork of the original React-based visualizer from [@KhoomeiK/gpt2-explorer](https://github.com/KhoomeiK/gpt2-explorer/tree/main/visualizer) that provides an interactive "microscope" into how language models process and understand text.

## What This App Does

This tool helps you understand how GPT-2 processes text by visualizing the relationships between tokens across different layers. It's particularly valuable for:

- **🧠 Understanding Language Model Internals**: See how semantic relationships between tokens evolve as they're processed through different GPT-2 layers
- **🔬 Research & Interpretability**: Debug model behavior, analyze attention patterns, and study how different prompts affect internal representations
- **🎓 Educational Use**: Demonstrate how transformers work visually and teach AI concepts through interactive exploration
- **🛠 Practical Applications**: Optimize prompts, debug model behavior, and identify unexpected token associations

## Features

- **Interactive Graphs**: Pan, zoom, and hover for activation details
- **Layer Exploration**: See how token relationships evolve across layers
- **Adjustable Parameters**: Threshold, neighbor limits, node size, and edge width
- **Multiple Metrics**: Cosine similarity and dot product
- **Color Schemes**: Position-based or activation-based
- **Real-time Statistics**: Graph density and similarity stats
- **Token Search**: Find and highlight specific tokens in the graph
- **Auto-play**: Animate through layers automatically
