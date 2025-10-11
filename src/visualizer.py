"""
Visualization Module
Handles Plotly-based graph visualization
"""

from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class GraphVisualizer:
    def __init__(self):
        self.fig = None

    def create_interactive_graph(
        self,
        graph_state: Dict,
        show_labels: bool = True,
        animation_frame: Optional[int] = None,
        metric: str = "cosine",
        search_token: Optional[str] = None,
        node_size: int = 35,
        edge_width_multiplier: float = 3.0,
        activations: Optional[np.ndarray] = None,
    ) -> go.Figure:
        """Create rich interactive Plotly graph"""
        nodes = graph_state["nodes"]
        edges = graph_state["edges"]

        if not nodes:
            # Return empty figure if no nodes
            return go.Figure().add_annotation(
                text="No nodes to display",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        # Handle search token highlighting and prepare activation data
        node_colors = []
        node_sizes = []
        node_opacities = []
        activation_magnitudes = []

        for i, node in enumerate(nodes):
            # Check if this node matches the search token
            if search_token and search_token.lower() in node["token"].lower():
                # Highlight matching tokens - make them larger than base size
                node_colors.append("red")
                node_sizes.append(int(node_size * 1.3))  # 30% larger than base size
                node_opacities.append(1.0)
            else:
                # Use original styling with custom node size
                node_colors.append(node["color"])
                node_sizes.append(node_size)
                node_opacities.append(0.8)

            # Calculate activation magnitude for this node
            if activations is not None and i < len(activations):
                if metric == "cosine":
                    activation_mag = np.linalg.norm(activations[i])
                else:  # dot product
                    activation_mag = np.sum(activations[i])
                activation_magnitudes.append(f"{activation_mag:.3f}")
            else:
                activation_magnitudes.append("N/A")

        # Create node trace with enhanced styling and search highlighting
        node_trace = go.Scatter(
            x=[node["x"] for node in nodes],
            y=[node["y"] for node in nodes],
            mode="markers+text" if show_labels else "markers",
            text=[node["token"] for node in nodes] if show_labels else [],
            textposition="middle center",
            textfont=dict(size=12, color="white"),
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color="white"),
                opacity=node_opacities,
            ),
            hovertemplate="<b>%{text}</b><br>Activation: %{customdata}<extra></extra>",
            customdata=activation_magnitudes,
            name="Tokens",
        )

        # Normalize edge weights for visualization
        if edges:
            weights = [edge["weight"] for edge in edges]
            min_weight = min(weights)
            max_weight = max(weights)
            weight_range = max_weight - min_weight

            # Avoid division by zero
            if weight_range == 0:
                weight_range = 1

        # Create enhanced edge traces
        edge_traces = []
        for edge in edges:
            source_node = nodes[edge["source"]]
            target_node = nodes[edge["target"]]

            # Normalize weight for visualization with enhanced differentiation
            if metric == "cosine":
                # Cosine similarity: use more aggressive scaling for better opacity differentiation
                # Map [-1, 1] to [0, 1] with exponential scaling
                raw_normalized = (edge["weight"] + 1) / 2  # Map [-1,1] to [0,1]
                # Apply more aggressive exponential scaling to amplify differences
                normalized_weight = raw_normalized**2  # Square for more differentiation
            else:
                # Dot product: normalize to [0.1, 1.0] range with exponential scaling
                raw_normalized = (edge["weight"] - min_weight) / weight_range
                normalized_weight = raw_normalized**0.5  # Square root for more differentiation
            normalized_weight = max(
                0.05, min(1.0, normalized_weight)
            )  # Lower minimum for more range
            edge_trace = go.Scatter(
                x=[source_node["x"], target_node["x"], None],
                y=[source_node["y"], target_node["y"], None],
                mode="lines",
                line=dict(
                    width=max(
                        0.8, normalized_weight * edge_width_multiplier
                    ),  # Use custom edge width multiplier
                    color=f"rgba(0,0,0,{normalized_weight})",  # Use opacity to show weight differences
                ),
                hoverinfo="none",
                showlegend=False,
            )
            edge_traces.append(edge_trace)

        # Create figure with enhanced layout
        fig = go.Figure(data=edge_traces + [node_trace])

        # Enhanced layout
        fig.update_layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=60),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

        return fig

    def create_similarity_heatmap(self, similarities: np.ndarray, tokens: List[str]) -> go.Figure:
        """Create similarity matrix heatmap"""
        fig = go.Figure(
            data=go.Heatmap(
                z=similarities, x=tokens, y=tokens, colorscale="RdBu", zmid=0, hoverongaps=False
            )
        )

        fig.update_layout(
            title="Token Similarity Matrix",
            xaxis_title="Tokens",
            yaxis_title="Tokens",
            width=600,
            height=600,
        )

        return fig

    def create_layer_comparison(
        self, graph_states: List[Dict], layer_names: List[str]
    ) -> go.Figure:
        """Create side-by-side comparison of different layers"""
        n_layers = len(graph_states)

        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=n_layers,
            subplot_titles=layer_names,
            specs=[[{"type": "scatter"} for _ in range(n_layers)]],
        )

        for i, (graph_state, layer_name) in enumerate(zip(graph_states, layer_names)):
            nodes = graph_state["nodes"]
            edges = graph_state["edges"]

            # Add nodes
            fig.add_trace(
                go.Scatter(
                    x=[node["x"] for node in nodes],
                    y=[node["y"] for node in nodes],
                    mode="markers+text",
                    text=[node["token"] for node in nodes],
                    textposition="middle center",
                    textfont=dict(size=10),
                    marker=dict(
                        size=20,
                        color=[node["color"] for node in nodes],
                        line=dict(width=1, color="white"),
                    ),
                    showlegend=False,
                ),
                row=1,
                col=i + 1,
            )

            # Add edges
            for edge in edges:
                source_node = nodes[edge["source"]]
                target_node = nodes[edge["target"]]

                fig.add_trace(
                    go.Scatter(
                        x=[source_node["x"], target_node["x"], None],
                        y=[source_node["y"], target_node["y"], None],
                        mode="lines",
                        line=dict(width=edge["weight"] * 3, color="rgba(0,0,0,0.3)"),
                        hoverinfo="none",
                        showlegend=False,
                    ),
                    row=1,
                    col=i + 1,
                )

        fig.update_layout(title="Layer Comparison", showlegend=False, height=400)

        # Update axes
        for i in range(n_layers):
            fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=i + 1)
            fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=i + 1)

        return fig

    def create_statistics_plot(self, stats_data: Dict) -> go.Figure:
        """Create statistics visualization"""
        fig = go.Figure()

        # Add bar chart for edge counts by layer
        if "edge_counts_by_layer" in stats_data:
            fig.add_trace(
                go.Bar(
                    x=list(stats_data["edge_counts_by_layer"].keys()),
                    y=list(stats_data["edge_counts_by_layer"].values()),
                    name="Edge Count",
                )
            )

        fig.update_layout(title="Graph Statistics", xaxis_title="Layer", yaxis_title="Count")

        return fig

