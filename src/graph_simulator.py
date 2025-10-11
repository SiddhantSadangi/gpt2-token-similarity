"""
Graph Simulation Module
Handles graph layout and simulation using NetworkX
"""

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

import streamlit as st


class GraphSimulator:
    def __init__(self):
        self.graph = nx.Graph()
        self.positions = {}
        self.previous_positions = {}
        self.animation_state = {}

    def create_graph(
        self,
        tokens: List[str],
        edges: List[Tuple],
        colors: List[str],
        morph_from_previous: bool = True,
    ) -> Dict:
        """Create graph with smooth morphing capability"""
        self.graph.clear()

        # Add nodes
        for i, (token, color) in enumerate(zip(tokens, colors)):
            self.graph.add_node(i, token=token, color=color)

        # Add edges
        for source, target, weight in edges:
            self.graph.add_edge(source, target, weight=weight)

        # Compute layout with morphing
        if morph_from_previous and self.positions:
            self.positions = self._morph_layout(edges)
        else:
            self.positions = nx.spring_layout(
                self.graph,
                k=1,
                iterations=100,
                seed=42,  # For reproducible layouts
            )

        return self._get_graph_state()

    def _morph_layout(self, new_edges: List[Tuple]) -> Dict:
        """Smooth morphing between layouts"""
        # Create temporary graph for new layout
        temp_graph = nx.Graph()

        # Add nodes
        for node_id in self.graph.nodes():
            temp_graph.add_node(node_id, **self.graph.nodes[node_id])

        # Add new edges
        for source, target, weight in new_edges:
            temp_graph.add_edge(source, target, weight=weight)

        # Compute new layout
        new_positions = nx.spring_layout(
            temp_graph,
            k=1,
            iterations=50,
            pos=self.positions,  # Start from current positions
            seed=42,
        )

        # Smooth interpolation between old and new positions
        alpha = 0.3  # Interpolation factor
        for node_id in new_positions:
            if node_id in self.positions:
                old_pos = self.positions[node_id]
                new_pos = new_positions[node_id]
                self.positions[node_id] = (
                    old_pos[0] * (1 - alpha) + new_pos[0] * alpha,
                    old_pos[1] * (1 - alpha) + new_pos[1] * alpha,
                )
            else:
                self.positions[node_id] = new_positions[node_id]

        return self.positions

    def _get_graph_state(self) -> Dict:
        """Convert to format suitable for visualization"""
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            x, y = self.positions[node_id]
            nodes.append(
                {"id": node_id, "token": data["token"], "x": x, "y": y, "color": data["color"]}
            )

        edges = []
        for source, target, data in self.graph.edges(data=True):
            edges.append({"source": source, "target": target, "weight": data["weight"]})

        return {"nodes": nodes, "edges": edges}

    def update_graph(self, new_edges: List[Tuple], morph: bool = True) -> Dict:
        """Update graph with new edges while preserving node positions"""
        # Store current edges for morphing
        current_edges = [(e[0], e[1], e[2]) for e in self.graph.edges(data="weight")]

        # Update edges
        self.graph.clear_edges()
        for source, target, weight in new_edges:
            self.graph.add_edge(source, target, weight=weight)

        # Morph layout if requested
        if morph:
            self.positions = self._morph_layout(new_edges)

        return self._get_graph_state()

    def get_graph_stats(self) -> Dict:
        """Get statistics about the current graph"""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "is_connected": nx.is_connected(self.graph),
            "num_components": nx.number_connected_components(self.graph),
        }

    def reset_layout(self):
        """Reset to random layout"""
        self.positions = {}
        self.previous_positions = {}
