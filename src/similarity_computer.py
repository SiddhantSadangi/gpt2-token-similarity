"""
Similarity Computation Module
Handles token similarity calculations and edge extraction
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st


class SimilarityComputer:
    def __init__(self):
        self.cache = {}

    @st.cache_data
    def compute_similarities_cached(_self, activations_hash: str, metric: str) -> np.ndarray:
        """Cache similarity computations for performance"""
        # This will be called with a hash of the activations
        # The actual computation happens in the non-cached method
        pass

    def compute_similarities(self, activations: np.ndarray, metric: str = "cosine") -> np.ndarray:
        """Compute pairwise similarities between tokens"""
        T, D = activations.shape

        if metric == "cosine":
            # Use optimized cosine similarity
            similarities = cosine_similarity(activations)
        else:  # dot product
            similarities = np.dot(activations, activations.T)

        return similarities

    def threshold_edges(
        self, similarities: np.ndarray, threshold: float, k_cap: int
    ) -> List[Tuple]:
        """Extract edges above threshold, capped at k per node"""
        T = similarities.shape[0]
        edges = []

        for i in range(T):
            # Get similarities for token i
            sims = similarities[i]
            # Find indices above threshold (excluding self)
            candidates = np.where((sims >= threshold) & (np.arange(T) != i))[0]

            if len(candidates) > 0:
                # Sort by similarity and take top k
                sim_values = sims[candidates]
                top_indices = np.argsort(sim_values)[::-1][:k_cap]

                for idx in top_indices:
                    j = candidates[idx]
                    if i < j:  # Avoid duplicate edges
                        edges.append((i, j, sim_values[idx]))

        return edges

    def normalize_rows(self, data: np.ndarray, T: int, D: int) -> np.ndarray:
        """Normalize rows for cosine similarity (from original worker)"""
        out = np.zeros_like(data)
        for i in range(T):
            norm = np.linalg.norm(data[i])
            if norm > 0:
                out[i] = data[i] / norm
            else:
                out[i] = data[i]
        return out

    def compute_similarities_optimized(
        self, activations: np.ndarray, metric: str = "cosine"
    ) -> np.ndarray:
        """Optimized similarity computation matching original implementation"""
        T, D = activations.shape

        if metric == "cosine":
            # Normalize rows first
            normalized = self.normalize_rows(activations, T, D)
            similarities = np.dot(normalized, normalized.T)
        else:  # dot product
            similarities = np.dot(activations, activations.T)

        # Set diagonal to 1.0 for cosine similarity
        if metric == "cosine":
            np.fill_diagonal(similarities, 1.0)

        return similarities

    def get_similarity_stats(self, similarities: np.ndarray) -> Dict:
        """Get statistics about the similarity matrix"""
        # Remove diagonal for stats
        mask = ~np.eye(similarities.shape[0], dtype=bool)
        off_diagonal = similarities[mask]

        return {
            "mean": float(np.mean(off_diagonal)),
            "std": float(np.std(off_diagonal)),
            "min": float(np.min(off_diagonal)),
            "max": float(np.max(off_diagonal)),
            "median": float(np.median(off_diagonal)),
        }
