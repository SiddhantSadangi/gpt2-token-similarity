"""
Utility Functions
Helper functions for colors, data processing, and common operations
"""

import hashlib
from typing import List, Tuple

import numpy as np


def seq_index_to_color(i: int, T: int) -> str:
    """Convert sequence index to HSL color (matching original implementation)"""
    hue_start = 210  # blue
    hue_end = 330  # magenta
    hue = hue_start + (hue_end - hue_start) * (i / max(1, T - 1))
    return f"hsl({hue}, 80%, 55%)"


def create_color_gradient(T: int) -> List[str]:
    """Create color gradient for T tokens"""
    return [seq_index_to_color(i, T) for i in range(T)]


def create_activation_based_colors(activations: np.ndarray, metric: str = "cosine") -> List[str]:
    """Create colors based on activation magnitudes for better visual differentiation"""
    if metric == "cosine":
        # For cosine similarity, use the magnitude of the activation vectors
        magnitudes = np.linalg.norm(activations, axis=1)
    else:
        # For dot product, use the sum of activations
        magnitudes = np.sum(activations, axis=1)

    # Normalize magnitudes to [0, 1] range
    min_mag = np.min(magnitudes)
    max_mag = np.max(magnitudes)
    if max_mag > min_mag:
        normalized = (magnitudes - min_mag) / (max_mag - min_mag)
    else:
        normalized = np.ones_like(magnitudes) * 0.5

    # Create colors with varying saturation and lightness based on magnitude
    colors = []
    for mag in normalized:
        # Use different hue ranges for different magnitude levels
        if mag < 0.33:
            hue = 240  # Blue for low magnitude
        elif mag < 0.66:
            hue = 120  # Green for medium magnitude
        else:
            hue = 0  # Red for high magnitude

        # Vary saturation and lightness based on magnitude
        saturation = 60 + int(mag * 40)  # 60-100%
        lightness = 40 + int(mag * 30)  # 40-70%

        colors.append(f"hsl({hue}, {saturation}%, {lightness}%)")

    return colors


def hash_activations(activations: np.ndarray) -> str:
    """Create hash of activations for caching"""
    return hashlib.md5(activations.tobytes()).hexdigest()


def normalize_activations(activations: np.ndarray) -> np.ndarray:
    """Normalize activations to unit vectors"""
    norms = np.linalg.norm(activations, axis=1, keepdims=True)
    return activations / (norms + 1e-8)


def compute_sequence_stats(tokens: List[str]) -> dict:
    """Compute basic statistics about token sequence"""
    return {
        "length": len(tokens),
        "unique_tokens": len(set(tokens)),
        "avg_token_length": np.mean([len(token) for token in tokens]),
        "total_chars": sum(len(token) for token in tokens),
    }


def format_number(num: float, decimals: int = 2) -> str:
    """Format number for display"""
    if abs(num) < 0.001:
        return f"{num:.2e}"
    elif abs(num) < 1:
        return f"{num:.3f}"
    else:
        return f"{num:.{decimals}f}"


def truncate_text(text: str, max_length: int = 50) -> str:
    """Truncate text with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def create_token_display(tokens: List[str], max_display: int = 20) -> str:
    """Create formatted token display string"""
    if len(tokens) <= max_display:
        return " ".join(tokens)
    else:
        visible_tokens = tokens[:max_display]
        return " ".join(visible_tokens) + f" ... (+{len(tokens) - max_display} more)"


def validate_prompt(prompt: str, max_length: int = 512) -> Tuple[bool, str]:
    """Validate prompt input"""
    if not prompt.strip():
        return False, "Prompt cannot be empty"

    if len(prompt) > max_length:
        return False, f"Prompt too long (max {max_length} characters)"

    return True, ""


def create_progress_bar(current: int, total: int, width: int = 20) -> str:
    """Create text progress bar"""
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {current}/{total}"


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division with default value"""
    return a / b if b != 0 else default
