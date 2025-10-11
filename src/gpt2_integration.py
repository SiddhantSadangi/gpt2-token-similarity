"""
GPT-2 Integration for Streamlit App
Handles model loading and activation extraction
"""

import json
import os
from typing import Dict

import streamlit as st
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer


class GPT2StreamlitAdapter:
    def __init__(self, weights_path: str = "weights/"):
        self.weights_path = weights_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = None

    @st.cache_resource
    def load_model(_self):
        """Load GPT-2 model with local weights or Hugging Face"""
        if _self.model is None:
            try:
                # Try to load from local weights first
                if os.path.exists(f"{_self.weights_path}/manifest.json"):
                    _self._load_local_model()
                else:
                    # Fallback to Hugging Face
                    _self._load_huggingface_model()
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                # Fallback to Hugging Face
                _self._load_huggingface_model()

        return _self.model, _self.tokenizer

    def _load_local_model(self):
        """Load model from local weight files"""
        # Load manifest
        with open(f"{self.weights_path}/manifest.json", "r") as f:
            manifest = json.load(f)

        # Create model from config
        self.config = GPT2Config.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel(self.config)

        # Load weights from local files
        self._load_weights(manifest)

        self._evaluate_model()

    def _load_huggingface_model(self):
        """Load model from Hugging Face Hub"""
        self.model = GPT2LMHeadModel.from_pretrained(
            "gpt2", dtype=torch.float32, low_cpu_mem_usage=True
        )
        self._evaluate_model()

    def _evaluate_model(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self, manifest: Dict):
        """Load weights from local .bin files"""
        # This would implement the weight loading logic from gpt2_webgl.ts
        # For now, we'll use Hugging Face weights as fallback
        pass

    def run_prompt(self, prompt: str, max_length: int = 128) -> Dict:
        """Run prompt through GPT-2 and capture layer activations"""
        model, tokenizer = self.load_model()

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True)
        input_ids = inputs["input_ids"].to(self.device)

        # Get token strings
        token_strings = [tokenizer.decode([token_id]) for token_id in input_ids[0]]

        # Run forward pass with hidden states
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of hidden states for each layer

        # Extract activations for each layer
        layers = []
        T = len(token_strings)

        for layer_idx, hidden_state in enumerate(hidden_states):
            # hidden_state shape: [batch_size, seq_len, hidden_size]
            activations = hidden_state[0].cpu().numpy()  # Remove batch dimension
            D = activations.shape[1]

            layers.append(
                {
                    "tokens": token_strings,
                    "T": T,
                    "D": D,
                    "layerIndex": layer_idx,
                    "data": activations.flatten(),  # Flatten to match original format
                }
            )

        L = len(layers) - 1  # Last layer index

        return {"layers": layers, "L": L, "T": T, "D": D, "tokens": token_strings}

    def get_model_info(self) -> Dict:
        """Get model configuration information"""
        if self.config is None:
            self.config = GPT2Config.from_pretrained("gpt2")

        return {
            "n_layer": self.config.n_layer,
            "n_embd": self.config.n_embd,
            "n_head": self.config.n_head,
            "vocab_size": self.config.vocab_size,
        }
