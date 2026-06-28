"""
src/flood/prune.py
==================
Performs virtual attention head pruning by zero-masking attention head output projection weights.
"""

from evaluate import mask_heads_in_model

def prune_attention_heads(model, heads_to_prune_list):
    """
    Takes a Hugging Face model and a list of (layer_idx, head_idx) tuples,
    makes a copy, zero-masks the specified heads' c_proj weights to virtually
    prune them, and returns the pruned model.
    """
    return mask_heads_in_model(model, heads_to_prune_list)
