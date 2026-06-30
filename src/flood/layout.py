"""
src/flood/layout.py
===================
Centralized helper to retrieve model structural configuration mappings
(layers, heads, hidden_size) from model names or head counts.
"""

def get_model_layout(model_name_or_heads):
    """
    Returns a dictionary with 'layers', 'heads', and 'hidden_size' keys
    based on the model name or total head count.
    """
    if isinstance(model_name_or_heads, int):
        n_total = model_name_or_heads
    else:
        name = str(model_name_or_heads).lower()
        if "medium" in name:
            n_total = 384
        elif "pythia-70m" in name:
            n_total = 48
        elif "pythia-160m" in name:
            n_total = 144
        else:
            # Default to GPT-2 Small
            n_total = 144
            
    if n_total == 144:
        return {"layers": 12, "heads": 12, "hidden_size": 768}
    elif n_total == 288:
        return {"layers": 24, "heads": 12, "hidden_size": 768}
    elif n_total == 384:
        return {"layers": 24, "heads": 16, "hidden_size": 1024}
    elif n_total == 48:
        return {"layers": 6, "heads": 8, "hidden_size": 512}
    else:
        # Fallback logic
        heads = 12
        layers = n_total // heads
        return {"layers": layers, "heads": heads, "hidden_size": 768}
