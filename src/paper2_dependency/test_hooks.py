"""
Unit test for hook execution ordering in map_dependencies.py.
Verifies that ablation hooks successfully modify representations
before capture hooks execute, and that representations propagate downstream.
"""

import os
import torch
import unittest
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from map_dependencies import (
    _capture_head_outputs,
    _capture_head_outputs_ablated,
    load_model
)

class TestHookOrdering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Locate the local model or download a tiny dummy model
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_path = os.path.join(script_dir, "..", "..", "gpt2_local")
        if not os.path.exists(local_path):
            local_path = os.path.join(script_dir, "..", "gpt2_local")
            
        cls.model_path = local_path if os.path.exists(local_path) else "gpt2"
        cls.model, cls.tokenizer = load_model(cls.model_path)
        cls.text = "Hello world, testing hooks."

    def test_ablation_occurs_before_capture(self):
        """
        Verify that capturing an ablated head returns exactly 0.0,
        proving that the ablation pre-hook executed before the capture pre-hook.
        """
        ablate_layer = 0
        ablate_head = 0
        
        # Capture outputs with L0H00 ablated
        ablated_captures = _capture_head_outputs_ablated(
            self.model, self.tokenizer, self.text, ablate_layer, ablate_head
        )
        
        # Captured slice for ablated head L0H00 must be exactly zero
        ablated_slice = ablated_captures[(ablate_layer, ablate_head)]
        self.assertEqual(
            torch.sum(torch.abs(ablated_slice)).item(),
            0.0,
            f"Ablation hook failed: L{ablate_layer}H{ablate_head} representation is not zero."
        )

        # Captured slice for unablated head L0H01 must NOT be zero
        unablated_slice = ablated_captures[(ablate_layer, 1)]
        self.assertNotEqual(
            torch.sum(torch.abs(unablated_slice)).item(),
            0.0,
            "Unablated head representation was silently zeroed out."
        )
        
        # Captured slice for downstream head L1H00 must NOT be zero (propagation test)
        downstream_slice = ablated_captures[(1, 0)]
        self.assertNotEqual(
            torch.sum(torch.abs(downstream_slice)).item(),
            0.0,
            "Downstream head representation was silently zeroed out."
        )

        print("\n  [OK] Hook order verification passed: Ablation runs before capture.")

if __name__ == "__main__":
    unittest.main()
