"""Smoke tests for DecAlign event fusion (no GPU / no LLM)."""
import unittest
import torch
from decalign_event import DecAlignEventFusion, ModalityAlignFusion
from decalign_transformer import build_causal_attn_mask


class TestModalityAlignFusion(unittest.TestCase):
    def test_m2_forward(self):
        B, S, E, H = 2, 8, 24, 40
        m = ModalityAlignFusion(
            embed_dim=E,
            hidden_size=H,
            d_model=16,
            num_heads=4,
            nlevels=1,
            num_prototypes=4,
            active_indices=(0, 2),
        )
        pad = torch.ones(B, S, dtype=torch.bool)
        x = torch.randn(B, S, E)
        out = m(x, x, x, pad)
        self.assertEqual(out["event_partial"].shape, (B, S, H))

    def test_m1_forward(self):
        B, S, E, H = 2, 5, 16, 32
        m = ModalityAlignFusion(
            embed_dim=E,
            hidden_size=H,
            d_model=16,
            num_heads=4,
            nlevels=1,
            num_prototypes=3,
            active_indices=(1,),
        )
        pad = torch.ones(B, S, dtype=torch.bool)
        x = torch.randn(B, S, E)
        out = m(x, x, x, pad)
        self.assertEqual(out["event_partial"].shape, (B, S, H))


class TestDecAlignEvent(unittest.TestCase):
    def test_forward_shapes(self):
        B, S, E, H = 3, 12, 32, 48
        m = DecAlignEventFusion(
            embed_dim=E,
            hidden_size=H,
            d_model=32,
            num_heads=4,
            nlevels=2,
            num_prototypes=5,
        )
        pad = torch.ones(B, S, dtype=torch.bool)
        x = torch.randn(B, S, E)
        out = m(x, x, x, pad)
        self.assertEqual(out["event_hidden"].shape, (B, S, H))
        self.assertEqual(out["final_rep"].shape[-1], 6 * 32)

    def test_last_step_depends_on_first(self):
        """Last timestep output should change if an earlier timestep input changes."""
        torch.manual_seed(1)
        B, S, E, H = 1, 6, 16, 24
        m = DecAlignEventFusion(
            embed_dim=E,
            hidden_size=H,
            d_model=16,
            num_heads=4,
            nlevels=1,
            num_prototypes=3,
        )
        pad = torch.ones(B, S, dtype=torch.bool)
        x = torch.randn(B, S, E)
        x2 = x.clone()
        x2[:, 0, :] += 5.0
        y1 = m(x, x, x, pad)["event_hidden"][0, -1]
        y2 = m(x2, x2, x2, pad)["event_hidden"][0, -1]
        self.assertFalse(torch.allclose(y1, y2, atol=1e-4))

    def test_causal_mask_upper_tri(self):
        m = build_causal_attn_mask(4, torch.device("cpu"))
        self.assertTrue(m[0, 1])


if __name__ == "__main__":
    unittest.main()
