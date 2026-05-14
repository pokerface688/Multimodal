"""Smoke tests for MoIR and config guard (no full LLM load)."""
import unittest

import torch

from moir import MoIR, check_moir_encoding


class TestMoIRSmoke(unittest.TestCase):
    def test_forward_shapes_symmetric(self):
        B, S, D = 2, 8, 32
        m = MoIR(embed_dim=D, exchange_ratio=0.12, top_q=16, token_subsample=8, symmetric=True)
        a = torch.randn(B, S, D, requires_grad=True)
        b = torch.randn(B, S, D, requires_grad=True)
        mask = torch.ones(B, S, dtype=torch.bool)
        mask[0, 6:] = False
        ao, bo = m(a, b, mask)
        self.assertEqual(ao.shape, a.shape)
        self.assertEqual(bo.shape, b.shape)

    def test_gradient_reaches_alpha(self):
        B, S, D = 1, 6, 16
        m = MoIR(embed_dim=D, exchange_ratio=0.2, top_q=8, symmetric=False)
        a = torch.randn(B, S, D, requires_grad=True)
        b = torch.randn(B, S, D, requires_grad=True)
        mask = torch.ones(B, S, dtype=torch.bool)
        ao, _ = m(a, b, mask)
        loss = ao.sum()
        loss.backward()
        self.assertIsNotNone(m.alpha_logits.grad)
        self.assertFalse(torch.isnan(m.alpha_logits.grad).any())

    def test_check_moir_encoding_ok(self):
        check_moir_encoding(False, "RoPE")
        check_moir_encoding(True, "TimePositionEncoding")

    def test_check_moir_encoding_raises(self):
        with self.assertRaises(ValueError):
            check_moir_encoding(True, "RoPE")


if __name__ == "__main__":
    unittest.main()
