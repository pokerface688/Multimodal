"""
MoIR: dual-stream channel exchange via truncated SVD channel scoring (see project memory).
Used after bidirectional fusion: enriches event stream A and time stream B in embed_dim space.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def check_moir_encoding(use_moir: bool, tem_enc_type: str) -> None:
    if use_moir and tem_enc_type != "TimePositionEncoding":
        raise ValueError(
            "use_moir requires tem_enc_type == 'TimePositionEncoding' "
            f"(got tem_enc_type={tem_enc_type!r})."
        )


class MoIR(nn.Module):
    """
    Args:
        embed_dim: D for both streams
        exchange_ratio: fraction of channels (bottom by SVD score) to blend
        top_q: max rank components for SVD-based scores
        token_subsample: max valid tokens per sample for SVD (0 = use all)
        symmetric: if True, enrich both A<-B and B<-A
        init_alpha: initial blend weight toward source after sigmoid
    """

    def __init__(
        self,
        embed_dim: int,
        exchange_ratio: float = 0.10,
        top_q: int = 64,
        token_subsample: int = 0,
        symmetric: bool = True,
        init_alpha: float = 0.5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.exchange_ratio = exchange_ratio
        self.top_q = top_q
        self.token_subsample = token_subsample
        self.symmetric = symmetric
        init_alpha = float(min(max(init_alpha, 1e-4), 1.0 - 1e-4))
        logit = math.log(init_alpha / (1.0 - init_alpha))
        self.alpha_logits = nn.Parameter(torch.full((embed_dim,), logit))

    def _channel_scores(self, X: torch.Tensor) -> torch.Tensor:
        """X: [T, D] float32, centered or not — returns scores [D] on CPU/device of X."""
        T, D = X.shape
        if T < 2 or D < 2:
            return X.float().var(dim=0)

        Xc = X - X.mean(dim=0, keepdim=True)
        q = min(self.top_q, T - 1, D)
        if q < 1:
            return X.float().var(dim=0)

        # torch.linalg.svd: Vh shape [min(T,D), D]
        _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        q_eff = min(q, S.shape[0], Vh.shape[0])
        Vq = Vh[:q_eff, :]
        Sq = S[:q_eff]
        scores = (Vq * Vq).t() @ (Sq * Sq)
        return scores

    def _rms_match_scalar(self, seq_vals: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        """Match RMS of constant pooled to RMS of seq_vals (1D same length)."""
        rms_t = seq_vals.pow(2).mean().sqrt().clamp(min=1e-6)
        rms_s = pooled.abs().clamp(min=1e-6)
        return pooled * (rms_t / rms_s)

    def _enrich_stream(
        self,
        target: torch.Tensor,
        other: torch.Tensor,
        non_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        target, other: [B, S, D], non_pad_mask: [B, S] bool True = valid
        Returns updated target (same dtype/shape).
        """
        B, S, D = target.shape
        out = target.clone()
        dtype = target.dtype
        alpha = torch.sigmoid(self.alpha_logits).to(dtype=dtype)

        k_low = max(1, int(round(self.exchange_ratio * D)))

        for b in range(B):
            m = non_pad_mask[b].bool()
            if not m.any():
                continue
            idx_all = m.nonzero(as_tuple=False).squeeze(-1)
            if self.token_subsample > 0 and idx_all.numel() > self.token_subsample:
                idx = idx_all[: self.token_subsample]
            else:
                idx = idx_all

            X = target[b, idx].detach().float()
            with torch.no_grad():
                scores = self._channel_scores(X)
                _, bottom_idx = torch.topk(scores, k=min(k_low, D), largest=False)

            pooled_other = (other[b].float() * m.unsqueeze(-1).float()).sum(dim=0) / m.sum().clamp(min=1).float()

            for d in bottom_idx.tolist():
                tv = target[b, m, d].float()
                s_raw = pooled_other[d]
                s_adj = self._rms_match_scalar(tv, s_raw).to(dtype=dtype)
                a = alpha[d]
                out[b, m, d] = a * s_adj + (1.0 - a) * target[b, m, d]

        return out

    def forward(
        self,
        a_embeds: torch.Tensor,
        b_embeds: torch.Tensor,
        non_pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            a_embeds: [B, S, D] (e.g. fused event)
            b_embeds: [B, S, D] (e.g. time positional encoding)
            non_pad_mask: [B, S] bool or 0/1; True/1 = valid token
        Returns:
            (a_out, b_out) same shapes as inputs
        """
        if non_pad_mask is None:
            non_pad_mask = torch.ones(
                a_embeds.shape[0], a_embeds.shape[1], dtype=torch.bool, device=a_embeds.device
            )
        else:
            non_pad_mask = non_pad_mask.bool()

        a_out = self._enrich_stream(a_embeds, b_embeds, non_pad_mask)
        if self.symmetric:
            b_out = self._enrich_stream(b_embeds, a_embeds, non_pad_mask)
        else:
            b_out = b_embeds
        return a_out, b_out
