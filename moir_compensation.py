"""MoIR (Multi-modal Information Router) — ported from upstream for this repo.

Source: https://github.com/olivesgatech/MoIR (moir/router.py)
Paper: Kim, Prabhushankar, AlRegib — Information Router for Mitigating Modality Dominance in VLM (2026).

Implements Sec. 3.2–3.3: truncated-SVD channel informativeness, bottom-k' channels,
learnable per-channel alpha gates, RMS-matched injection from the other modality's pool.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class MoIR(nn.Module):
    """Multi-modal Information Router.

    Args:
        hidden_size: Embedding dim D shared by the two modalities.
        exchange_ratio: k' in the paper. Fraction of channels to route.
        top_q: Truncated SVD rank used to compute channel scores.
        token_subsample: Cap on tokens used for SVD to bound cost. 0 disables.
        symmetric: If True, route both A->B and B->A. If False, only enrich A from B.
        init_alpha: Initial value of the routing gate sigmoid(alpha).
    """

    def __init__(
        self,
        hidden_size: int,
        exchange_ratio: float = 0.10,
        top_q: int = 64,
        token_subsample: int = 256,
        symmetric: bool = True,
        init_alpha: float = 0.5,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.exchange_ratio = float(exchange_ratio)
        self.top_q = int(top_q)
        self.token_subsample = int(token_subsample)
        self.symmetric = bool(symmetric)

        init_logit = math.log(init_alpha / (1.0 - init_alpha + 1e-12) + 1e-12)
        self.alpha_a_logits = nn.Parameter(torch.full((hidden_size,), float(init_logit)))
        self.alpha_b_logits = nn.Parameter(torch.full((hidden_size,), float(init_logit)))

    @torch.no_grad()
    def _channel_informativeness(self, X: torch.Tensor) -> Optional[torch.Tensor]:
        """Eq. (1): S_d = sum_i sigma_i^2 * v_{i,d}^2; variance fallback on failure."""
        if X is None or X.numel() == 0 or not torch.isfinite(X).all():
            return None
        T, D = X.shape
        if T < 2 or D < 2:
            return None

        if self.token_subsample > 0 and T > self.token_subsample:
            X = X[: self.token_subsample]

        X = X.float()
        X = X - X.mean(dim=0, keepdim=True)

        def var_fallback(Z: torch.Tensor) -> torch.Tensor:
            return torch.nan_to_num((Z * Z).mean(dim=0), nan=0.0, posinf=0.0, neginf=0.0)

        q = min(self.top_q, X.shape[0] - 1, X.shape[1])
        if q < 1:
            return var_fallback(X)

        try:
            _, S, V = torch.svd_lowrank(X, q=q, niter=2)
            scores = V.pow(2) @ S.pow(2)
            scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
            if not torch.isfinite(scores).all():
                return var_fallback(X)
            return scores
        except Exception:
            return var_fallback(X)

    @staticmethod
    def _rms_match(dst: torch.Tensor, src: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        rms_dst = torch.sqrt((dst * dst).mean() + eps)
        rms_src = torch.sqrt((src * src).mean() + eps)
        scale = (rms_dst / (rms_src + eps)).clamp(0.1, 10.0)
        return src * scale

    def forward(
        self,
        a_embeds: torch.Tensor,
        b_embeds: torch.Tensor,
        a_valid_mask: Optional[torch.Tensor] = None,
        b_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route between modality A and B. Shapes [B, L, D]; masks True = valid token."""
        if not torch.isfinite(a_embeds).all() or not torch.isfinite(b_embeds).all():
            return a_embeds, b_embeds

        device = a_embeds.device
        orig_dtype = a_embeds.dtype
        B, La, D = a_embeds.shape
        _, Lb, _ = b_embeds.shape

        if a_valid_mask is None:
            a_valid_mask = torch.ones((B, La), device=device, dtype=torch.bool)
        if b_valid_mask is None:
            b_valid_mask = torch.ones((B, Lb), device=device, dtype=torch.bool)

        a = a_embeds.float()
        b = b_embeds.float()

        a_pool = self._masked_mean(a, a_valid_mask)
        b_pool = self._masked_mean(b, b_valid_mask)

        a_out = a.clone()
        b_out = b.clone()

        k_low = max(1, int(round(self.exchange_ratio * D)))
        alpha_a = torch.sigmoid(self.alpha_a_logits.to(device).float()).view(1, 1, D)
        alpha_b = torch.sigmoid(self.alpha_b_logits.to(device).float()).view(1, 1, D)

        for batch_idx in range(B):
            with torch.no_grad():
                X_a = a_out[batch_idx][a_valid_mask[batch_idx]]
                S_a = self._channel_informativeness(X_a)
                sel_a = (
                    torch.topk(S_a, k=k_low, largest=False).indices
                    if S_a is not None and torch.isfinite(S_a).all()
                    else None
                )

            if sel_a is not None and sel_a.numel() > 0:
                a_out[batch_idx : batch_idx + 1] = self._apply_routing(
                    dst=a_out[batch_idx : batch_idx + 1],
                    src_pool=b_pool[batch_idx : batch_idx + 1],
                    channels=sel_a,
                    alpha=alpha_a,
                    D=D,
                )

            if self.symmetric:
                with torch.no_grad():
                    X_b = b_out[batch_idx][b_valid_mask[batch_idx]]
                    S_b = self._channel_informativeness(X_b)
                    sel_b = (
                        torch.topk(S_b, k=k_low, largest=False).indices
                        if S_b is not None and torch.isfinite(S_b).all()
                        else None
                    )

                if sel_b is not None and sel_b.numel() > 0:
                    b_out[batch_idx : batch_idx + 1] = self._apply_routing(
                        dst=b_out[batch_idx : batch_idx + 1],
                        src_pool=a_pool[batch_idx : batch_idx + 1],
                        channels=sel_b,
                        alpha=alpha_b,
                        D=D,
                    )

        if not torch.isfinite(a_out).all() or not torch.isfinite(b_out).all():
            return a_embeds, b_embeds

        return a_out.to(orig_dtype), b_out.to(orig_dtype)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).float()
        denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (x * m).sum(dim=1, keepdim=True) / denom

    def _apply_routing(
        self,
        dst: torch.Tensor,
        src_pool: torch.Tensor,
        channels: torch.Tensor,
        alpha: torch.Tensor,
        D: int,
    ) -> torch.Tensor:
        """Eq. (3): on selected channels, blend pooled other-modality signal."""
        L = dst.shape[1]
        src = src_pool.expand(-1, L, -1)
        src = self._rms_match(dst[:, :, channels], src[:, :, channels])

        mask = torch.zeros((D,), device=dst.device, dtype=torch.float32)
        mask[channels] = 1.0
        mask = mask.view(1, 1, D)

        full_src = torch.zeros_like(dst)
        full_src[:, :, channels] = src
        blended = alpha * full_src + (1.0 - alpha) * dst
        return dst * (1.0 - mask) + blended * mask
