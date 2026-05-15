"""
Modality-align fusion for event sequences: text / image / skipgram (subset),
decoupling, heterogeneity + homogeneity, causal attention along the event axis.
Time is concatenated after this module in model.py (not an alignment branch).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from decalign_transformer import TransformerEncoder, build_causal_attn_mask


def _active_slots_from_indices(active_indices: Sequence[int]) -> Tuple[bool, bool, bool]:
    s = set(int(i) for i in active_indices)
    if not s.issubset({0, 1, 2}) or len(s) != len(active_indices):
        raise ValueError("active_indices must be unique subset of {0,1,2}")
    return (0 in s, 1 in s, 2 in s)


class ModalityAlignFusion(nn.Module):
    """
    ``active_indices``: which of (text, image, skipgram) participate, e.g. (0,1,2), (0,2), (0,).
    Inputs ``text_x, image_x, skipgram_x`` are always ``[B,S,embed_dim]``; inactive slots may be zeros.
    Losses and OT use **only** active modalities; fusion concat uses active branches only.
    Returns ``event_partial`` ``[B,S,hidden_size]`` (no time); model concatenates time then projects to LLM.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_size: int,
        d_model: int,
        num_heads: int,
        nlevels: int,
        num_prototypes: int,
        active_indices: Sequence[int],
        conv1d_kernel_size: int = 1,
        attn_dropout: float = 0.1,
        attn_dropout_a: float = 0.1,
        attn_dropout_v: float = 0.1,
        relu_dropout: float = 0.1,
        embed_dropout: float = 0.1,
        res_dropout: float = 0.1,
        lambda_ot: float = 0.1,
        ot_num_iters: int = 50,
        use_recon: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.d = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layers = nlevels
        self.num_prototypes = max(1, int(num_prototypes))
        self.ot_reg = lambda_ot
        self.ot_num_iters = ot_num_iters
        k = conv1d_kernel_size
        self.conv1d_kernel_size = k

        self.active_indices: Tuple[int, ...] = tuple(int(i) for i in active_indices)
        self.active_mask = _active_slots_from_indices(self.active_indices)
        self.M = len(self.active_indices)
        if self.M < 1:
            raise ValueError("Need at least one active modality for ModalityAlignFusion")

        if (self.M * self.d) % num_heads != 0:
            raise ValueError(f"(M*d)={self.M * self.d} must be divisible by num_heads={num_heads}")
        if self.d % num_heads != 0:
            raise ValueError(f"d={self.d} must be divisible by num_heads={num_heads}")

        self.ad_l = attn_dropout
        self.ad_a = attn_dropout_a
        self.ad_v = attn_dropout_v
        self._slot_attn_drop = {0: self.ad_l, 1: self.ad_a, 2: self.ad_v}

        self.projs = nn.ModuleDict()
        self.encoder_uni = nn.ModuleDict()
        for idx in self.active_indices:
            self.projs[str(idx)] = nn.Conv1d(embed_dim, self.d, kernel_size=k, padding=0, bias=False)
            self.encoder_uni[str(idx)] = nn.Conv1d(self.d, self.d, kernel_size=1, padding=0, bias=False)
        self.encoder_com = nn.Conv1d(self.d, self.d, kernel_size=1, padding=0, bias=False)

        for idx in self.active_indices:
            self.register_parameter(f"proto_{idx}", nn.Parameter(torch.randn(self.num_prototypes, self.d)))
            self.register_parameter(f"logvar_{idx}", nn.Parameter(torch.zeros(self.num_prototypes, self.d)))

        self.transformer_fusion = TransformerEncoder(
            embed_dim=self.M * self.d,
            num_heads=num_heads,
            layers=nlevels,
            attn_dropout=attn_dropout,
            relu_dropout=relu_dropout,
            res_dropout=res_dropout,
            embed_dropout=embed_dropout,
            attn_mask=True,
        )

        self.cross_nets = nn.ModuleDict()
        self.mem_nets = nn.ModuleDict()
        if self.M > 1:
            for mi in self.active_indices:
                others = [j for j in self.active_indices if j != mi]
                for oj in others:
                    key = f"{mi}_{oj}"
                    self.cross_nets[key] = self._make_cross_net(self.d, self._slot_attn_drop[oj])
                mem_in = (self.M - 1) * self.d
                if mem_in % num_heads != 0:
                    raise ValueError(f"(M-1)*d={mem_in} must be divisible by num_heads={num_heads}")
                self.mem_nets[str(mi)] = TransformerEncoder(
                    embed_dim=mem_in,
                    num_heads=num_heads,
                    layers=max(self.layers, 3),
                    attn_dropout=self._slot_attn_drop[mi],
                    relu_dropout=0.1,
                    res_dropout=0.1,
                    embed_dropout=0.1,
                    attn_mask=False,
                )

        cma_in_dim = self.M * (self.M - 1) * self.d if self.M > 1 else self.d
        self.cma_proj = nn.Linear(cma_in_dim, self.M * self.d) if self.M > 1 else nn.Identity()
        self.to_hidden = nn.Linear(2 * self.M * self.d, hidden_size)

        self.use_recon = use_recon
        self.recon_heads = nn.ModuleDict()
        if use_recon:
            for idx in self.active_indices:
                self.recon_heads[str(idx)] = nn.Conv1d(2 * self.d, self.d, kernel_size=1, bias=True)

    def _make_cross_net(self, embed_dim: int, ad: float) -> TransformerEncoder:
        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=self.layers,
            attn_dropout=ad,
            relu_dropout=0.1,
            res_dropout=0.1,
            embed_dropout=0.1,
            attn_mask=False,
        )

    @staticmethod
    def compute_decoupling_loss(s: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        n = s.size(0)
        s_flat = s.reshape(n, -1)
        c_flat = c.reshape(n, -1)
        cos_sim = F.cosine_similarity(s_flat, c_flat, dim=1)
        return cos_sim.mean()

    def compute_prototypes(self, features: torch.Tensor, proto: torch.Tensor, logvar: torch.Tensor):
        feat_avg = features.mean(dim=2)
        diff = feat_avg.unsqueeze(1) - proto.unsqueeze(0)
        dist_sq = (diff**2).sum(dim=2)
        w = F.softmax(-dist_sq, dim=1)
        return w

    @staticmethod
    def pairwise_cost(mu1, logvar1, mu2, logvar2, eps=1e-9):
        diff = mu1.unsqueeze(1) - mu2.unsqueeze(0)
        dist_sq = torch.sum(diff**2, dim=2)
        sigma1 = torch.exp(logvar1)
        sigma2 = torch.exp(logvar2)
        cov_term = torch.sum(
            sigma1.unsqueeze(1)
            + sigma2.unsqueeze(0)
            - 2 * torch.sqrt(sigma1.unsqueeze(1) * sigma2.unsqueeze(0) + eps),
            dim=2,
        )
        return dist_sq + cov_term

    def multi_marginal_sinkhorn(self, c_cost, nu_l, nu_a, nu_v, reg, num_iters=50, eps=1e-9):
        k = c_cost.size(0)
        k_tensor = torch.exp(-c_cost / reg)
        u = torch.ones(k, device=c_cost.device, dtype=c_cost.dtype)
        v = torch.ones(k, device=c_cost.device, dtype=c_cost.dtype)
        w = torch.ones(k, device=c_cost.device, dtype=c_cost.dtype)
        for _ in range(num_iters):
            u = nu_l / (torch.sum(k_tensor * v.view(1, k, 1) * w.view(1, 1, k), dim=(1, 2)) + eps)
            v = nu_a / (torch.sum(k_tensor * u.view(k, 1, 1) * w.view(1, 1, k), dim=(0, 2)) + eps)
            w = nu_v / (torch.sum(k_tensor * u.view(k, 1, 1) * v.view(1, k, 1), dim=(0, 1)) + eps)
        t_mat = (u.view(k, 1, 1) * v.view(1, k, 1) * w.view(1, 1, k)) * k_tensor
        ot_loss = torch.sum(t_mat * c_cost)
        entropy = -torch.sum(t_mat * torch.log(t_mat + eps))
        ot_loss = ot_loss + 0.001 * reg * entropy
        return t_mat, ot_loss

    @staticmethod
    def sinkhorn2_marginal(cost: torch.Tensor, nu: torch.Tensor, mu: torch.Tensor, reg: float, num_iters: int, eps: float = 1e-9):
        """cost, transport on K×K; nu, mu shape [K] positive."""
        k = cost.size(0)
        g = torch.exp(-cost / reg)
        u = torch.ones(k, device=cost.device, dtype=cost.dtype)
        v = torch.ones(k, device=cost.device, dtype=cost.dtype)
        for _ in range(num_iters):
            u = nu / (g @ v + eps)
            v = mu / (g.transpose(0, 1) @ u + eps)
        t_mat = u.unsqueeze(1) * g * v.unsqueeze(0)
        ot_loss = torch.sum(t_mat * cost)
        entropy = -torch.sum(t_mat * torch.log(t_mat + eps))
        ot_loss = ot_loss + 0.001 * reg * entropy
        return t_mat, ot_loss

    def compute_hetero_loss_m3(self, tensors: List[torch.Tensor], protos: List[torch.Tensor], logvars: List[torch.Tensor]):
        s0, s1, s2 = tensors
        p0, p1, p2 = protos
        lv0, lv1, lv2 = logvars
        w0 = self.compute_prototypes(s0, p0, lv0)
        w1 = self.compute_prototypes(s1, p1, lv1)
        w2 = self.compute_prototypes(s2, p2, lv2)
        eps = 1e-9
        nu0 = w0.mean(dim=0)
        nu1 = w1.mean(dim=0)
        nu2 = w2.mean(dim=0)
        nu0 = nu0 / (nu0.sum() + eps)
        nu1 = nu1 / (nu1.sum() + eps)
        nu2 = nu2 / (nu2.sum() + eps)
        cost_01 = self.pairwise_cost(p0, lv0, p1, lv1, eps=eps)
        cost_02 = self.pairwise_cost(p0, lv0, p2, lv2, eps=eps)
        cost_12 = self.pairwise_cost(p1, lv1, p2, lv2, eps=eps)
        c_tensor = cost_01.unsqueeze(2) + cost_02.unsqueeze(1) + cost_12.unsqueeze(0)
        _, ot_loss = self.multi_marginal_sinkhorn(c_tensor, nu0, nu1, nu2, reg=self.ot_reg, num_iters=self.ot_num_iters)
        f0 = s0.mean(dim=2)
        f1 = s1.mean(dim=2)
        f2 = s2.mean(dim=2)
        local = 0.0
        local = local + torch.mean(w0 * torch.sum((f0.unsqueeze(1) - p1.unsqueeze(0)) ** 2, dim=2))
        local = local + torch.mean(w0 * torch.sum((f0.unsqueeze(1) - p2.unsqueeze(0)) ** 2, dim=2))
        local = local + torch.mean(w1 * torch.sum((f1.unsqueeze(1) - p0.unsqueeze(0)) ** 2, dim=2))
        local = local + torch.mean(w1 * torch.sum((f1.unsqueeze(1) - p2.unsqueeze(0)) ** 2, dim=2))
        local = local + torch.mean(w2 * torch.sum((f2.unsqueeze(1) - p0.unsqueeze(0)) ** 2, dim=2))
        local = local + torch.mean(w2 * torch.sum((f2.unsqueeze(1) - p1.unsqueeze(0)) ** 2, dim=2))
        return ot_loss + local

    def compute_hetero_loss_m2(self, tensors: List[torch.Tensor], protos: List[torch.Tensor], logvars: List[torch.Tensor]):
        s0, s1 = tensors
        p0, p1 = protos
        lv0, lv1 = logvars
        w0 = self.compute_prototypes(s0, p0, lv0)
        w1 = self.compute_prototypes(s1, p1, lv1)
        eps = 1e-9
        nu0 = w0.mean(dim=0)
        nu1 = w1.mean(dim=0)
        nu0 = nu0 / (nu0.sum() + eps)
        nu1 = nu1 / (nu1.sum() + eps)
        cost = self.pairwise_cost(p0, lv0, p1, lv1, eps=eps)
        _, ot_loss = self.sinkhorn2_marginal(cost, nu0, nu1, self.ot_reg, self.ot_num_iters, eps=eps)
        f0 = s0.mean(dim=2)
        f1 = s1.mean(dim=2)
        local = torch.mean(w0 * torch.sum((f0.unsqueeze(1) - p1.unsqueeze(0)) ** 2, dim=2))
        local = local + torch.mean(w1 * torch.sum((f1.unsqueeze(1) - p0.unsqueeze(0)) ** 2, dim=2))
        return ot_loss + local

    def compute_hetero_loss(self, s_by_idx: Dict[int, torch.Tensor]) -> torch.Tensor:
        order = list(self.active_indices)
        tensors = [s_by_idx[i] for i in order]
        protos = [getattr(self, f"proto_{i}") for i in order]
        logvars = [getattr(self, f"logvar_{i}") for i in order]
        if self.M == 1:
            return torch.tensor(0.0, device=tensors[0].device, dtype=tensors[0].dtype)
        if self.M == 2:
            return self.compute_hetero_loss_m2(tensors, protos, logvars)
        return self.compute_hetero_loss_m3(tensors, protos, logvars)

    @staticmethod
    def compute_mmd(x, y, kernel_bandwidth=1.0):
        xx = torch.mm(x, x.t())
        yy = torch.mm(y, y.t())
        xy = torch.mm(x, y.t())
        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)
        k_xx = torch.exp(-(rx.t() + rx - 2 * xx) / (2 * kernel_bandwidth))
        k_yy = torch.exp(-(ry.t() + ry - 2 * yy) / (2 * kernel_bandwidth))
        k_xy = torch.exp(-(rx.t() + ry - 2 * xy) / (2 * kernel_bandwidth))
        return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

    def compute_homo_loss(self, c_by_idx: Dict[int, torch.Tensor]) -> torch.Tensor:
        order = list(self.active_indices)
        if self.M == 1:
            return torch.tensor(0.0, device=c_by_idx[order[0]].device, dtype=c_by_idx[order[0]].dtype)

        def compute_stats(c):
            mu = c.mean(dim=(0, 2))
            sigma = c.var(dim=(0, 2))
            centered = c - mu.view(1, -1, 1)
            skew = (centered**3).mean(dim=(0, 2)) / (sigma + 1e-6).pow(1.5)
            return mu, sigma, skew

        stats_list = [compute_stats(c_by_idx[i]) for i in order]
        l_sem = torch.tensor(0.0, device=c_by_idx[order[0]].device, dtype=c_by_idx[order[0]].dtype)
        for a in range(self.M):
            for b in range(a + 1, self.M):
                mu_a, sig_a, sk_a = stats_list[a]
                mu_b, sig_b, sk_b = stats_list[b]
                l_sem = l_sem + (mu_a - mu_b).pow(2).sum()
                l_sem = l_sem + (sig_a - sig_b).pow(2).sum()
                l_sem = l_sem + (sk_a - sk_b).pow(2).sum()

        mmd_sum = torch.tensor(0.0, device=c_by_idx[order[0]].device, dtype=c_by_idx[order[0]].dtype)
        pools = [c_by_idx[i].mean(dim=2) for i in order]
        for a in range(self.M):
            for b in range(a + 1, self.M):
                mmd_sum = mmd_sum + self.compute_mmd(pools[a], pools[b])
        return l_sem + mmd_sum

    @staticmethod
    def _causal_prefix_mean(c: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        m = pad_mask.to(dtype=c.dtype).unsqueeze(1)
        masked = c * m
        cs = masked.cumsum(dim=2)
        cnt = m.cumsum(dim=2).clamp(min=1.0)
        return cs / cnt

    def _encode_inputs(
        self,
        text_x: torch.Tensor,
        image_x: torch.Tensor,
        skipgram_x: torch.Tensor,
        pad: torch.Tensor,
    ) -> Tuple[
        Dict[int, torch.Tensor],
        Dict[int, torch.Tensor],
        Dict[int, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        xs = {0: text_x, 1: image_x, 2: skipgram_x}
        s_by_idx: Dict[int, torch.Tensor] = {}
        c_by_idx: Dict[int, torch.Tensor] = {}
        proj_by_idx: Dict[int, torch.Tensor] = {}
        dec_loss = torch.tensor(0.0, device=text_x.device, dtype=text_x.dtype)
        recon_loss = torch.tensor(0.0, device=text_x.device, dtype=text_x.dtype)

        for idx in self.active_indices:
            x = xs[idx].transpose(1, 2)
            proj = self.projs[str(idx)](x)
            s_i = self.encoder_uni[str(idx)](proj)
            c_i = self.encoder_com(proj)
            s_by_idx[idx] = s_i
            c_by_idx[idx] = c_i
            proj_by_idx[idx] = proj
            dec_loss = dec_loss + self.compute_decoupling_loss(s_i, c_i)
            if self.use_recon:
                sc = torch.cat([s_i, c_i], dim=1)
                pred = self.recon_heads[str(idx)](sc)
                err = (pred - proj).pow(2).mean(dim=1)
                mask = pad.bool()
                if mask.any():
                    recon_loss = recon_loss + (err * mask.float()).sum() / (mask.float().sum().clamp(min=1.0))

        hete_loss = self.compute_hetero_loss(s_by_idx)
        homo_loss = self.compute_homo_loss(c_by_idx)
        return s_by_idx, c_by_idx, proj_by_idx, dec_loss, recon_loss, hete_loss, homo_loss

    def forward(
        self,
        text_x: torch.Tensor,
        image_x: torch.Tensor,
        skipgram_x: torch.Tensor,
        batch_non_pad_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pad = batch_non_pad_mask.bool()
        s_by_idx, c_by_idx, proj_by_idx, dec_loss, recon_loss, hete_loss, homo_loss = self._encode_inputs(
            text_x, image_x, skipgram_x, pad
        )

        sp_list = [s_by_idx[i].permute(2, 0, 1) for i in self.active_indices]
        t_target = min(t.size(0) for t in sp_list)
        sp_list = [t[:t_target] for t in sp_list]
        causal = build_causal_attn_mask(t_target, sp_list[0].device)

        fused = torch.cat(sp_list, dim=2)
        trans_out = self.transformer_fusion(fused, attn_mask=causal)
        fusion_rep_trans = trans_out.permute(1, 0, 2)

        if self.M == 1:
            fusion_rep_cma = torch.zeros_like(fusion_rep_trans)
        else:
            h_blocks = []
            for mi in self.active_indices:
                cross_chunks = []
                qi = sp_list[self.active_indices.index(mi)]
                for oj in self.active_indices:
                    if oj == mi:
                        continue
                    kj = sp_list[self.active_indices.index(oj)]
                    key = f"{mi}_{oj}"
                    h_ij = self.cross_nets[key](qi, kj, kj, attn_mask=causal)
                    cross_chunks.append(h_ij)
                h_cat = torch.cat(cross_chunks, dim=2)
                h_mem = self.mem_nets[str(mi)](h_cat, attn_mask=causal)
                h_blocks.append(h_mem)
            cma_in = torch.cat(h_blocks, dim=2).permute(1, 0, 2)
            bsz, seq_len, cma_dim = cma_in.shape
            if isinstance(self.cma_proj, nn.Identity):
                fusion_rep_cma = cma_in
            else:
                fusion_rep_cma = self.cma_proj(cma_in.reshape(bsz * seq_len, cma_dim)).reshape(bsz, seq_len, self.M * self.d)

        homo_parts = []
        for idx in self.active_indices:
            c_i = c_by_idx[idx]
            homo_parts.append(self._causal_prefix_mean(c_i, pad))
        fusion_rep_homo = torch.cat(homo_parts, dim=1).permute(0, 2, 1)

        fusion_rep_hete = fusion_rep_trans + fusion_rep_cma
        final_rep = torch.cat([fusion_rep_hete, fusion_rep_homo], dim=2)
        event_partial = self.to_hidden(final_rep)

        return {
            "event_partial": event_partial,
            "dec_loss": dec_loss,
            "hete_loss": hete_loss,
            "homo_loss": homo_loss,
            "recon_loss": recon_loss,
            "final_rep": final_rep,
            "s_by_idx": s_by_idx,
            "c_by_idx": c_by_idx,
        }


class DecAlignEventFusion(ModalityAlignFusion):
    """Backward-compatible alias: three slots (indices 0,1,2) — historically type/text/time."""

    def __init__(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.setdefault("active_indices", (0, 1, 2))
        kwargs.setdefault("use_recon", False)
        super().__init__(*args, **kwargs)

    def forward(self, type_x, text_x, time_x, batch_non_pad_mask):  # type: ignore[override]
        out = super().forward(type_x, text_x, time_x, batch_non_pad_mask)
        out["event_hidden"] = out["event_partial"]
        return out
