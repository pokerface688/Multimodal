"""
DecAlign-style fusion for event sequences: type / text / time streams, causal attention
along the event axis, heterogeneity + homogeneity branches, and auxiliary losses.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from decalign_transformer import TransformerEncoder, build_causal_attn_mask


class DecAlignEventFusion(nn.Module):
    """
    Three modalities (type, text, time) share target dim ``d`` after Conv1d projection.
    Returns per-step fused hidden vectors for the LLM and scalar dec/hete/homo losses.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_size: int,
        d_model: int,
        num_heads: int,
        nlevels: int,
        num_prototypes: int,
        conv1d_kernel_size: int = 1,
        attn_dropout: float = 0.1,
        attn_dropout_a: float = 0.1,
        attn_dropout_v: float = 0.1,
        relu_dropout: float = 0.1,
        embed_dropout: float = 0.1,
        res_dropout: float = 0.1,
        lambda_ot: float = 0.1,
        ot_num_iters: int = 50,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.d = d_model
        self.num_heads = num_heads
        self.layers = nlevels
        self.num_prototypes = num_prototypes
        self.ot_reg = lambda_ot
        self.ot_num_iters = ot_num_iters
        k = conv1d_kernel_size
        self.conv1d_kernel_size = k

        self.proj_l = nn.Conv1d(embed_dim, self.d, kernel_size=k, padding=0, bias=False)
        self.proj_a = nn.Conv1d(embed_dim, self.d, kernel_size=k, padding=0, bias=False)
        self.proj_v = nn.Conv1d(embed_dim, self.d, kernel_size=k, padding=0, bias=False)

        self.encoder_uni_l = nn.Conv1d(self.d, self.d, kernel_size=1, padding=0, bias=False)
        self.encoder_uni_a = nn.Conv1d(self.d, self.d, kernel_size=1, padding=0, bias=False)
        self.encoder_uni_v = nn.Conv1d(self.d, self.d, kernel_size=1, padding=0, bias=False)
        self.encoder_com = nn.Conv1d(self.d, self.d, kernel_size=1, padding=0, bias=False)

        self.proto_l = nn.Parameter(torch.randn(num_prototypes, self.d))
        self.proto_a = nn.Parameter(torch.randn(num_prototypes, self.d))
        self.proto_v = nn.Parameter(torch.randn(num_prototypes, self.d))
        self.logvar_l = nn.Parameter(torch.zeros(num_prototypes, self.d))
        self.logvar_a = nn.Parameter(torch.zeros(num_prototypes, self.d))
        self.logvar_v = nn.Parameter(torch.zeros(num_prototypes, self.d))

        self.ad_l = attn_dropout
        self.ad_a = attn_dropout_a
        self.ad_v = attn_dropout_v

        self.transformer_fusion = TransformerEncoder(
            embed_dim=3 * self.d,
            num_heads=num_heads,
            layers=nlevels,
            attn_dropout=attn_dropout,
            relu_dropout=relu_dropout,
            res_dropout=res_dropout,
            embed_dropout=embed_dropout,
            attn_mask=True,
        )
        self.trans_l_with_a = self._make_net("la")
        self.trans_l_with_v = self._make_net("lv")
        self.trans_a_with_l = self._make_net("al")
        self.trans_a_with_v = self._make_net("av")
        self.trans_v_with_l = self._make_net("vl")
        self.trans_v_with_a = self._make_net("va")
        self.trans_l_mem = self._make_net("l_mem", layers=3)
        self.trans_a_mem = self._make_net("a_mem", layers=3)
        self.trans_v_mem = self._make_net("v_mem", layers=3)

        self.cma_proj = nn.Linear(6 * self.d, 3 * self.d)
        self.to_hidden = nn.Linear(6 * self.d, hidden_size)

    def _make_net(self, self_type: str, layers: int = -1):
        if self_type == "la":
            embed_dim, ad = self.d, self.ad_a
        elif self_type == "lv":
            embed_dim, ad = self.d, self.ad_v
        elif self_type in ("al", "vl"):
            embed_dim, ad = self.d, self.ad_l
        elif self_type in ("av",):
            embed_dim, ad = self.d, self.ad_v
        elif self_type == "va":
            embed_dim, ad = self.d, self.ad_a
        elif self_type == "l_mem":
            embed_dim, ad = 2 * self.d, self.ad_l
        elif self_type == "a_mem":
            embed_dim, ad = 2 * self.d, self.ad_a
        elif self_type == "v_mem":
            embed_dim, ad = 2 * self.d, self.ad_v
        else:
            raise ValueError(self_type)
        nl = max(self.layers, layers) if layers > 0 else self.layers
        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=nl,
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
        n, d_dim, t = features.shape
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

    def compute_hetero_loss(self, s_l, s_a, s_v):
        w_l = self.compute_prototypes(s_l, self.proto_l, self.logvar_l)
        w_a = self.compute_prototypes(s_a, self.proto_a, self.logvar_a)
        w_v = self.compute_prototypes(s_v, self.proto_v, self.logvar_v)
        eps = 1e-9
        nu_l = w_l.mean(dim=0)
        nu_a = w_a.mean(dim=0)
        nu_v = w_v.mean(dim=0)
        nu_l = nu_l / (nu_l.sum() + eps)
        nu_a = nu_a / (nu_a.sum() + eps)
        nu_v = nu_v / (nu_v.sum() + eps)
        cost_la = self.pairwise_cost(self.proto_l, self.logvar_l, self.proto_a, self.logvar_a, eps=eps)
        cost_lv = self.pairwise_cost(self.proto_l, self.logvar_l, self.proto_v, self.logvar_v, eps=eps)
        cost_av = self.pairwise_cost(self.proto_a, self.logvar_a, self.proto_v, self.logvar_v, eps=eps)
        c_tensor = cost_la.unsqueeze(2) + cost_lv.unsqueeze(1) + cost_av.unsqueeze(0)
        _, ot_loss = self.multi_marginal_sinkhorn(c_tensor, nu_l, nu_a, nu_v, reg=self.ot_reg, num_iters=self.ot_num_iters)
        feat_l = s_l.mean(dim=2)
        feat_a = s_a.mean(dim=2)
        feat_v = s_v.mean(dim=2)
        loss_la = torch.mean(w_l * torch.sum((feat_l.unsqueeze(1) - self.proto_a.unsqueeze(0)) ** 2, dim=2))
        loss_lv = torch.mean(w_l * torch.sum((feat_l.unsqueeze(1) - self.proto_v.unsqueeze(0)) ** 2, dim=2))
        loss_al = torch.mean(w_a * torch.sum((feat_a.unsqueeze(1) - self.proto_l.unsqueeze(0)) ** 2, dim=2))
        loss_av = torch.mean(w_a * torch.sum((feat_a.unsqueeze(1) - self.proto_v.unsqueeze(0)) ** 2, dim=2))
        loss_vl = torch.mean(w_v * torch.sum((feat_v.unsqueeze(1) - self.proto_l.unsqueeze(0)) ** 2, dim=2))
        loss_va = torch.mean(w_v * torch.sum((feat_v.unsqueeze(1) - self.proto_a.unsqueeze(0)) ** 2, dim=2))
        local_proto_loss = loss_la + loss_lv + loss_al + loss_av + loss_vl + loss_va
        return ot_loss + local_proto_loss

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

    def compute_homo_loss(self, c_l, c_a, c_v):
        def compute_stats(c):
            mu = c.mean(dim=(0, 2))
            sigma = c.var(dim=(0, 2))
            centered = c - mu.view(1, -1, 1)
            skew = (centered**3).mean(dim=(0, 2)) / (sigma + 1e-6).pow(1.5)
            return mu, sigma, skew

        mu_l, sigma_l, skew_l = compute_stats(c_l)
        mu_a, sigma_a, skew_a = compute_stats(c_a)
        mu_v, sigma_v, skew_v = compute_stats(c_v)
        l_sem = (
            (mu_l - mu_a).pow(2).sum()
            + (mu_l - mu_v).pow(2).sum()
            + (mu_a - mu_v).pow(2).sum()
            + (sigma_l - sigma_a).pow(2).sum()
            + (sigma_l - sigma_v).pow(2).sum()
            + (sigma_a - sigma_v).pow(2).sum()
            + (skew_l - skew_a).pow(2).sum()
            + (skew_l - skew_v).pow(2).sum()
            + (skew_a - skew_v).pow(2).sum()
        )
        c_l_pool = c_l.mean(dim=2)
        c_a_pool = c_a.mean(dim=2)
        c_v_pool = c_v.mean(dim=2)
        mmd_la = self.compute_mmd(c_l_pool, c_a_pool)
        mmd_lv = self.compute_mmd(c_l_pool, c_v_pool)
        mmd_av = self.compute_mmd(c_a_pool, c_v_pool)
        return l_sem + mmd_la + mmd_lv + mmd_av

    @staticmethod
    def _causal_prefix_mean(c: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """c: [B,d,S], pad_mask: [B,S] bool True=valid -> prefix mean per position [B,d,S] (causal)."""
        m = pad_mask.to(dtype=c.dtype).unsqueeze(1)
        masked = c * m
        cs = masked.cumsum(dim=2)
        cnt = m.cumsum(dim=2).clamp(min=1.0)
        return cs / cnt

    def forward(
        self,
        type_x: torch.Tensor,
        text_x: torch.Tensor,
        time_x: torch.Tensor,
        batch_non_pad_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        type_x, text_x, time_x: [B, S, embed_dim]
        batch_non_pad_mask: [B, S] bool, True for valid (non-pad) positions.
        """
        pad = batch_non_pad_mask.bool()
        x_l = type_x.transpose(1, 2)
        x_a = text_x.transpose(1, 2)
        x_v = time_x.transpose(1, 2)
        proj_l = self.proj_l(x_l)
        proj_a = self.proj_a(x_a)
        proj_v = self.proj_v(x_v)

        s_l = self.encoder_uni_l(proj_l)
        s_a = self.encoder_uni_a(proj_a)
        s_v = self.encoder_uni_v(proj_v)
        c_l = self.encoder_com(proj_l)
        c_a = self.encoder_com(proj_a)
        c_v = self.encoder_com(proj_v)

        dec_loss = (
            self.compute_decoupling_loss(s_l, c_l)
            + self.compute_decoupling_loss(s_a, c_a)
            + self.compute_decoupling_loss(s_v, c_v)
        )
        hete_loss = self.compute_hetero_loss(s_l, s_a, s_v)
        homo_loss = self.compute_homo_loss(c_l, c_a, c_v)

        s_l_p = s_l.permute(2, 0, 1)
        s_a_p = s_a.permute(2, 0, 1)
        s_v_p = s_v.permute(2, 0, 1)
        t_target = min(s_l_p.size(0), s_a_p.size(0), s_v_p.size(0))
        s_l_p = s_l_p[:t_target]
        s_a_p = s_a_p[:t_target]
        s_v_p = s_v_p[:t_target]

        causal = build_causal_attn_mask(t_target, s_l_p.device)
        fused = torch.cat([s_l_p, s_a_p, s_v_p], dim=2)
        trans_out = self.transformer_fusion(fused, attn_mask=causal)

        h_l_with_as = self.trans_l_with_a(s_l_p, s_a_p, s_a_p, attn_mask=causal)
        h_l_with_vs = self.trans_l_with_v(s_l_p, s_v_p, s_v_p, attn_mask=causal)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls, attn_mask=causal)
        h_a_with_ls = self.trans_a_with_l(s_a_p, s_l_p, s_l_p, attn_mask=causal)
        h_a_with_vs = self.trans_a_with_v(s_a_p, s_v_p, s_v_p, attn_mask=causal)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as, attn_mask=causal)
        h_v_with_ls = self.trans_v_with_l(s_v_p, s_l_p, s_l_p, attn_mask=causal)
        h_v_with_as = self.trans_v_with_a(s_v_p, s_a_p, s_a_p, attn_mask=causal)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs, attn_mask=causal)

        fusion_rep_trans = trans_out.permute(1, 0, 2)
        cma_in = torch.cat([h_ls, h_as, h_vs], dim=2).permute(1, 0, 2)
        bsz, seq_len, six_d = cma_in.shape
        fusion_rep_cma = self.cma_proj(cma_in.reshape(bsz * seq_len, six_d)).reshape(bsz, seq_len, 3 * self.d)

        c_l_pf = self._causal_prefix_mean(c_l, pad)
        c_a_pf = self._causal_prefix_mean(c_a, pad)
        c_v_pf = self._causal_prefix_mean(c_v, pad)
        fusion_rep_homo = torch.cat([c_l_pf, c_a_pf, c_v_pf], dim=1).permute(0, 2, 1)

        fusion_rep_hete = fusion_rep_trans + fusion_rep_cma
        final_rep = torch.cat([fusion_rep_hete, fusion_rep_homo], dim=2)
        event_hidden = self.to_hidden(final_rep)

        return {
            "event_hidden": event_hidden,
            "dec_loss": dec_loss,
            "hete_loss": hete_loss,
            "homo_loss": homo_loss,
            "s_l": s_l,
            "s_a": s_a,
            "s_v": s_v,
            "final_rep": final_rep,
        }
