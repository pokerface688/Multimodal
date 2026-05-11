#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事件序列 Skip-gram 嵌入  --  single file version
python event2vec.py  --dataset <name>  --output_dir ./  --embed_dim 64
"""
import argparse, random, os, time, pickle
from collections import Counter
from itertools import accumulate
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------- 1. 参数 ----------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dataset', type=str, required=True)
    p.add_argument('--output_dir', default='./embed', help='输出文件夹')
    p.add_argument('--embed_dim', type=int, default=64)
    p.add_argument('--window', type=int, default=5, help='skip-gram 窗口半径')
    p.add_argument('--neg', type=int, default=5, help='负采样个数')
    p.add_argument('--batch_size', type=int, default=2048)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

args = get_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cuda', args.device)

# ---------- 2. 读数据 ----------
with open(f"/data/run01/scxi244/EVENT/eventsfm/data/MTBENCH/Events/train.pkl", 'rb') as f:
    data = pickle.load(f)
split = 'train'          # 若需其它 split 请自行调整
type_seqs = [[x["type_event"] for x in seq] for seq in data[split]]

# ---- 计算 n_types ----
all_events = [e for seq in type_seqs for e in seq]
n_types = len(set(all_events))
print(f'loaded {args.dataset} dataset, {len(type_seqs)} sequences, {n_types} unique events')

# ---------- 3. Dataset ----------
class EventSkipDataset(Dataset):
    def __init__(self, type_seqs, window, neg, n_types):
        self.n_types = n_types
        self.neg = neg
        pairs = []
        for s in type_seqs:
            for i, c in enumerate(s):
                left  = max(0, i - window)
                right = min(len(s), i + window + 1)
                for j in range(left, right):
                    if i == j:
                        continue
                    pairs.append((c, s[j]))
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        c, p = self.pairs[idx]
        neg = torch.randint(0, self.n_types, (self.neg,))
        neg = torch.where(neg == c, (neg + 1) % self.n_types, neg)
        return torch.tensor(c), torch.tensor(p), neg

ds = EventSkipDataset(type_seqs, args.window, args.neg, n_types)
dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                num_workers=4, pin_memory=True)

# ---------- 4. Model ----------
class SkipEvent(nn.Module):
    def __init__(self, n_types, emb_dim):
        super().__init__()
        self.c_emb  = nn.Embedding(n_types, emb_dim)
        self.ctx_emb = nn.Embedding(n_types, emb_dim)
        # 初始化小一点
        self.c_emb.weight.data.uniform_(-0.5/emb_dim, 0.5/emb_dim)
        self.ctx_emb.weight.data.uniform_(-0.5/emb_dim, 0.5/emb_dim)

    def forward(self, c, p, n):
        c_vec = self.c_emb(c)                              # [B, emb]
        p_vec = self.ctx_emb(p)                            # [B, emb]
        n_vec = self.ctx_emb(n)                            # [B, neg, emb]
        pos_score = (c_vec * p_vec).sum(-1)                # [B]
        neg_score = torch.bmm(n_vec, c_vec.unsqueeze(-1)).squeeze(-1)  # [B, neg]
        loss = - (torch.log(torch.sigmoid(pos_score)).mean()
                  + torch.log(torch.sigmoid(-neg_score)).mean())
        return loss

model = SkipEvent(n_types, args.embed_dim).to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)

# ---------- 5. Train ----------
model.train()
for epoch in range(1, args.epochs+1):
    total_loss = 0
    bar = tqdm(dl, desc=f'Epoch {epoch}')
    for c, p, n in bar:
        c, p, n = c.to(device), p.to(device), n.to(device)
        opt.zero_grad()
        loss = model(c, p, n)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        bar.set_postfix(loss=f'{loss.item():.4f}')
    print(f'Epoch {epoch} done, avg_loss={total_loss/len(dl):.4f}')

# ---------- 6. 验证：embed 两两余弦相似度 ----------
@torch.no_grad()
def cosine_sim_stats(emb: torch.Tensor):
    """emb: [N, dim] -> 打印相似度统计"""
    emb = emb / emb.norm(dim=1, keepdim=True)   # 单位化
    sim = emb @ emb.t()                          # [N, N]
    # 去掉对角线
    mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    sim_vec = sim.masked_select(mask)
    print('Cosine similarity  ----  '
          f'mean={sim_vec.mean().item():.4f}, '
          f'max={sim_vec.max().item():.4f}, '
          f'min={sim_vec.min().item():.4f}, '
          f'std={sim_vec.std().item():.4f}')

embed_matrix = model.c_emb.weight.data
cosine_sim_stats(embed_matrix)      # 打印验证结果

# ---------- 7. 保存 ----------
output_path = Path(args.output_dir)
output_path.mkdir(exist_ok=True, parents=True)
save_file = output_path / f'{args.dataset}_embedding.pt'
torch.save(embed_matrix.cpu(), save_file)   # 仅保存 embed 向量
print(f'saved -> {save_file}')