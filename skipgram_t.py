#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事件序列 Skip-gram 嵌入  --  时间窗口版
python event2vec_time_window.py  --dataset <name>  --output_dir ./  --embed_dim 64  --window 5
"""
import argparse, random, os, time, pickle, bisect
from collections import Counter
from pathlib import Path
from itertools import accumulate

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------- 1. 参数 ----------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dataset', type=str, required=True)
    p.add_argument('--output_dir', default='./embed/skipgram_t', help='输出文件夹')
    p.add_argument('--embed_dim', type=int, default=64)
    p.add_argument('--window', type=int, default=5,
                   help='原固定窗口半径，仅用于校准目标')
    p.add_argument('--time_window', type=float, default=None,
                   help='时间窗口长度(秒)。留空则自动校准到与固定窗口平均上下文数一致')
    p.add_argument('--window_side', type=str, default='both', help='上下文窗口策略')
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
with open(f"/data/run01/scxi244/EVENT/eventsfm/data/Taxipro/Manhattan_2000/Events/train.pkl", 'rb') as f:
    data = pickle.load(f)
split = 'train'
seqs = data[split]          # 每条是 List[Dict]，含 "type_event" 与 "time_since_start"
type_seqs = [[x["type_event"] for x in s] for s in seqs]
time_delta_seqs = [[x["time_since_last_event"] for x in s] for s in seqs]
time_seqs = [list(accumulate(seq)) for seq in time_delta_seqs]

all_events = [e for seq in type_seqs for e in seq]
n_types = len(set(all_events))
print(f'loaded {args.dataset} dataset, {len(type_seqs)} sequences, {n_types} unique events')

# ---------- 3. 自动校准 time_window ----------
def count_avg_neighbors(time_seqs, delta):
    """给定时间窗口长度 delta，返回平均上下文事件数（含左右）"""
    total, cnt = 0, 0
    for tseq in time_seqs:
        times = np.array(tseq, dtype=np.float64)
        for i, t in enumerate(times):
            left  = t - delta
            right = t + delta
            # 二分找左右边界
            l_idx = bisect.bisect_left(times, left)
            r_idx = bisect.bisect_right(times, right)
            # 去掉自己
            n_nei = r_idx - l_idx - 1
            total += n_nei
            cnt   += 1
    return total / cnt

target = 2 * args.window          # 原固定窗口平均上下文数
if args.time_window is None:
    print('Calibrating time_window to match average neighbor count ~=', target)
    # 简单二分搜索区间
    low, high = 0.1, 86400*7      # 从 0.1 秒到 7 天，可按需调
    best = high
    for _ in range(25):
        mid = (low + high) / 2
        avg = count_avg_neighbors(time_seqs, mid)
        if abs(avg - target) < abs(count_avg_neighbors(time_seqs, best) - target):
            best = mid
        if avg < target:
            low = mid
        else:
            high = mid
    args.time_window = best
    print(f'Auto-selected time_window = {args.time_window:.2f} sec, '
          f'avg_neighbors = {count_avg_neighbors(time_seqs, args.time_window):.2f}')

# ---------- 4. Dataset ----------
class EventTimeWindowDataset(Dataset):
    def __init__(self, type_seqs, time_seqs, time_window, neg, n_types, window_side='both'):
        assert window_side in {'left', 'right', 'both'}
        self.n_types = n_types
        self.neg = neg
        self.window_side = window_side
        pairs = []

        for tseq, tpseq in zip(time_seqs, type_seqs):
            times = np.array(tseq, dtype=np.float64)
            for i, (t, c) in enumerate(zip(times, tpseq)):
                if window_side == 'left':
                    left  = t - time_window
                    right = t
                elif window_side == 'right':
                    left  = t
                    right = t + time_window
                else:  # 'both'
                    left  = t - time_window
                    right = t + time_window

                l_idx = bisect.bisect_left(times, left)
                r_idx = bisect.bisect_right(times, right)
                for j in range(l_idx, r_idx):
                    if i == j:
                        continue
                    pairs.append((c, tpseq[j]))

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        c, p = self.pairs[idx]
        neg = torch.randint(0, self.n_types, (self.neg,))
        neg = torch.where(neg == c, (neg + 1) % self.n_types, neg)
        return torch.tensor(c), torch.tensor(p), neg

ds = EventTimeWindowDataset(type_seqs, time_seqs, args.time_window, args.neg, n_types, args.window_side)
dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                num_workers=4, pin_memory=True)

# ---------- 5. Model ----------
class SkipEvent(nn.Module):
    def __init__(self, n_types, emb_dim):
        super().__init__()
        self.c_emb  = nn.Embedding(n_types, emb_dim)
        self.ctx_emb = nn.Embedding(n_types, emb_dim)
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

# ---------- 6. Train ----------
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

# ---------- 7. 验证：embed 两两余弦相似度 ----------
@torch.no_grad()
def cosine_sim_stats(emb: torch.Tensor):
    emb = emb / emb.norm(dim=1, keepdim=True)
    sim = emb @ emb.t()
    mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    sim_vec = sim.masked_select(mask)
    print('Cosine similarity  ----  '
          f'mean={sim_vec.mean().item():.4f}, '
          f'max={sim_vec.max().item():.4f}, '
          f'min={sim_vec.min().item():.4f}, '
          f'std={sim_vec.std().item():.4f}')

embed_matrix = model.c_emb.weight.data
cosine_sim_stats(embed_matrix)

# ---------- 8. 保存 ----------
output_path = Path(args.output_dir)
output_path.mkdir(exist_ok=True, parents=True)
save_file = output_path / f'{args.dataset}_timeWindow{args.time_window:.2f}_embedding.pt'
torch.save(embed_matrix.cpu(), save_file)
print(f'saved -> {save_file}')