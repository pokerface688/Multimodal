import pickle, os, h5py
import numpy as np
import random
from typing import List, Dict, Optional
from itertools import accumulate

import torch
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset, Subset
from transformers import PreTrainedTokenizer


class BaseDataset(Dataset):
    def __init__(self, data: Dict, dataset_id: str):
        self.time_seqs = data['time_seqs']
        self.time_delta_seqs = data['time_delta_seqs']
        self.type_seqs = data['type_seqs']
        self.root_cause = data['root_cause']
        self.image_path_seqs = data.get('image_path_seqs', [[""] * len(seq) for seq in self.time_seqs])
        self.text_seqs = data.get('text_seqs', [[""] * len(seq) for seq in self.time_seqs])
        self.dataset_id = dataset_id

    def __len__(self):
        return len(self.time_seqs)

    def __getitem__(self, idx):
        return {
            'time_seqs': self.time_seqs[idx],
            'time_delta_seqs': self.time_delta_seqs[idx],
            'type_seqs': self.type_seqs[idx],
            'root_cause': self.root_cause[idx],
            'image_path_seqs': self.image_path_seqs[idx],
            'text_seqs': self.text_seqs[idx],
            'dataset_id': self.dataset_id
        }


class HDF5Dataset(Dataset):
    """HDF5 版本同理"""
    def __init__(self, hdf5_path: str, dataset_id: str):
        self.hdf5_path = hdf5_path
        self.dataset_id = dataset_id
        self._open_file()

    def _open_file(self):
        self.hfile = h5py.File(self.hdf5_path, "r", swmr=True)
        self.events_dset = self.hfile["events"]
        self.index_dset = self.hfile["indices"]

    def __len__(self):
        return len(self.index_dset)

    def __getitem__(self, idx):
        if not hasattr(self, "hfile"):
            self._open_file()
        start, length = self.index_dset[idx]
        events = self.events_dset[start:start+length]
        return {
            'time_seqs': events["time"].tolist(),
            'time_delta_seqs': events["delta_time"].tolist(),
            'type_seqs': events["event_id"].tolist(),
            'image_path_seqs': ["" for _ in range(length)],
            'text_seqs': ["" for _ in range(length)],
            'dataset_id': self.dataset_id
        }

    def __del__(self):
        if hasattr(self, "hfile"):
            self.hfile.close()


class MultiDataset(Dataset):
    """
    把多个子数据集 wrap 在一起，按权重做 dataset-level 采样。
    采样粒度：先选数据集，再在该数据集里随机抽一条。
    """
    def __init__(self, datasets: List[Dataset], weights: List[float]):
        self.datasets = datasets
        self.weights = torch.tensor(weights, dtype=torch.float)
        self.total_size = int(sum(len(d) * w for d, w in zip(datasets, weights)))

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        dataset_idx = torch.multinomial(self.weights, 1).item()
        dataset = self.datasets[dataset_idx]
        inner_idx = torch.randint(len(dataset), (1,)).item()
        return dataset[inner_idx]


# ------------------ 统一 collator ------------------
class DataCollatorEventEmbedding:
    """
    不再文本化，而是：
    1. 对每个事件：type_id → 查表得E_type，delta_t → Time2Vec得E_time
    2. token_i = E_type[i] + E_time[i]
    3. 右对齐预测：最后一个token只参与loss计算
    """
    def __init__(self, type_embeddings: Dict[str, torch.Tensor], max_event: int=128, use_prompt: bool=False):
        self.type_embeddings = type_embeddings  # Dict[dataset_name, tensor(K, d)]
        self.max_event = max_event
        self.use_prompt = use_prompt
        if use_prompt:
            if type_embeddings.get('prompt') is None:
                raise ValueError("use_prompt=True 时必须传入 prompt_emb")
            self.prompt_emb = type_embeddings.get('prompt')                # [Lp, D]
            self.prompt_len = self.prompt_emb.shape[0]
        else:
            self.prompt_emb = None
            self.prompt_len = 0

    def __call__(self, raw_batch: List[Dict]):
        # 1. 统计当前 batch 最大事件条数
        max_evt = max(len(item['time_seqs']) for item in raw_batch)
        max_evt = min(max_evt, self.max_event)
        total_len = self.prompt_len + max_evt
        

        # 2. 准备数据：右对齐，前面padding
        batch_size = len(raw_batch)
        hidden_size = list(self.type_embeddings.values())[0].shape[1]
        
        # 构建嵌入序列 [batch, seq_len, hidden_size]
        event_embeddings = torch.zeros(batch_size, total_len, hidden_size, dtype=torch.bfloat16)
        time_seqs = torch.zeros(batch_size, total_len, dtype=torch.float32)
        time_delta_seqs = torch.zeros(batch_size, total_len, dtype=torch.float32)
        type_seqs = torch.zeros(batch_size, total_len, dtype=torch.long)
        batch_non_pad_mask = torch.zeros(batch_size, total_len, dtype=torch.bool)
        attention_mask = torch.zeros(batch_size, total_len, dtype=torch.bool)
        root_cause = torch.zeros(batch_size, dtype=torch.long)
        batch_image_paths = []
        batch_texts = []
        
        for i, item in enumerate(raw_batch):
            t, d, k = item['time_seqs'], item['time_delta_seqs'], item['type_seqs']
            img_seq = item['image_path_seqs']
            txt_seq = item['text_seqs']
            L = len(t)
            rc = item['root_cause']
            dataset_id = item['dataset_id']
            
            # 右对齐：取最后max_evt个事件
            if L > max_evt:
                t, d, k = t[-max_evt:], d[-max_evt:], k[-max_evt:]
                L = max_evt
                img_seq = img_seq[-max_evt:]
                txt_seq = txt_seq[-max_evt:]
            # 计算padding长度（前面pad）
            pad_len = max_evt - L
            
            # 获取嵌入 (K, d) → (L, d)
            type_emb = self.type_embeddings[dataset_id][k]  # shape: [L, hidden_size]
            
            # 时间编码将在模型中动态计算，这里只保存原始值
            # 每个序列包括三部分:  [左填充pad_len + prompt长度prompt_len + 事件序列L]
            event_embeddings[i, -L:, :] = type_emb
            time_seqs[i, -L:] = torch.tensor(t)
            time_delta_seqs[i, -L:] = torch.tensor(d)
            type_seqs[i, -L:] = torch.tensor(k)
            batch_non_pad_mask[i, -L:] = True
            attention_mask[i, pad_len:] = True
            root_cause[i] = rc
            batch_image_paths.append([''] * pad_len + img_seq)
            batch_texts.append([''] * pad_len + txt_seq)
        return {
            'event_embeddings': event_embeddings,              # [B, S, D]
            'time_seqs': time_seqs,                            # [B, S]
            'time_delta_seqs': time_delta_seqs,                # [B, S]
            'type_seqs': type_seqs,                            # [B, S]
            'batch_non_pad_mask': batch_non_pad_mask,          # [B, S]
            'attention_mask': attention_mask,                  # [B, S]
            'image_paths': batch_image_paths,
            'texts': batch_texts,
            'root_cause': root_cause,                  # [B]
            'dataset_id': [item['dataset_id'] for item in raw_batch]

        }


# ------------------ 对外唯一入口 ------------------
def build_multiloader(data_config: Dict, split: str, batch_size: int, train_subset_ratio=1.0,  
                      num_workers=0, type_embeddings_path: str = None, use_prompt: bool=False, use_root_cause=False):
    """
    data_config 格式:
        retweet:
          data_path: ...
          data_format: pkl
          weight: 0.7
          prompt: "..."
          num_event_types: 3
    split in ['train','dev','test']
    """
    # 加载类型嵌入
    if type_embeddings_path:
        type_embeddings = torch.load(type_embeddings_path, weights_only=True)
        type_embeddings = {k: v.to(torch.bfloat16) for k, v in type_embeddings.items()}
    else:
        type_embeddings = None

    sub_datasets, weights = [], []
    for name, spec in data_config.items():
        # 1. 构造子数据集
        path = spec['data_path'].replace('train', split)
        if spec['data_format'] == 'pkl':
            with open(path, 'rb') as f:
                data = pickle.load(f)
            # 原始事件序列解析
            type_seqs = []
            time_delta_seqs = []
            image_path_seqs = []
            text_seqs = []

            # 逐样本、逐事件读取：严格同索引绑定
            for seq in data[split]:
                typ = [x["type_event"] for x in seq]
                dlt = [x["time_since_last_event"] for x in seq]
                # 每个事件对应自己的path和text，旧数据无则为空字符串
                img = [x.get("image_path", "") for x in seq]
                txt = [x.get("text", "") for x in seq]

                type_seqs.append(typ)
                time_delta_seqs.append(dlt)
                image_path_seqs.append(img)
                text_seqs.append(txt)
            time_seqs = [list(accumulate(seq)) for seq in time_delta_seqs]    
            if use_root_cause:
                root_cause = data['labels']
            else:
                root_cause = [-1 for seq in data[split]]
            data = {'time_seqs': time_seqs, 
                    'time_delta_seqs': time_delta_seqs, 
                    'type_seqs': type_seqs,'image_path_seqs': image_path_seqs, 'text_seqs': text_seqs,
                    'root_cause': root_cause}
            ds = BaseDataset(data, dataset_id=name)
        elif spec['data_format'] == 'h5':
            ds = HDF5Dataset(path, dataset_id=name)
        else:
            raise ValueError
        sub_datasets.append(ds)
        weights.append(spec['weight'] if split == 'train' else 1.0)  # 验证集均匀采样

    if split == 'train' and 0 < train_subset_ratio < 1:
        sub_datasets = [
            Subset(ds, random.sample(range(len(ds)), int(len(ds) * train_subset_ratio)))
            for ds in sub_datasets
        ]

    if split == 'train':
        # 训练集：权重采样
        multi_ds = MultiDataset(sub_datasets, weights)
    else:
        # 验证/测试：简单 Concat
        multi_ds = ConcatDataset(sub_datasets)

    collator = DataCollatorEventEmbedding(type_embeddings, use_prompt=use_prompt)
    loader = DataLoader(multi_ds,
                        batch_size=batch_size,
                        shuffle=(split=='train'),
                        collate_fn=collator,
                        num_workers=num_workers,
                        pin_memory=True)
    return loader