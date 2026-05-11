# Type embeddings

## Skip-gram（原有流程）

对每个数据集跑 `skipgram.py`，再用 `merge_embedding.py` 合并为 `./embed/type_embeddings.pt`，训练时由 `Runner` 读取（见 `config/model.yaml` 的 `type_embeddings_path`）。可参考 `dataembed.sh`。

## Hybrid（skip-gram + `type_descriptions`）

脚本：[build_hybrid_type_embeddings.py](build_hybrid_type_embeddings.py)

需要本机可加载的 HuggingFace 文本编码器（默认 `./bert`，与 `model.py` 中 BERT 路径一致）。

### 生成融合向量

在项目根目录执行（按你实际路径改 `--skipgram_path` / `--text_encoder`）：

```bash
python build_hybrid_type_embeddings.py \
  --dataset_config ./config/dataset.yaml \
  --skipgram_path ./embed/type_embeddings.pt \
  --output_dir ./embed/hybrid_desc \
  --text_encoder ./bert \
  --datasets SCEDC XTraffic Danmaku weather \
  --desc_weight 1.0
```

- **`--datasets`**：这些名字必须同时出现在 `type_embeddings.pt` 与 `dataset.yaml` 中，且 yaml 里 `type_descriptions` 覆盖 `0 .. num_event_types-1`；否则默认报错退出。
- **省略 `--datasets`**：对 skip-gram 文件里**每个** key，若 yaml 有完整 `type_descriptions` 则融合，否则该数据集右侧描述块填 **0**（输出宽度仍为 `skip_dim + bert_hidden`，与训练侧一致）。
- **`--missing_policy skip`**：当指定了 `--datasets` 但某条 yaml 缺描述时，不退出，对该数据集用 **0** 描述块。

可选：`--device cuda:0`、`--encode_batch_size 16` 等。

### 训练时指定融合 embedding

```bash
python main.py -d SCEDC XTraffic Danmaku weather --type_embeddings_path ./embed/hybrid_desc
```

`Runner` 会加载 `{type_embeddings_path}/type_embeddings.pt`。融合后第二维为 **skip 维 + BERT hidden**，与纯 skip-gram 的 64 维不同，需重新训练或从头适配 checkpoint。

### 说明

- 融合方式：对 skip-gram 行与描述向量分别 **L2 normalize**，描述再乘 `--desc_weight`，再 **concat**。
- 若开启多模态融合（`use_text` / `use_image`），`embed_dim` 需能被 `num_heads=4` 整除；脚本在 `fused_dim % 4 != 0` 时会打印警告。
