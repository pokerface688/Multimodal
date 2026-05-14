# DecAlign 代码实现说明

本文档描述本仓库内 **`DecAlign/`** 子目录中的实现：多模态情感分析（MOSI / MOSEI 回归，IEMOCAP 分类），对应论文 *DecAlign: Hierarchical Cross-Modal Alignment for Decoupled Multimodal Representation Learning*（ICLR 2026，[arXiv:2503.11892](https://arxiv.org/abs/2503.11892)）。

> **注意**：根目录 `.gitignore` 使用 `*.md` 忽略一般 Markdown，但对 **`docs/`** 下文件配置了 **`!docs/**/*.md`**，因此 `docs/` 内文档可正常纳入 Git。更细的架构与对齐图解见 [DecAlign-architecture-detail.md](./DecAlign-architecture-detail.md)。**事件序列 `config/model.yaml` 中 DecAlign 超参如何选取** 见 [DecAlign-hyperparameters.md](./DecAlign-hyperparameters.md)。

---

## 1. 目录与职责

| 路径 | 说明 |
|------|------|
| `DecAlign/main.py` | CLI 入口、`DMD_run` / `_run`、多 seed 结果写 CSV |
| `DecAlign/config.py` | 默认配置 + 数据集覆盖 + 读取 `config/dec_config.json` → `SimpleNamespace` |
| `DecAlign/config/dec_config.json` | 实验超参（会覆盖 `config.py` 中同名键） |
| `DecAlign/data_loader.py` | `MMDataset`、`MMDataLoader`，pickle 特征与 batch |
| `DecAlign/models/model.py` | **`DecAlign`** 模型：解耦、异质 OT、同质对齐、融合、输出 |
| `DecAlign/trains/ATIO.py` | **`DecAlignTrainer`**：训练 / 验证 / 测试循环与损失组合 |
| `DecAlign/trains/subNets/BertTextEncoder.py` | BERT 文本编码（MMSA 三通道整型或浮点特征） |
| `DecAlign/trains/subNets/transformer.py` | 局部 Transformer 编码器 |
| `DecAlign/utils/functions.py` | 随机种子、GPU 分配等 |
| `DecAlign/utils/metrices.py` | 各数据集评测指标封装 |
| `DecAlign/run.py` / `eval.py` | 脚本式启动训练 / 测试（硬编码参数为主） |

---

## 2. 程序入口与运行流程

### 2.1 命令行（`main.py`）

`parse_args()` 主要参数：

- `--model`：默认 `decalign`
- `--dataset`：`mosi` \| `mosei` \| `iemocap`
- `--data_dir`、`--model_save_dir`、`--res_save_dir`、`--log_dir`
- `--mode`：`train` \| `test`
- `--seeds`：一个或多个随机种子（默认 `[1111]`）
- `--gpu_ids`、`--num_workers`

`main()` 调用 `DMD_run(...)`。

### 2.2 `DMD_run`（`main.py`）

1. 将 `model_name`、`dataset_name` 转小写。
2. 加载配置文件：默认同目录下 **`config/dec_config.json`**（可通过 `config_file` 覆盖）。
3. 若未指定目录，模型 / 结果 / 日志默认写到用户主目录下 **`~/MMSA/`** 各子路径；当前 CLI 一般会传入 `./pt`、`./result`、`./log`。
4. 调用 `get_config_regression(model_name, dataset_name, config_file, data_dir)` 得到 **`args`**（`SimpleNamespace`）。
5. 写入运行期字段：`args.mode`、`args.model_save_path`、`args.device`（`assign_gpu`）、可选 `feature_T/A/V` 覆盖路径。
6. 对每个 `seed`：`setup_seed` → `_run(args)` → 收集指标字典。
7. 将多 seed 结果汇总为均值 / 标准差，追加写入 **`res_save_dir/normal/{dataset}.csv`**。

### 2.3 `_run`（`main.py`）

```text
MMDataLoader(args) → DecAlign(args).cuda() → DecAlignTrainer(args)
```

- **`mode == 'test'`**：`load_state_dict` 已保存权重 → `trainer.do_test(test)`。
- **否则**：`trainer.do_train` → 再加载最佳 `model_save_path` → `do_test(test)`，释放显存。

---

## 3. 配置（`config.py` + `dec_config.json`）

### 3.1 合并顺序

1. `get_config_regression` 内建 **`default_config`**（含通用模型字段、训练超参占位）。
2. 按 **`dataset_name`** 写入 **`dataset_config`**（`featurePath`、`seq_lens`、`feature_dims`、`train_mode` 等）。
3. `default_config.update(dataset_config)`。
4. 若 **`config_file`** 存在：JSON 读入后 **`update`** 到同一字典（**`dec_config.json` 优先级最高**）。
5. `dict_to_namespace` 转为 **`SimpleNamespace`**，全工程用 **`args.xxx`** 访问。

### 3.2 数据集默认路径（相对 `data_dir`）

- **MOSI**：`MOSI/mosi_aligned_50.pkl`，`feature_dims` 示例 `[768, 5, 20]`，`train_mode`: `regression`。
- **MOSEI**：`MOSEI/mosei_aligned_50.pkl`，`feature_dims` `[768, 74, 35]`，`regression`。
- **IEMOCAP**：`IEMOCAP/iemocap_data.pkl`，`feature_dims` `[768, 74, 35]`，`train_mode`: **`classification`**。

### 3.3 `dec_config.json` 与代码的对应关系

| JSON 字段 | 代码中的典型用途 |
|-----------|------------------|
| `use_bert` / `use_finetune` / `transformers` / `pretrained` | `BertTextEncoder` |
| `need_data_aligned` | 数据集中是否构造 `audio_lengths` / `vision_lengths`；**`models/model.py`** 中各模态时间长度分支 |
| `dst_feature_dim_nheads` | `[d, num_heads]` → 统一投影维 **`d_l = d_a = d_v = d`**，`TransformerEncoder` 头数 |
| `nlevels` | Transformer 层数 |
| `conv1d_kernel_size_l/a/v` | 各模态 `Conv1d` 投影核大小 |
| `num_prototypes` | 原型个数 **K**（`proto_*` 形状 `[K, d]`） |
| `lambda_ot` | `model.py` 中 **`self.ot_reg`**，Sinkhorn 熵正则尺度 |
| `ot_num_iters` | **`multi_marginal_sinkhorn`** 迭代次数 |
| `alpha1` | **解耦损失** `dec_loss` 权重 |
| `alpha2` | **异质 + 同质** `(hete_loss + homo_loss)` 权重 |
| `batch_size` / `learning_rate` / `weight_decay` / `num_epochs` / `patience` / `clip` / `factor` | `DecAlignTrainer` 与优化器 |

当前仓库中的 **`dec_config.json`** 示例将 **`need_data_aligned`** 设为 **`true`**，与 `config.py` 里 MOSI 默认 `seq_lens [50,50,50]` 等一致时，模型内文本 / 音频 / 视频序列长度取 **对齐后的 50**。

---

## 4. 数据管道（`data_loader.py`）

### 4.1 `MMDataset`

- 按 **`args.dataset_name`** 分派到 `__init_mosi`（MOSEI 复用）、`__init_iemocap`。
- 从 **`args.featurePath`** 读取 **pickle**。
- **文本**：`use_bert=True` 时优先使用 **`text_bert`**；若为 **`[N, 3, L]`** 且为整型，则保持 **MMSA BERT 输入格式**（`input_ids`、`token_type_ids`、`attention_mask`）；否则为浮点特征。
- **音频 / 视频**：`audio`、`vision` 转为 `float32`；可选 **`feature_T` / `feature_A` / `feature_V`** 覆盖子特征并回写 **`args.feature_dims`**。
- **标签**：MOSI/MOSEI 用 **`regression_labels`** → `labels['M']`；IEMOCAP 用 **`classification_labels`**。
- **`need_data_aligned == False`** 时，从数据或 side pickle 读取 **`audio_lengths` / `vision_lengths`**（对齐数据无长度列表时用满长填充）。

### 4.2 `MMDataLoader`

构建 `train` / `valid` / `test` 的 **`DataLoader`**（batch 内含 `text`、`audio`、`vision`、`labels['M']`，以及按配置附加的长度字段），并把 **`args.seq_lens`**、**`args.feature_dims`** 等与模型一致的字段写回 **`args`**。

---

## 5. 模型 `DecAlign`（`models/model.py`）

以下按 **前向逻辑** 组织；张量形状以 batch **`B`**、投影维 **`d`**（如 40）、时间长度 **`T_l, T_a, T_v`** 为例，实际长度由数据集与 `need_data_aligned` 决定。

### 5.1 文本与模态形状

- 若 **`use_bert`**：`BertTextEncoder` 输出 **`[B, T_l, 768]`**（或已是 `[B, T_l, D]` 的浮点文本）。
- 转 **`[B, D, T]``（channel-first）**：`text.transpose(1,2)`，`audio`、`video` 同理。

### 5.2 初始投影

- **`proj_l` / `proj_a` / `proj_v`**：`Conv1d(orig_d_*, d, kernel_size=k_*)`，得到 **`proj_x_*`**，形状 **`[B, d, T_*']`**（`k=1` 时 `T_*' = T_*`）。

### 5.3 解耦（Decoupling）

- **`encoder_uni_*`**：`Conv1d(d, d, 1)` → **`s_l, s_a, s_v`**（模态特有 / 异质分支的输入）。
- **`encoder_com`**：共享 **`Conv1d(d, d, 1)`** 分别作用于三模态投影 → **`c_l, c_a, c_v`**（模态共有 / 同质分支）。

**`dec_loss`**（`compute_decoupling_loss`）：对每个模态，将 **`s`** 与 **`c`** 展平后算 **余弦相似度**，训练目标为 **最小化其均值**（与「互补 / 公共」分离一致）。

### 5.4 异质对齐（Heterogeneity）

**`compute_hetero_loss(s_l, s_a, s_v)`**：

1. **`compute_prototypes`**：对 **`s_*`** 在时间维均值得到 **`[B, d]`**，与可学习 **`proto_*`**（`[K, d]`）算距离，softmax 得软分配 **`w_*`**，形状 **`[B, K]`**。
2. 对 batch 平均并归一化得边际 **`nu_l, nu_a, nu_v`**（各 **`[K]`**）。
3. **`pairwise_cost`**：两两模态原型间 **欧氏项 + 对角协方差匹配项**（由 **`logvar_*`** 得到方差）。
4. 构造 **三阶联合代价张量** **`C[i,j,k]`**（三对 pairwise 之和）。
5. **`multi_marginal_sinkhorn`**：在 **`C`** 上迭代缩放 **`u, v, w`**，得联合传输 **`T`**，**`ot_loss = sum(T*C) + 小系数 * 熵项`**。
6. **局部原型损失**：各模态样本特征与其它模态 **`proto_*`** 的加权平方距离之和。

**`hete_loss = ot_loss + local_proto_loss`**。

超参：**`num_prototypes` → K**，**`lambda_ot` → `ot_reg`**，**`ot_num_iters`**。

### 5.5 同质对齐（Homogeneity）

**`compute_homo_loss(c_l, c_a, c_v)`**：

- 对每个 **`c_*`** 在时间维与 batch 维上统计 **均值、方差、偏度**，三模态两两对齐（**`L_sem`**）。
- 对时间平均后的 **`c_*`** 两两算 **RBF 核 MMD**（**`L_mmd`**）。

**`homo_loss = L_sem + L_mmd`**。

### 5.6 融合与输出

**异质分支 A — Transformer 时序融合**

- **`s_*`** 转为 **`[T, B, d]`**，截断到 **`T_target = min(T_l, T_a, T_v)`**。
- 拼接 **`[T_target, B, 3d]`** → **`transformer_fusion`**（`embed_dim=3d`）→ 取最后时间步 **`fusion_rep_trans`**，形状 **`[B, 3d]`**。

**异质分支 B — 跨模态注意力**

- 六组 **`trans_*_with_*`**（每模态 query，另两模态 key/value）+ **`trans_*_mem`** 处理拼接记忆。
- 取各模态最后表示拼 **`[B, 6d]`** → **`cma_proj`** → **`fusion_rep_cma`**，**`[B, 3d]`**。

**同质分支**

- **`c_*`** 时间维 **`mean`** → 拼 **`fusion_rep_homo`**，**`[B, 3d]`**。

**汇总**

- **`fusion_rep_hete = fusion_rep_trans + fusion_rep_cma`**。
- **`final_rep = cat(fusion_rep_hete, fusion_rep_homo, dim=1)`** → **`[B, 6d]`**。
- **`out_layer`**：`Linear(6d → 1)`（MOSI/MOSEI 回归）或 **`→ 6`**（IEMOCAP 多类）。

### 5.7 `forward` 返回值

字典中包含但不限于：

- **`output_logit`**：主任务预测。
- **`dec_loss` / `hete_loss` / `homo_loss`**：辅助损失标量。
- **`s_*` / `c_*`**、各融合中间量：便于调试或可视化。

---

## 6. 训练与评测（`trains/ATIO.py`）

### 6.1 `DecAlignTrainer`

- **`criterion`**：`train_mode == 'regression'` 时为 **`nn.L1Loss()`**，否则 **`nn.CrossEntropyLoss()`**。
- **`metrics`**：`MetricsTop(args.train_mode).getMetrics(dataset_name)`。

### 6.2 `do_train`

- 优化器：**`Adam`**，学习率 **`args.learning_rate`**，**`weight_decay`** 来自配置。
- 调度器：**`ReduceLROnPlateau`**，`mode='min'`（回归）或 **`max`**（分类），`patience` / **`factor`** 来自配置。
- 每个 batch：

```python
outputs = model(text, audio, video)
logits = outputs['output_logit']
main_loss = criterion(logits.view(-1), labels.view(-1))
loss = main_loss + args.alpha1 * dec_loss + args.alpha2 * (hete_loss + homo_loss)
```

- **`clip_grad_norm_`**：若存在 **`args.clip`** 则裁剪。
- 验证：**`do_test` on valid**；以验证集 **`eval_results[dataset_name.upper()]`**（此处为 **验证损失**）作为 plateau 与 **early stopping** 信号；最优 **`state_dict`** 写入 **`args.model_save_path`**。

### 6.3 `do_test`

- 仅 **`main_loss`**（**不包含** `dec_loss` / `hete_loss` / `homo_loss`）累计平均，并写入返回字典中数据集大写键对应的 **loss 字段**；同时 **`metrics(y_pred, y_true)`** 计算相关性、准确率等（依数据集实现而定）。

### 6.4 实现注意（IEMOCAP）

分类路径中 **`logits.view(-1)`** 与 **`CrossEntropyLoss`** 的维度约定需与 **`out_layer`** 输出形状 **`[B, C]`** 一致；若 logits 实际为 **`[B, 1, C]`** 等，应核对避免 silent error。建议在接入 IEMOCAP 时做一次 **单元测试或单 batch 打印 shape**。

---

## 7. 辅助脚本

- **`scripts/run_decalign.sh`**：调用根目录 **`main.py`**，传入数据集、数据目录、保存路径等。
- **`run.py` / `eval.py`**：快速跑 MOSI 训练 / 测试，路径需与本地数据布局一致。

---

## 8. 与论文模块的对照（便于读论文时查代码）

| 论文概念 | 代码位置（约） |
|----------|----------------|
| 模态特有 / 共有分解 | `encoder_uni_*`, `encoder_com`, `dec_loss` |
| 原型 + 多边际最优传输 | `proto_*`, `logvar_*`, `multi_marginal_sinkhorn`, `compute_hetero_loss` |
| 同质矩 + MMD | `compute_homo_loss`, `compute_mmd` |
| 层次异质融合（Transformer + 跨模态） | `transformer_fusion`, `trans_*_with_*`, `trans_*_mem`, `cma_proj` |
| 同质拼接 | `fusion_rep_homo`, `final_rep` |

---

## 9. 参考文献与上游代码

- 论文：[arXiv:2503.11892](https://arxiv.org/abs/2503.11892)  
- 官方实现参考：[https://github.com/taco-group/DecAlign](https://github.com/taco-group/DecAlign)  

本文档仅描述 **当前 `DecAlign/` 目录内代码行为**；若上游更新，请以实际文件为准。
