# DecAlign 模型架构与对齐实现细节

本文汇总 **`DecAlign/models/model.py`** 中的**解耦、异质对齐、同质对齐、融合与输出**的实现细节，并附 **Mermaid** 架构图，便于对照代码与论文。

**相关文档**：[DecAlign-implementation.md](./DecAlign-implementation.md)（入口、配置、数据与训练循环）。

---

## 1. 解耦：同质 / 异质分支在代码里是什么

**不是**「编码后各自多层 MLP」，而是：三模态先 **`Conv1d` 投影**到统一维 **`d`**，再：

| 分支 | 模块 | 参数是否跨模态共享 | 数学含义 |
|------|------|-------------------|----------|
| **异质 `s_l,s_a,s_v`** | `encoder_uni_l` / `encoder_uni_a` / `encoder_uni_v` | **否**（三套 `Conv1d(d,d,k=1)`） | 每个模态独有的逐点线性变换 |
| **同质 `c_l,c_a,c_v`** | **`encoder_com`（同一模块）** | **是** | 同一组权重分别作用在 `proj_x_l/proj_x_a/proj_x_v` 上 |

`kernel_size=1` 的 `Conv1d` 在 **`[N,d,T]`** 上**不混合时间步**，只在通道维做 **`d→d`** 线性映射；等价于对每个时间步做一次 **共享权重的 `Linear(d,d)`**，与「多层 MLP（含非线性）」不同：**此处无激活，单层线性**。

**解耦损失 `dec_loss`**：`compute_decoupling_loss` 将 **`s`** 与 **`c`** 展平后算 **余弦相似度**，训练时**最小化**三模态该量的和，鼓励 **unique 与 common 分离**。

---

## 2. 卷积层与「MLP」的区分（实现语境）

- **单层 `Conv1d` / `Linear`（无激活）**：对输入是**（仿射）线性**变换。
- **MLP**：通常指 **`Linear → ReLU/GELU → Linear → …`**，**非线性来自激活函数**，不是来自「叫 MLP」本身。
- **本模型中的 1×1 卷积**：属于**逐点线性**，便于在 **`[N,d,T]`** 上写；与一层无激活 `Linear` 同类。

---

## 3. 异质对齐实现（`compute_hetero_loss`，作用在 **`s_*`**）

**目标**：在**可学习原型**空间中，让三模态「独有」表示**可运输地对齐**，并加**样本—异模态原型**局部项。

### 3.1 步骤摘要

1. **软分配 `w_* ∈ ℝ^{N×K}`**  
   对 **`s_*`** 时间维 **`mean` → `[N,d]`**，与 **`proto_* ∈ ℝ^{K×d}`** 算平方距离，`softmax(-dist)` 得每个样本对 \(K\) 个原型的权重。  
   （`compute_prototypes`；`logvar_*` 参与下面代价，不直接进 softmax。）

2. **边际 `ν_l, ν_a, ν_v ∈ ℝ^K`**  
   对 batch 在 \(N\) 上 **`w_*`** 取平均再归一化（和为 1）。

3. **两两原型代价 `cost_* ∈ ℝ^{K×K}`**  
   **`pairwise_cost`**：原型均值 **欧氏平方** + **对角方差项**（\(\sigma=\exp(\logvar)\)）。

4. **三阶联合代价 `C ∈ ℝ^{K×K×K}`**  
   \(C[i,j,k] = cost_{la}[i,j] + cost_{lv}[i,k] + cost_{av}[j,k]\)。

5. **多边际 Sinkhorn**  
   `K_tensor = exp(-C/reg)`，交替更新 **`u,v,w`**，使联合 **`T ∝ u⊗v⊗w ⊙ K_tensor`** 与 **`ν_l,ν_a,ν_v`** 一致；  
   **`ot_loss = Σ T·C + 0.001·reg·entropy(T)`**。

6. **局部原型损失**  
   各模态 **`feat_* = s_*`** 时间均值，与**另一模态**的 **`proto`** 加权平方距离（权重为本模态 **`w_*`**），六项（l→a/v，a→l/v，v→l/a）求和。

7. **`hete_loss = ot_loss + local_proto_loss`**

### 3.2 超参对应

- **`num_prototypes`** → \(K\)
- **`lambda_ot`** → `ot_reg`（Sinkhorn 核 **`exp(-C/reg)`**）
- **`ot_num_iters`** → Sinkhorn 迭代次数

---

## 4. 同质对齐实现（`compute_homo_loss`，作用在 **`c_*`**）

**目标**：让三模态 **common** 分支在**低阶统计**与**分布距离**上一致。

### 4.1 矩匹配 `L_sem`

对每个 **`c_*`（`[N,d,T]`）** 在 **`(N,T)`** 上算：

- 均值 **`mu`**
- 方差 **`sigma`**
- 偏度 **`skew`**

对 **l/a/v 两两** 的 **`mu`、`sigma`、`skew`** 做 **平方差之和**。

### 4.2 MMD `L_mmd`

- **`c_*`** 时间维 **`mean` → `[N,d]`**。
- 两两模态 **`compute_mmd`**：RBF 核在样本维上构造 **`K_xx, K_yy, K_xy`**，  
  **`MMD = mean(K_xx)+mean(K_yy)-2·mean(K_xy)`**（带宽超参 `kernel_bandwidth=1.0`）。
- **`L_mmd = MMD(l,a)+MMD(l,v)+MMD(a,v)`**。

### 4.3 总同质损失

**`homo_loss = L_sem + L_mmd`**。

---

## 5. 融合与输出（`forward` 后半段）

在 **`s_*`、`c_*`** 与三种 **`loss`** 计算之后：

### 5.1 异质分支 A：Transformer 时序融合

- **`s_*` → `[T,N,d]`**，截断到 **`T_target = min(T_l,T_a,T_v)`**。
- **`concat` → `[T_target,N,3d]`** → **`transformer_fusion`**（`embed_dim=3d`）→ 取**最后时间步** **`fusion_rep_trans`**（**`[N,3d]`**）。

### 5.2 异质分支 B：跨模态注意力（CMA）

- 六组 **`trans_*_with_*`**（query 为一模态，key/value 为另一模态序列）+ **`trans_*_mem`**（3 层）处理拼接输出。
- **`last_h_l, last_h_a, last_h_v`** 拼 **`[N,6d]`** → **`cma_proj`** → **`fusion_rep_cma`**（**`[N,3d]`**）。

### 5.3 同质分支

- **`c_*`** 时间维 **`mean`** → 拼 **`fusion_rep_homo`**（**`[N,3d]`**）。

### 5.4 汇总与预测头

- **`fusion_rep_hete = fusion_rep_trans + fusion_rep_cma`**（逐元素加）。
- **`final_rep = concat(fusion_rep_hete, fusion_rep_homo)`** → **`[N,6d]`**。
- **`out_layer`**：`Linear(6d → 1)`（MOSI/MOSEI）或 **`→ 6`**（IEMOCAP）。

---

## 6. 训练时的损失组合（`trains/ATIO.py`）

```text
loss = main_loss + alpha1 * dec_loss + alpha2 * (hete_loss + homo_loss)
```

- **回归**：`main_loss = L1(logits, labels)`  
- **`alpha1`、`alpha2`**：来自 **`config/dec_config.json`**

验证 / 测试循环里累计的 **loss** 一般为 **主任务 `criterion` only**（不把 `dec/hete/homo` 加进 reported eval loss）。

---

## 7. 张量形状速查（对齐配置 `need_data_aligned=true` 时常见 `T=50`）

| 符号 | 典型形状 | 说明 |
|------|-----------|------|
| `proj_x_*` | `[N, d, T]` | 三模态统一维 `d`（如 40） |
| `s_*, c_*` | `[N, d, T]` | 解耦后 |
| `w_*` | `[N, K]` | 原型软分配 |
| `nu_*` | `[K]` | 批上平均边际 |
| `C` | `[K, K, K]` | 联合代价 |
| `fusion_rep_trans` / `fusion_rep_cma` / `fusion_rep_homo` | `[N, 3d]` | 融合中间量 |
| `final_rep` | `[N, 6d]` | 进 `out_layer` 前 |

---

## 8. 架构总览图（前向 + 损失挂载点）

```mermaid
flowchart TB
  subgraph IN["输入"]
    TXT["text: BERT 或 float 序列"]
    AUD["audio [N,T,Da]"]
    VID["video [N,T,Dv]"]
  end

  subgraph ENC["编码与形状"]
    BERT["BertTextEncoder 可选"]
    TR["transpose → [N,D,T]"]
    TXT --> BERT --> TR
    AUD --> TR2["transpose"]
    VID --> TR3["transpose"]
  end

  subgraph PROJ["统一维投影 Conv1d"]
    PL["proj_l"]
    PA["proj_a"]
    PV["proj_v"]
  end

  TR --> PL
  TR2 --> PA
  TR3 --> PV

  subgraph DEC["解耦 1x1 Conv1d"]
    UL["encoder_uni_l → s_l"]
    UA["encoder_uni_a → s_a"]
    UV["encoder_uni_v → s_v"]
    COM["encoder_com 共享 → c_l c_a c_v"]
  end

  PL --> UL & COM
  PA --> UA & COM
  PV --> UV & COM

  subgraph LOSSES["辅助损失"]
    LDEC["dec_loss: cos s 与 c"]
    LHET["hete_loss: OT + local proto"]
    LHOM["homo_loss: 矩 + MMD"]
  end

  UL & UA & UV & COM --> LDEC
  UL & UA & UV --> LHET
  COM --> LHOM

  subgraph HETE_FUSE["异质融合"]
    subgraph PATHA["路径A Transformer"]
      CAT1["s 截断 min T 后 concat → 3d"]
      TF["transformer_fusion"]
      OUTA["fusion_rep_trans 末步"]
    end
    subgraph PATHB["路径B 跨模态注意力"]
      CMA1["trans l/a/v with * + mem"]
      OUTB["concat → cma_proj → fusion_rep_cma"]
    end
    UL & UA & UV --> CAT1 --> TF --> OUTA
    UL & UA & UV --> CMA1 --> OUTB
    SUM["fusion_rep_hete = trans + cma"]
    OUTA --> SUM
    OUTB --> SUM
  end

  subgraph HOMO_FUSE["同质融合"]
    POOL["c 时间 mean → concat"]
    OUTH["fusion_rep_homo"]
  end

  COM --> POOL --> OUTH

  subgraph OUT["输出"]
    FIN["final_rep = cat hete homo → 6d"]
    HEAD["out_layer → logits"]
  end

  SUM --> FIN
  OUTH --> FIN
  FIN --> HEAD
```

---

## 9. 解耦子模块细化图

```mermaid
flowchart LR
  subgraph per_modality["每个模态 m ∈ l,a,v"]
    PM["proj_x_m: Conv1d orig→d"]
    PM --> UM["encoder_uni_m: Conv1d d→d 独立权重"]
    PM --> CM["encoder_com: Conv1d d→d 共享权重"]
    UM --> SM["s_m"]
    CM --> CMm["c_m"]
  end

  SM --> DLOSS["dec_loss: minimize mean cos s_m , c_m"]
  CMm --> DLOSS
```

---

## 10. 异质对齐数据流图

```mermaid
flowchart TB
  SL["s_l,s_a,s_v"]

  SL --> P1["时间 mean → feat N×d"]
  P1 --> W["compute_prototypes vs proto logvar → w N×K"]
  W --> NU["nu = mean w 归一化 K"]

  NU --> CMAT["pairwise_cost 三对 → C i,j,k"]
  PSL["proto_l,a,v + logvar"] --> CMAT

  CMAT --> SK["multi_marginal Sinkhorn → ot_loss"]
  NU --> SK

  SL --> LOC["local_proto_loss: w 加权 样本到异模态 proto"]

  SK --> HET["hete_loss = ot + local"]
  LOC --> HET
```

---

## 11. 同质对齐数据流图

```mermaid
flowchart TB
  CL["c_l,c_a,c_v"]

  CL --> STAT["每模态 mu sigma skew 在 N,T 上"]
  STAT --> LSEM["L_sem: 两两差的平方和"]

  CL --> POOL["时间 mean → N×d"]
  POOL --> MMD["compute_mmd 三对 la lv av"]
  MMD --> LMMD["L_mmd 求和"]

  LSEM --> HOM["homo_loss = L_sem + L_mmd"]
  LMMD --> HOM
```

---

## 12. 异质双路融合与输出图

```mermaid
flowchart TB
  S["s_l s_a s_v"]

  S --> A1["permute T N d 截断 T_target"]
  A1 --> C1["concat dim=2 → 3d"]
  C1 --> Tenc["transformer_fusion"]
  Tenc --> FT["fusion_rep_trans"]

  S --> A2["permute 截断 同 T_target"]
  A2 --> CMA["6 组 cross-attn + 3 mem"]
  CMA --> P6["concat last → 6d"]
  P6 --> CP["cma_proj → 3d"]
  CP --> FC["fusion_rep_cma"]

  FT --> ADD["elementwise +"]
  FC --> ADD
  ADD --> HE["fusion_rep_hete N×3d"]

  C["c_l c_a c_v"] --> M["mean T"]
  M --> FH["fusion_rep_homo N×3d"]

  HE --> CAT["concat dim=1 → 6d"]
  FH --> CAT
  CAT --> OL["out_layer → logits"]
```

---

## 13. 源码索引

| 内容 | 文件与符号 |
|------|------------|
| 解耦与投影 | `models/model.py`：`proj_*`、`encoder_uni_*`、`encoder_com` |
| `dec_loss` | `compute_decoupling_loss` |
| 异质 OT | `compute_hetero_loss`、`compute_prototypes`、`pairwise_cost`、`multi_marginal_sinkhorn` |
| 同质对齐 | `compute_homo_loss`、`compute_mmd` |
| 前向融合 | `forward` 步骤 7–7.4 |
| 训练加权 | `trains/ATIO.py`：`DecAlignTrainer.do_train` |

---

## 14. 事件序列仓库集成（`decalign_event.py` / `model.py`）

当前实现使用 **`decalign_event.ModalityAlignFusion`**（`DecAlignEventFusion` 为其三槽位 `(0,1,2)` 的兼容别名）。对齐支路为 **文本 / 图像 / skipgram** 三个槽位中的子集，由配置 `use_text`、`use_image`、`use_skipgram` 决定 `active_indices`；若三者均为关且 `use_decalign: true`，则退化为 **单槽位 (2)**，用 **`event_embeddings`（类型嵌入）** 作为 skipgram 槽输入，以便与旧「仅类型」配置兼容。

- **时间**：`TimePositionalEncoding` 在 **`ModalityAlignFusion` 之后** 与 `event_partial` 在最后一维拼接，经 **`decalign_post_time: Linear(hidden_size + embed_dim, hidden_size)`** 得到送入 LLM 的 `inputs_embeds`。时间**不再**参与解耦与异质/同质对齐。
- **异质原型**：每个数据集在 `model.py` 中持有独立的 `align_fusion[dataset_name]`，**`num_prototypes = max(2, 2 × num_event_types)`**（`dataset.yaml` 中该集的 `num_event_types`）。混合 batch 时按 `dataset_id` **分组**前向，对齐损失按组内样本数加权平均。
- **损失与两阶段训练**：`dec_loss` / `hete_loss` / `homo_loss` / 可选 `recon_loss`（`lambda_decalign_recon > 0` 且模块 `use_recon`）与下游 `type_loss`、`time_loss` 相加。若 `align_pretrain_epochs > 0`，当 `epoch ≤ align_pretrain_epochs` 时使用 `lambda_decalign_*_warm` 与 `loss_ratio_warm`，否则使用 `*_main` 与 `loss_ratio_main`（`runner.train` 将 `epoch` 传入 `forward`）。
- **数据**：可选 `skipgram_embeddings` 表（与 `type_embeddings` 同结构），由 `dataset.DataCollatorEventEmbedding` 写入 batch；路径由 `skipgram_embeddings_path` 与 `use_skipgram` 控制。
- **与参考 DecAlign 差异**：异质 OT 在 **M=3** 时仍为三边际 Sinkhorn，**M=2** 时为双边际 Sinkhorn；**M=1** 时 `hete_loss` 与 `homo_loss` 为 0。CMA/Transformer 仍沿事件维使用因果 mask（`decalign_transformer.build_causal_attn_mask`）。

---

*文档生成自 `DecAlign/` 当前实现；若代码变更请以仓库为准。*
