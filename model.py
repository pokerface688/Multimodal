import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from PIL import Image
# 屏蔽Pillow的PNG调试日志
import logging
logging.getLogger("PIL").setLevel(logging.WARNING)  # 只显示警告及以上，屏蔽DEBUG/INFO
from collections import OrderedDict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModel
from peft import LoraConfig, TaskType, get_peft_model
from torchvision import models, transforms
from layer import Time2Vec, MixLogNormal, TypeHead, TimeHead, TimeHeadMLP, TimePositionalEncoding
from utils import compute_type_loss, compute_time_loss, compute_time_rmse, compute_type_acc, compute_RCA_loss, compute_JEPA_loss
from logger import ProjectLogger

logger = ProjectLogger(__name__).get_logger()

def apply_rope(x, theta):
    """
    x: [..., D]  theta: [...] 长度与 x 除最后一维外广播
    返回同 shape 旋转后向量
    """
    device, dtype = x.device, x.dtype
    D = x.shape[-1]
    half = D // 2
    # 频率
    freq = 1.0 / (10000 ** (torch.arange(half, device=device) / half))
    # 相位
    theta = theta.unsqueeze(-1) * freq                      # [..., half]
    cos, sin = torch.cos(theta), torch.sin(theta)
    # 旋转
    x1, x2 = x[..., :half], x[..., half:]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.cat([rx1, rx2], dim=-1).to(dtype)
    
def compute_instance_crossmodal_contrastive_loss(context_emb, text_emb, img_emb, non_pad_mask, temperature=0.07):
    """
    计算实例级别的跨模态对比损失 (类似 CLIP)
    正样本：同一个事件(同一时间步)的 LLM输出特征(Context), Text, Image
    负样本：Batch 内其他时间步/其他序列的特征
    """
    loss = torch.tensor(0.0, device=context_emb.device)
    valid_pairs = 0
    
    # 展平并获取有效掩码
    B, S, D = context_emb.shape
    mask_flat = non_pad_mask.view(-1).bool()
    
    # 1. 提取有效的 Context 特征并 L2 归一化
    c_flat = context_emb.view(-1, D)[mask_flat]  # [N, D]
    c_norm = F.normalize(c_flat, p=2, dim=1)
    N = c_norm.size(0)
    
    if N < 2:
        return loss

    # 2. Context <-> Text 对齐
    if text_emb is not None:
        t_flat = text_emb.view(-1, text_emb.shape[-1])[mask_flat]
        # 过滤掉文本为空(全0)的有效索引
        t_valid_mask = (t_flat.abs().sum(dim=-1) > 1e-5)
        if t_valid_mask.any():
            c_sub = c_norm[t_valid_mask]
            t_sub = F.normalize(t_flat[t_valid_mask], p=2, dim=1)
            
            # 计算相似度矩阵 [M, M]
            logits_c2t = torch.matmul(c_sub, t_sub.T) / temperature
            logits_t2c = logits_c2t.T
            
            # 对角线为正样本
            sub_labels = torch.arange(c_sub.size(0), device=context_emb.device)
            loss_c2t = F.cross_entropy(logits_c2t, sub_labels)
            loss_t2c = F.cross_entropy(logits_t2c, sub_labels)
            loss += (loss_c2t + loss_t2c) / 2.0
            valid_pairs += 1

    # 3. Context <-> Image 对齐
    if img_emb is not None:
        i_flat = img_emb.view(-1, img_emb.shape[-1])[mask_flat]
        i_valid_mask = (i_flat.abs().sum(dim=-1) > 1e-5)
        if i_valid_mask.any():
            c_sub = c_norm[i_valid_mask]
            i_sub = F.normalize(i_flat[i_valid_mask], p=2, dim=1)
            
            logits_c2i = torch.matmul(c_sub, i_sub.T) / temperature
            logits_i2c = logits_c2i.T
            
            sub_labels = torch.arange(c_sub.size(0), device=context_emb.device)
            loss_c2i = F.cross_entropy(logits_c2i, sub_labels)
            loss_i2c = F.cross_entropy(logits_i2c, sub_labels)
            loss += (loss_c2i + loss_i2c) / 2.0
            valid_pairs += 1

    return loss / valid_pairs if valid_pairs > 0 else loss

class BidirectionalCrossModalFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1, use_text=True, use_image=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
        self.scale = self.head_dim ** -0.5
        
        self.use_text = use_text
        self.use_image = use_image
        
        # ==================================================
        # 核心：为每个模态单独定义Q、K、V投影矩阵
        # ==================================================
        # 事件模态的QKV投影
        self.event_q = nn.Linear(embed_dim, embed_dim)
        self.event_k = nn.Linear(embed_dim, embed_dim)
        self.event_v = nn.Linear(embed_dim, embed_dim)
        
        if self.use_text:
            self.text_q = nn.Linear(embed_dim, embed_dim)
            self.text_k = nn.Linear(embed_dim, embed_dim)
            self.text_v = nn.Linear(embed_dim, embed_dim)
            self.fusion_t = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim)
            )
            
        if self.use_image:
            self.img_q = nn.Linear(embed_dim, embed_dim)
            self.img_k = nn.Linear(embed_dim, embed_dim)
            self.img_v = nn.Linear(embed_dim, embed_dim)
            self.fusion_i = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim)
            )
        
        self.attn_out_proj = nn.Linear(embed_dim, embed_dim)
        
        # ---------- 最终多模态融合层 ----------
        # 动态计算输入维度，避免只用单模态时报错
        num_modalities = 1 + int(self.use_text) + int(self.use_image)
        self.final_fusion = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
        
        # ---------- Transformer 标准组件 ----------
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def _split_heads(self, x):
        B, N, D = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
    def _merge_heads(self, x):
        B, H, N, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, H * d)
        
    def _scaled_dot_product_attention(self, q, k, v, mask=None):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1), -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = self._merge_heads(attn_out)
        attn_out = self.attn_out_proj(attn_out)
        return attn_out
        
    def _single_modal_bidirectional_fusion(self, 
                                          event_emb, 
                                          modal_emb, 
                                          modal_mask,
                                          event_q_proj, event_k_proj, event_v_proj,
                                          modal_q_proj, modal_k_proj, modal_v_proj,
                                          fusion_layer):
        B, S, D = event_emb.shape
        
        event_q = self._split_heads(event_q_proj(event_emb))
        event_k = self._split_heads(event_k_proj(event_emb))
        event_v = self._split_heads(event_v_proj(event_emb))
        
        modal_q = self._split_heads(modal_q_proj(modal_emb))
        modal_k = self._split_heads(modal_k_proj(modal_emb))
        modal_v = self._split_heads(modal_v_proj(modal_emb))
        
        # ================= 原理修正核心 =================
        # 1. 必须使用因果掩码 (下三角矩阵)，允许当前步 Attend 到过去所有的历史信息
        # tril=True 表示有效，~tril=True 表示需要被屏蔽(未来信息)
        causal_mask = torch.tril(torch.ones((S, S), device=event_emb.device, dtype=torch.bool))
        
        # 方向1：Event -> Modal (Event是Q，Modal是KV)
        mask_e2m = ~causal_mask.unsqueeze(0).expand(B, -1, -1)
        if modal_mask is not None:
            # 屏蔽未来的 Modal，以及本身就是 Padding/空数据的 Modal
            modal_mask_expanded = modal_mask.unsqueeze(1).expand(-1, S, -1)
            mask_e2m = mask_e2m | modal_mask_expanded
            
        attn_e2m = self._scaled_dot_product_attention(q=event_q, k=modal_k, v=modal_v, mask=mask_e2m)
        
        # 方向2：Modal -> Event (Modal是Q，Event是KV)
        # 屏蔽未来的 Event。Event的Padding通常在外部统一处理，此处仅应用因果掩码
        mask_m2e = ~causal_mask.unsqueeze(0).expand(B, -1, -1)
        
        attn_m2e = self._scaled_dot_product_attention(q=modal_q, k=event_k, v=event_v, mask=mask_m2e)
        
        # 融合双向结果
        fused = torch.cat([attn_e2m, attn_m2e], dim=-1)
        fused = fusion_layer(fused)
        
        # 如果当前时间步的 Modal 本身无效，将其查询结果置零，避免引入噪声
        if modal_mask is not None:
            fused[modal_mask] = 0.0
            
        return fused
        
    def forward(self, event_emb, text_emb=None, img_emb=None):
        B, S, D = event_emb.shape
        fused_components = [event_emb]  # 始终保留原始事件嵌入
        
        if self.use_text and text_emb is not None:
            text_mask = text_emb.abs().sum(dim=-1) < 1e-5
            fused_t = self._single_modal_bidirectional_fusion(
                event_emb, text_emb, text_mask,
                self.event_q, self.event_k, self.event_v,
                self.text_q, self.text_k, self.text_v,
                self.fusion_t
            )
            fused_components.append(fused_t)
        
        if self.use_image and img_emb is not None:
            img_mask = img_emb.abs().sum(dim=-1) < 1e-5
            fused_i = self._single_modal_bidirectional_fusion(
                event_emb, img_emb, img_mask,
                self.event_q, self.event_k, self.event_v,
                self.img_q, self.img_k, self.img_v,
                self.fusion_i
            )
            fused_components.append(fused_i)
        
        if len(fused_components) > 1:
            concatenated = torch.cat(fused_components, dim=-1)
            fused = self.final_fusion(concatenated)
        else:
            fused = event_emb
        
        x = self.norm1(event_emb + fused)
        x = self.norm2(x + self.ffn(x))
        return x

class EventPredictionModel(nn.Module):
    def __init__(self, model_config, data_config, type_embeddings_path: str):
        super().__init__()
        model_path = model_config.model_path + model_config.model_name
        self.use_image = model_config.use_image
        self.use_text = model_config.use_text
        type_embed = torch.load(type_embeddings_path, weights_only=True)
        self.type_embeddings = {k: v.to(torch.bfloat16) for k, v in type_embed.items()}
        # 验证嵌入维度一致性
        sample_emb = list(self.type_embeddings.values())[0]
        self.embed_dim = sample_emb.shape[1]

        
        # ---------- LLM（Decoder-only）----------
        llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ) if model_config.use_quantization else None,
            torch_dtype=torch.bfloat16,
            device_map={"": model_config.gpu}
        )

        if model_config.peft_type == 'lora':
            peft_config = LoraConfig(
                r=model_config.lora_rank,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj", "k_proj"],  # 修改：也包含k_proj以控制RoPE
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.llm = get_peft_model(llm, peft_config)
            logger.info("Use lora to finetune LLM with time-based RoPE.")
        elif model_config.peft_type == 'migration':
            peft_config = LoraConfig(
                r=model_config.lora_rank,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj", "k_proj"],  # 修改：也包含k_proj以控制RoPE
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.llm = get_peft_model(llm, peft_config)
            for name, param in self.llm.named_parameters():
                param.requires_grad = False
            logger.info("Migration. Freeze LLM with trained lora.")
        else:
            self.llm = llm
            logger.info("Freeze LLM.")
        
        if 'gemma3' in model_config.model_name:
            self.hidden_size = self.llm.config.text_config.hidden_size
        else:
            self.hidden_size = self.llm.config.hidden_size
        self.device = next(self.llm.parameters()).device
        
        
        # ---------- Tokenizer（仅用于特殊token）----------
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        
        if self.use_text:
            self.text_encoder = AutoModel.from_pretrained("./bert").to(torch.bfloat16).to(self.device)
            self.text_tokenizer = AutoTokenizer.from_pretrained("./bert")
            self.text_proj = nn.Linear(768, self.embed_dim).to(torch.bfloat16).to(self.device)
            for param in self.text_encoder.parameters(): param.requires_grad = False
            #for param in self.text_encoder.parameters(): param.requires_grad = True

        if self.use_image:
            # 1. 初始化一个没有预训练权重的 resnet18
            resnet = models.resnet18(pretrained=False) # 新版 PyTorch 也可以写成 weights=None
            
            # 2. 手动加载你下载到 ./resnet 目录下的本地权重文件
            # 请确保文件名与你实际下载的文件名一致
            local_weights_path = "./resnet/resnet18-f37072fd.pth" 
            resnet.load_state_dict(torch.load(local_weights_path, map_location="cpu"))
            
            # 3. 剥离最后的全连接层，按原逻辑放入 Sequential 并移动到对应设备和数据类型
            self.image_encoder = nn.Sequential(*list(resnet.children())[:-1]).to(torch.bfloat16).to(self.device)
            
            self.image_proj = nn.Linear(512, self.embed_dim).to(torch.bfloat16).to(self.device)
            self.image_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
            for param in self.image_encoder.parameters(): param.requires_grad = False
            #for param in self.image_encoder.parameters(): param.requires_grad = True
             # ================= 新增：图片特征缓存字典 =================
            # 用于缓存 image_encoder 的输出（512维），避免重复加载图片和计算 ResNet
            self.image_cache = {}
            
        
        # ========== 新增：对比学习投影头 (Contrastive Projection Heads) ==========
        # 将各模态映射到统一的对比学习空间 (例如 256 维)
        self.cl_dim = 256
        
        # Context 投影头 (将 LLM 的输出映射到对比空间)
        self.context_proj_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, self.cl_dim)
        ).to(torch.bfloat16).to(self.device)
        
        if self.use_text:
            self.text_proj_head = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, self.cl_dim)
            ).to(torch.bfloat16).to(self.device)
            
        if self.use_image:
            self.img_proj_head = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, self.cl_dim)
            ).to(torch.bfloat16).to(self.device)
            
        # ========== 新增：模态重建 (Masked Modality Reconstruction) ==========
        self.mask_prob = 0.15  # 随机 Mask 15% 的有效模态数据
        
        if self.use_text:
            # 文本的 Mask Token (可学习的参数)
            self.text_mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim, device=self.device, dtype=torch.bfloat16))
            nn.init.normal_(self.text_mask_token, std=0.02)
            
            # 文本重建头：从 LLM 的 hidden_size 映射回 embed_dim
            self.text_recon_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.GELU(),
                nn.Linear(self.hidden_size // 2, self.embed_dim)
            ).to(torch.bfloat16).to(self.device)
            
        if self.use_image:
            # 图像的 Mask Token (可学习的参数)
            self.img_mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim, device=self.device, dtype=torch.bfloat16))
            nn.init.normal_(self.img_mask_token, std=0.02)
            
            # 图像重建头
            self.img_recon_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.GELU(),
                nn.Linear(self.hidden_size // 2, self.embed_dim)
            ).to(torch.bfloat16).to(self.device)
        
        # ========== 融合层：开关都关 = 直接用原始嵌入 ==========
        self.fusion_required = self.use_image or self.use_text
        if self.fusion_required:
            
            in_dim = self.embed_dim * (1 + self.use_text + self.use_image)
            
            # 注释掉原来的 fusion_proj
            # self.fusion_proj = nn.Linear(in_dim, self.embed_dim).to(torch.bfloat16).to(self.device)
            
            # ====================== 修改：使用修正后的双向交叉注意力融合 ======================
            self.cross_modal_fusion = BidirectionalCrossModalFusion(
                embed_dim=self.embed_dim, 
                num_heads=4,  # 注意力头数，须能被 embed_dim 整除
                use_text=self.use_text,   # <--- 新增：传入文本开关
                use_image=self.use_image  # <--- 新增：传入图像开关
            ).to(torch.bfloat16).to(self.device)
            
        # ---------- 编码头 ----------
        self.tem_enc_type = model_config.tem_enc_type
        assert self.tem_enc_type in ["TimePositionEncoding", "RoPE"]

        if self.tem_enc_type == "TimePositionEncoding":
            self.tem_enc = TimePositionalEncoding(self.embed_dim).to(self.device)
            self.emb_up = nn.Linear(self.embed_dim * 2, self.hidden_size).to(self.device)
        elif self.tem_enc_type == "RoPE":
            self.emb_up = nn.Linear(self.embed_dim, self.hidden_size).to(self.device)

        self.time_scale = model_config.time_scale
        self.loss_ratio = model_config.loss_ratio
        self.RCA_ratio = model_config.RCA_ratio
        self.JEPA_ratio = model_config.JEPA_ratio
        self.RCA_type = model_config.RCA_type
        
        # ---------- 解码头 ----------
        # 类型头
        self.type_head = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(self.hidden_size, 128), nn.ReLU(), nn.Linear(128, spec.num_event_types)).to(self.device)
            for name, spec in data_config.items()
        })
        
        # 时间头：二选一
        if model_config.use_mixlognormal:
            self.time_head = nn.ModuleDict({
                name: MixLogNormal(self.hidden_size, model_config.num_mixture).to(self.device)
                for name in data_config.keys()
            })
        else:
            self.time_head = nn.ModuleDict({
                name: TimeHead(self.hidden_size).to(self.device)
                for name in data_config.keys()
            })

        # 根因分析 多分类头
        if self.RCA_type == "multi":
            self.RCA_head = nn.ModuleDict({
                name: nn.Sequential(nn.Linear(self.hidden_size, 128), nn.ReLU(), nn.Linear(128, spec.num_event_types)).to(self.device)
                for name, spec in data_config.items()
            })

        # 根因分析 二分类头
        elif self.RCA_type == "binary":
            self.RCA_head = nn.ModuleDict({
                name: nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.hidden_size, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1)
                    ).to(self.device)
                    for _ in range(spec.num_event_types)
                ])
                for name, spec in data_config.items()
            })

        # JPEA损失函数计算头
        self.JEPA_head = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(self.hidden_size, 128), nn.ReLU(), nn.Linear(128, self.hidden_size)).to(self.device)
            for name, spec in data_config.items()
        })
        self.log_contrastive_weight = nn.Parameter(
            torch.log(torch.tensor(0.01, device=self.device, dtype=torch.bfloat16))
        )
        self.log_recon_weight = nn.Parameter(
            torch.log(torch.tensor(0.01, device=self.device, dtype=torch.bfloat16))
        )
        
        # ========== 新增：可学习的多任务权重参数 ==========
        # 用于自动平衡 type_loss 和 time_loss 的比例
        self.log_sigma_type = nn.Parameter(
            torch.tensor(0.0, device=self.device, dtype=torch.bfloat16)
        )
        self.log_sigma_time = nn.Parameter(
            torch.tensor(0.0, device=self.device, dtype=torch.bfloat16)
        )
    def time_to_position(self, time_seqs):
        """
        将时间戳映射到RoPE位置索引
        Args:
            time_seqs: [B, S] 时间戳序列
        Returns:
            position_ids: [B, S] 位置索引，用于RoPE
        """
        position_ids = (time_seqs * self.time_scale)  # [B, S]
        return position_ids
        
     # ====================== 新增：批量文本编码函数 ======================
    @torch.no_grad()
    def encode_texts(self, texts_list):
        """
        输入：batch_texts = [batch, seq_len] 字符串列表
        输出：text_emb = [batch, seq_len, embed_dim]
        空字符串 → 零向量
        """
        B, S = len(texts_list), len(texts_list[0])
        text_emb = torch.zeros(B, S, self.embed_dim, dtype=torch.bfloat16, device=self.device)
        
        for i in range(B):
            # 过滤空文本
            valid_text_idx = [j for j, t in enumerate(texts_list[i]) if t.strip() != ""]
            valid_texts = [texts_list[i][j] for j in valid_text_idx]
            
            if not valid_texts:
                continue
                
            # 编码有效文本
            inputs = self.text_tokenizer(valid_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.text_encoder(**inputs)
            # 取池化特征 + 投影
            pooled_emb = outputs.last_hidden_state[:, 0, :]  # 取CLS token
            proj_emb = self.text_proj(pooled_emb.to(torch.bfloat16))
            
            # 回填到对应位置
            for idx, j in enumerate(valid_text_idx):
                text_emb[i, j] = proj_emb[idx]
                
        return text_emb

    # ====================== 修改：批量图片编码函数 (增加缓存机制) ======================
    @torch.no_grad()
    def encode_images(self, image_paths_list):
        """
        输入：batch_image_paths = [batch, seq_len] 路径列表
        输出：img_emb = [batch, seq_len, embed_dim]
        空路径 / 图片不存在 → 零向量（新增容错）
        """
        B, S = len(image_paths_list), len(image_paths_list[0])
        img_emb = torch.zeros(B, S, self.embed_dim, dtype=torch.bfloat16, device=self.device)
    
        for i in range(B):
            # 过滤空路径
            valid_img_idx = [j for j, p in enumerate(image_paths_list[i]) if p.strip() != ""]
            valid_paths = [image_paths_list[i][j] for j in valid_img_idx]
        
            if not valid_paths:
                continue
            
            feats_list = [None] * len(valid_paths)
            imgs_to_compute =[]
            compute_indices =[]
            
            # 1. 尝试从缓存获取 512 维特征
            for idx, p in enumerate(valid_paths):
                if p in self.image_cache:
                    feats_list[idx] = self.image_cache[p]
                else:
                    try:
                        # 核心：捕获图片不存在/损坏的异常
                        img = Image.open(p).convert("RGB")
                        img = self.image_transform(img).to(torch.bfloat16).to(self.device)
                        imgs_to_compute.append(img)
                        compute_indices.append(idx)
                    except FileNotFoundError:
                        # 图片不存在：打印警告，跳过
                        pass
                    except Exception as e:
                        # 其他图片错误（损坏、格式错误）：打印警告，跳过
                        logger.warning(f"图片读取失败 ({p}): {str(e)}")
            
            # 2. 批量计算未缓存的图片特征
            if len(imgs_to_compute) > 0:
                imgs_tensor = torch.stack(imgs_to_compute)  #[N, 3, 224, 224]
                computed_feats = self.image_encoder(imgs_tensor).flatten(1)  #[N, 512]
                
                # 存入缓存（存放到 CPU 避免 GPU 显存溢出）并记录到 feats_list
                for local_idx, original_idx in enumerate(compute_indices):
                    p = valid_paths[original_idx]
                    feat_cpu = computed_feats[local_idx].detach().cpu()
                    self.image_cache[p] = feat_cpu
                    feats_list[original_idx] = feat_cpu
            
            # 3. 收集所有成功获取的特征
            valid_feats = []
            valid_indices_in_batch =[]
            for idx, feat in enumerate(feats_list):
                if feat is not None:
                    valid_feats.append(feat)
                    valid_indices_in_batch.append(idx)
            
            # 没有能读取的图片，直接跳过
            if len(valid_feats) == 0:
                continue
            
            # 4. 统一进行投影计算 (将特征移回 GPU)
            feats_tensor = torch.stack(valid_feats).to(self.device)  #[N, 512]
            proj_emb = self.image_proj(feats_tensor) #[N, embed_dim]
        
            # 5. 回填到对应位置（只回填成功读取的图片）
            for local_idx, original_idx in enumerate(valid_indices_in_batch):
                j = valid_img_idx[original_idx]
                img_emb[i, j] = proj_emb[local_idx]
                
        return img_emb
        
    def forward(self, batch):
        """
        前向传播：事件嵌入 → LLM → 多任务预测
        修改：全程使用基于时间戳的RoPE
        """
        time_seqs = batch['time_seqs']
        time_delta_seqs = batch['time_delta_seqs']
        type_seqs = batch['type_seqs']
        event_embeddings = batch['event_embeddings']
        batch_non_pad_mask = batch['batch_non_pad_mask']
        attention_mask = batch['attention_mask']
        dataset_ids = batch['dataset_id']
        root_cause = batch['root_cause']

        bsz, seq_len = batch['time_delta_seqs'].shape
        
        
        # 对比学习和模态重建初始化
        cl_loss = torch.tensor(0.0, device=self.device)
        recon_loss = torch.tensor(0.0, device=self.device)
        masked_t_pos = None
        masked_i_pos = None
        
        # 保存原始特征的引用，用于后续对比学习和重建
        orig_text_emb = None
        orig_img_emb = None
        
        if self.fusion_required:
            text_emb = self.encode_texts(batch['texts']) if self.use_text else None
            img_emb = self.encode_images(batch['image_paths']) if self.use_image else None
            
            # 1. 拷贝原始特征，用于输出端的对比学习和重建
            orig_text_emb = text_emb.clone() if self.use_text else None
            orig_img_emb = img_emb.clone() if self.use_image else None
            
            # 2. 执行 Mask 逻辑 (仅在训练阶段进行 Mask)
            if self.training:
                if self.use_text:
                    valid_t_mask = (text_emb.abs().sum(dim=-1) > 1e-5) & batch_non_pad_mask
                    rand_t = torch.rand(text_emb.shape[:2], device=self.device) < self.mask_prob
                    masked_t_pos = valid_t_mask & rand_t
                    text_emb[masked_t_pos] = self.text_mask_token
                    
                if self.use_image:
                    valid_i_mask = (img_emb.abs().sum(dim=-1) > 1e-5) & batch_non_pad_mask
                    rand_i = torch.rand(img_emb.shape[:2], device=self.device) < self.mask_prob
                    masked_i_pos = valid_i_mask & rand_i
                    img_emb[masked_i_pos] = self.img_mask_token
            '''
            # 3. 双向交叉注意力融合和掩码重建
            event_embeddings = self.cross_modal_fusion(
                event_emb=event_embeddings, 
                text_emb=text_emb, 
                img_emb=img_emb
            )
            '''
            # 3. 双向交叉注意力融合 (注意：这里使用的是原始未投影的特征，保证语义完整)
            event_embeddings = self.cross_modal_fusion(
                event_emb=event_embeddings, 
                text_emb=orig_text_emb, 
                img_emb=orig_img_emb
            )
      
        # ================== 4. 过 LLM ==================
        if self.tem_enc_type == "TimePositionEncoding":
            tem_emb = self.tem_enc(time_delta_seqs)
            cat_emb = torch.cat((event_embeddings, tem_emb), dim=-1)
            event_emb = self.emb_up(cat_emb)

            outputs = self.llm(
                inputs_embeds=event_emb, 
                output_hidden_states=True, 
                attention_mask=attention_mask,
            )
        elif self.tem_enc_type == "RoPE":
            event_emb = self.emb_up(event_embeddings)

            position_ids = self.time_to_position(time_seqs)
            outputs = self.llm(
                inputs_embeds=event_emb, 
                output_hidden_states=True, 
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            
        last_hidden = outputs.hidden_states[-1] # [B, S, hidden_size]
        
        # ================== 5. 在输出端计算对比学习与重建 ==================
        if self.training and self.fusion_required:
            # --- 5.1 计算实例级跨模态对比损失 ---
            z_context = self.context_proj_head(last_hidden)
            
            z_text = None
            if self.use_text and orig_text_emb is not None:
                z_text = self.text_proj_head(orig_text_emb)
                
            z_img = None
            if self.use_image and orig_img_emb is not None:
                z_img = self.img_proj_head(orig_img_emb)

            cl_loss = compute_instance_crossmodal_contrastive_loss(
                context_emb=z_context,
                text_emb=z_text,  
                img_emb=z_img,
                non_pad_mask=batch_non_pad_mask,
                temperature=0.07
            )
            
            # --- 5.2 计算模态重建损失 ---
            if self.use_text and masked_t_pos is not None and masked_t_pos.any():
                pred_text = self.text_recon_head(last_hidden)
                loss_t = F.mse_loss(pred_text[masked_t_pos], orig_text_emb[masked_t_pos])
                recon_loss += loss_t
                
            if self.use_image and masked_i_pos is not None and masked_i_pos.any():
                pred_img = self.img_recon_head(last_hidden)
                loss_i = F.mse_loss(pred_img[masked_i_pos], orig_img_emb[masked_i_pos])
                recon_loss += loss_i

        # ================== 总损失汇总 ==================
        total_loss = 0.0
        
        # 建议对比学习初始权重调小，因为 InfoNCE Loss 数值通常比 MSE 大
        contrastive_weight = 0
        recon_weight = 0
        
        total_loss += contrastive_weight * cl_loss + recon_weight * recon_loss
        
        
        
        
        # ========================================================
        for i, did in enumerate(dataset_ids):
            # JEPA损失
            JEPA_head = self.JEPA_head[did]
            pred_next_hidden = JEPA_head(last_hidden[i:i+1, :-1, :])
            JEPA_loss = compute_JEPA_loss(pred_next_hidden, last_hidden[i:i+1, 1:, :])

            # 类型损失
            type_head = self.type_head[did]
            type_logits = type_head(pred_next_hidden)  # [1, S-1, K]
            type_loss = compute_type_loss(type_logits, type_seqs[i:i+1, 1:], batch_non_pad_mask[i:i+1, :-1])
            
            # 时间损失
            time_head = self.time_head[did]
            if isinstance(time_head, MixLogNormal):
                log_p_t = time_head(pred_next_hidden, time_delta_seqs[i:i+1, 1:])
                time_loss = compute_time_loss(log_p_t, batch_non_pad_mask[i:i+1, :-1])
            else:
                pred_dt = time_head(pred_next_hidden)
                # time_loss = F.huber_loss(
                #     pred_dt * batch_non_pad_mask[i:i+1, :-1].float(),
                #     time_delta_seqs[i:i+1, 1:] * batch_non_pad_mask[i:i+1, :-1].float(),
                #     delta=1.0
                # )
                time_loss = F.mse_loss(
                    pred_dt * batch_non_pad_mask[i:i+1, :-1].float(),
                    time_delta_seqs[i:i+1, 1:] * batch_non_pad_mask[i:i+1, :-1].float(),
                    reduction='sum'
                )

            # # 根因损失 多分类头
            if self.RCA_type == "multi":    
                RCA_head = self.RCA_head[did]
                RCA_logits = RCA_head(pred_next_hidden[0, -1, :])   
                RCA_loss = compute_RCA_loss(RCA_logits, root_cause[i])

            # 根因损失 二分类头
            elif self.RCA_type == "binary":
                RCA_heads = self.RCA_head[did]
                hidden = pred_next_hidden[0, -1, :]   # [D]

                logits = []
                targets = []

                for j, head in enumerate(RCA_heads):
                    logit = head(hidden)  # [1]
                    logits.append(logit)

                    target = torch.tensor(
                        [1.0 if j == root_cause[i].item() else 0.0],
                        device=hidden.device
                    )
                    targets.append(target)

                logits = torch.cat(logits)   # [K]
                targets = torch.cat(targets) # [K]

                RCA_loss = F.binary_cross_entropy_with_logits(logits, targets)

            total_loss += self.loss_ratio * (type_loss*1 + time_loss) + self.RCA_ratio * RCA_loss + self.JEPA_ratio * JEPA_loss
            
            '''
            # ---------- 可学习权重：自适应平衡 type_loss 和 time_loss ----------
            # 1. 从对数参数恢复出标准差 (sigma > 0)
            sigma_type = torch.exp(self.log_sigma_type)
            sigma_time = torch.exp(self.log_sigma_time)
            
            # 2. 计算加权损失 (核心公式)
            # 原理：如果某个任务的不确定性很高(sigma大)，就降低它的权重
            weighted_type = type_loss / (2 * sigma_type ** 2) + torch.log(sigma_type)
            weighted_time = time_loss / (2 * sigma_time ** 2) + torch.log(sigma_time)
            
            # 3. 累加总损失
            total_loss += (
                self.loss_ratio * (weighted_type + weighted_time)
                + self.RCA_ratio * RCA_loss
                + self.JEPA_ratio * JEPA_loss
            )
            '''
        return total_loss / bsz

    @torch.no_grad()
    def predict(self, batch):
        time_seqs = batch['time_seqs']
        time_delta_seqs = batch['time_delta_seqs']
        type_seqs = batch['type_seqs']
        event_embeddings = batch['event_embeddings']
        batch_non_pad_mask = batch['batch_non_pad_mask']
        attention_mask = batch['attention_mask']
        dataset_ids = batch['dataset_id']
        root_cause = batch['root_cause']
        
        #交叉注意力
        if self.fusion_required:
            text_emb = self.encode_texts(batch['texts']) if self.use_text else None
            img_emb = self.encode_images(batch['image_paths']) if self.use_image else None
            
            # 使用双向交叉注意力融合
            event_embeddings = self.cross_modal_fusion(
                event_emb=event_embeddings, 
                text_emb=text_emb, 
                img_emb=img_emb
            )
        

        bsz, seq_len = batch['time_delta_seqs'].shape
        
         # 1. 上投影并融合时间（替换原始event_embeddings为融合后的嵌入）
        if self.tem_enc_type == "TimePositionEncoding":
            tem_emb = self.tem_enc(time_delta_seqs)
            cat_emb = torch.cat((event_embeddings, tem_emb), dim=-1)  # 用融合嵌入替换原嵌入
            event_emb = self.emb_up(cat_emb)

            outputs = self.llm(
                inputs_embeds=event_emb, 
                output_hidden_states=True, 
                attention_mask=attention_mask,
            )
        elif self.tem_enc_type == "RoPE":
            event_emb = self.emb_up(event_embeddings)  # 用融合嵌入替换原嵌入

            position_ids = self.time_to_position(time_seqs)
            outputs = self.llm(
                inputs_embeds=event_emb, 
                output_hidden_states=True, 
                attention_mask=attention_mask,
                position_ids=position_ids
            )

        last_hidden = outputs.hidden_states[-1]  # [B, S, D]
        
        total_l2 = 0.0
        total_correct = 0
        total_events = 0
        RCA_correct = 0
        RCA_total = 0
        
        for i, did in enumerate(dataset_ids):
            JEPA_head = self.JEPA_head[did]
            pred_next_hidden = JEPA_head(last_hidden[i:i+1, :-1, :])
            
            # 类型预测
            type_head = self.type_head[did]
            type_logits = type_head(pred_next_hidden)
            c, t = compute_type_acc(type_logits, type_seqs[i:i+1, 1:], batch_non_pad_mask[i:i+1, :-1])
            total_correct += c
            total_events += t
            
            # 时间预测
            time_head = self.time_head[did]
            if isinstance(time_head, MixLogNormal):
                pred_dt, log_p_t = time_head.predict(pred_next_hidden, time_delta_seqs[i:i+1, 1:])
            else:
                pred_dt = time_head(pred_next_hidden)
            
            # RMSE
            l2, _ = compute_time_rmse(pred_dt, time_delta_seqs[i:i+1, 1:], batch_non_pad_mask[i:i+1, :-1])
            total_l2 += l2

            # # RCA 多分类头 
            if self.RCA_type == "multi":    
                RCA_head = self.RCA_head[did]
                RCA_logits = RCA_head(pred_next_hidden[0, -1, :]) 
                pred = torch.argmax(RCA_logits, dim=-1) 
                if pred == root_cause[i]:
                    RCA_correct += 1
                RCA_total += 1

            # RCA 二分类头
            elif self.RCA_type == "binary":            
                RCA_heads = self.RCA_head[did]
                hidden = pred_next_hidden[0, -1, :]
                probs = []
                for head in RCA_heads:
                    logit = head(hidden)
                    prob = torch.sigmoid(logit)
                    probs.append(prob)

                probs = torch.cat(probs)   # [K]
                pred = torch.argmax(probs)
                if pred == root_cause[i]:
                    RCA_correct += 1
                RCA_total += 1

            
        return total_l2, total_correct, total_events, RCA_correct, RCA_total