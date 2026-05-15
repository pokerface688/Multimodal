import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from PIL import Image
# 屏蔽Pillow的PNG调试日志
import logging
logging.getLogger("PIL").setLevel(logging.WARNING)  # 只显示警告及以上，屏蔽DEBUG/INFO
from collections import OrderedDict, defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModel
from peft import LoraConfig, TaskType, get_peft_model
from torchvision import models, transforms
from layer import Time2Vec, MixLogNormal, TypeHead, TimeHead, TimeHeadMLP, TimePositionalEncoding
from decalign_event import ModalityAlignFusion
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
    

class EventPredictionModel(nn.Module):
    def __init__(self, model_config, data_config, type_embeddings_path: str):
        super().__init__()
        model_path = model_config.model_path + model_config.model_name
        self.use_image = model_config.use_image
        self.use_text = model_config.use_text
        self.use_skipgram = bool(getattr(model_config, "use_skipgram", False))
        try:
            type_embed = torch.load(type_embeddings_path, weights_only=True)
        except TypeError:
            type_embed = torch.load(type_embeddings_path)
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
        
        # 事件投影头 (让静态的 event_embeddings 拥有可学习的非线性变换)
        self.event_proj_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        ).to(torch.bfloat16).to(self.device)
        
        if self.use_text:
            self.text_proj_head = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, self.embed_dim)
            ).to(torch.bfloat16).to(self.device)
            
        if self.use_image:
            self.img_proj_head = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, self.embed_dim)
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
        self.fusion_required = self.use_image or self.use_text or self.use_skipgram
        if self.fusion_required:
            
            in_dim = self.embed_dim * (1 + self.use_text + self.use_image + self.use_skipgram)
            
            # 注释掉原来的 fusion_proj
            self.fusion_proj = nn.Linear(in_dim, self.embed_dim).to(torch.bfloat16).to(self.device)
            
            
        # ---------- 编码头 ----------
        self.tem_enc_type = model_config.tem_enc_type
        assert self.tem_enc_type in ["TimePositionEncoding", "RoPE"]
        self.use_decalign = bool(getattr(model_config, "use_decalign", False))
        self.lambda_decalign_dec = float(getattr(model_config, "lambda_decalign_dec", 0.0))
        self.lambda_decalign_hete = float(getattr(model_config, "lambda_decalign_hete", 0.0))
        self.lambda_decalign_homo = float(getattr(model_config, "lambda_decalign_homo", 0.0))
        self.lambda_decalign_recon = float(getattr(model_config, "lambda_decalign_recon", 0.0))
        self.align_pretrain_epochs = int(getattr(model_config, "align_pretrain_epochs", 0))
        self.lambda_decalign_dec_warm = float(getattr(model_config, "lambda_decalign_dec_warm", self.lambda_decalign_dec))
        self.lambda_decalign_hete_warm = float(getattr(model_config, "lambda_decalign_hete_warm", self.lambda_decalign_hete))
        self.lambda_decalign_homo_warm = float(getattr(model_config, "lambda_decalign_homo_warm", self.lambda_decalign_homo))
        self.lambda_decalign_recon_warm = float(getattr(model_config, "lambda_decalign_recon_warm", self.lambda_decalign_recon))
        self.lambda_decalign_dec_main = float(getattr(model_config, "lambda_decalign_dec_main", self.lambda_decalign_dec))
        self.lambda_decalign_hete_main = float(getattr(model_config, "lambda_decalign_hete_main", self.lambda_decalign_hete))
        self.lambda_decalign_homo_main = float(getattr(model_config, "lambda_decalign_homo_main", self.lambda_decalign_homo))
        self.lambda_decalign_recon_main = float(getattr(model_config, "lambda_decalign_recon_main", self.lambda_decalign_recon))
        self.loss_ratio_warm = float(getattr(model_config, "loss_ratio_warm", model_config.loss_ratio))
        self.loss_ratio_main = float(getattr(model_config, "loss_ratio_main", model_config.loss_ratio))

        if self.use_decalign and self.tem_enc_type != "TimePositionEncoding":
            raise ValueError("use_decalign requires tem_enc_type=TimePositionEncoding (tem_enc for time after alignment).")

        if self.tem_enc_type == "TimePositionEncoding":
            self.tem_enc = TimePositionalEncoding(self.embed_dim).to(self.device)
            if not self.use_decalign:
                self.emb_up = nn.Linear(self.embed_dim * 2, self.hidden_size).to(self.device)
        elif self.tem_enc_type == "RoPE":
            self.emb_up = nn.Linear(self.embed_dim, self.hidden_size).to(self.device)

        if self.use_skipgram:
            sp = getattr(model_config, "skipgram_embeddings_path", None)
            if not sp:
                raise ValueError("use_skipgram=True requires model_config.skipgram_embeddings_path")
            try:
                sg = torch.load(sp, map_location="cpu", weights_only=True)
            except TypeError:
                sg = torch.load(sp, map_location="cpu")
            sg_dim = list(sg.values())[0].shape[1]
            if sg_dim != self.embed_dim:
                self.skipgram_proj = nn.Linear(sg_dim, self.embed_dim, bias=False).to(torch.bfloat16).to(self.device)
            else:
                self.skipgram_proj = nn.Identity()
        else:
            self.skipgram_proj = nn.Identity()

        if self.use_decalign:
            d_dm = int(getattr(model_config, "decalign_d_model", self.embed_dim))
            nh = int(getattr(model_config, "decalign_num_heads", 4))
            nl = int(getattr(model_config, "decalign_nlevels", 2))
            ck = int(getattr(model_config, "decalign_conv1d_kernel_size", 1))
            ad = float(getattr(model_config, "decalign_attn_dropout", 0.1))
            ada = float(getattr(model_config, "decalign_attn_dropout_a", ad))
            adv = float(getattr(model_config, "decalign_attn_dropout_v", ad))
            lot = float(getattr(model_config, "decalign_lambda_ot", 0.1))
            oti = int(getattr(model_config, "decalign_ot_num_iters", 50))
            self.decalign_use_recon = float(getattr(model_config, "lambda_decalign_recon", 0.0)) > 0

            active = tuple(i for i, on in enumerate([self.use_text, self.use_image, self.use_skipgram]) if on)
            self.align_fallback_type_to_skipgram_slot = False
            if len(active) < 1:
                self.align_fallback_type_to_skipgram_slot = True
                active = (2,)
            M = len(active)
            if (M * d_dm) % nh != 0:
                raise ValueError(f"decalign: (M*d_model)={M * d_dm} must be divisible by decalign_num_heads={nh}")
            if d_dm % nh != 0:
                raise ValueError(f"decalign: d_model={d_dm} must be divisible by decalign_num_heads={nh}")
            mem_in = (M - 1) * d_dm if M > 1 else d_dm
            if M > 1 and mem_in % nh != 0:
                raise ValueError(f"decalign: (M-1)*d_model={mem_in} must be divisible by decalign_num_heads={nh}")
            self.align_active_indices = active

            self.align_fusion = nn.ModuleDict()
            for name, spec in data_config.items():
                nk = max(2, 2 * int(spec.num_event_types))
                self.align_fusion[name] = ModalityAlignFusion(
                    embed_dim=self.embed_dim,
                    hidden_size=self.hidden_size,
                    d_model=d_dm,
                    num_heads=nh,
                    nlevels=nl,
                    num_prototypes=nk,
                    active_indices=active,
                    conv1d_kernel_size=ck,
                    attn_dropout=ad,
                    attn_dropout_a=ada,
                    attn_dropout_v=adv,
                    lambda_ot=lot,
                    ot_num_iters=oti,
                    use_recon=self.decalign_use_recon,
                ).to(torch.bfloat16).to(self.device)

            self.decalign_text_null = nn.Parameter(
                torch.zeros(1, 1, self.embed_dim, device=self.device, dtype=torch.bfloat16)
            )
            self.decalign_image_null = nn.Parameter(
                torch.zeros(1, 1, self.embed_dim, device=self.device, dtype=torch.bfloat16)
            )
            self.decalign_skipgram_null = nn.Parameter(
                torch.zeros(1, 1, self.embed_dim, device=self.device, dtype=torch.bfloat16)
            )
            nn.init.normal_(self.decalign_text_null, std=0.02)
            nn.init.normal_(self.decalign_image_null, std=0.02)
            nn.init.normal_(self.decalign_skipgram_null, std=0.02)

            self.decalign_post_time = nn.Linear(self.hidden_size + self.embed_dim, self.hidden_size).to(
                torch.bfloat16
            ).to(self.device)

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
            torch.log(torch.tensor(0.1, device=self.device, dtype=torch.bfloat16))
        )
        self.log_recon_weight = nn.Parameter(
            torch.log(torch.tensor(0.1, device=self.device, dtype=torch.bfloat16))
        )
        
        # ========== 新增：可学习的多任务权重参数 ==========
        # 用于自动平衡 type_loss 和 time_loss 的比例
        self.log_sigma_type = nn.Parameter(
            torch.tensor(0.0, device=self.device, dtype=torch.bfloat16)
        )
        self.log_sigma_time = nn.Parameter(
            torch.tensor(0.0, device=self.device, dtype=torch.bfloat16)
        )

    def _align_schedule(self, epoch: int):
        if self.align_pretrain_epochs > 0 and epoch <= self.align_pretrain_epochs:
            return (
                self.lambda_decalign_dec_warm,
                self.lambda_decalign_hete_warm,
                self.lambda_decalign_homo_warm,
                self.lambda_decalign_recon_warm,
                self.loss_ratio_warm,
            )
        return (
            self.lambda_decalign_dec_main,
            self.lambda_decalign_hete_main,
            self.lambda_decalign_homo_main,
            self.lambda_decalign_recon_main,
            self.loss_ratio_main,
        )

    def _modality_inputs_for_align(self, batch, bsz, seq_len, dtype):
        if getattr(self, "align_fallback_type_to_skipgram_slot", False):
            text_emb = self.decalign_text_null.expand(bsz, seq_len, -1).to(dtype)
            img_emb = self.decalign_image_null.expand(bsz, seq_len, -1).to(dtype)
            skipgram_emb = batch["event_embeddings"].to(dtype)
            return text_emb, img_emb, skipgram_emb
        if self.use_text:
            text_emb = self.encode_texts(batch["texts"]).to(dtype)
        else:
            text_emb = self.decalign_text_null.expand(bsz, seq_len, -1).to(dtype)
        if self.use_image:
            img_emb = self.encode_images(batch["image_paths"]).to(dtype)
        else:
            img_emb = self.decalign_image_null.expand(bsz, seq_len, -1).to(dtype)
        if getattr(self, "align_fallback_type_to_skipgram_slot", False):
            skipgram_emb = batch["event_embeddings"].to(dtype)
        elif self.use_skipgram:
            sk = batch["skipgram_embeddings"].to(self.device).to(dtype)
            skipgram_emb = self.skipgram_proj(sk)
        else:
            skipgram_emb = self.decalign_skipgram_null.expand(bsz, seq_len, -1).to(dtype)
        return text_emb, img_emb, skipgram_emb

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
            pooled_emb = outputs.last_hidden_state[:, 0, :]  # <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>取CLS token
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
        
    def forward(self, batch, epoch: int = 10**9):
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
        
        
        #对比学习
        cl_loss = torch.tensor(0.0, device=self.device)
        
        
        #模态重建
        recon_loss = torch.tensor(0.0, device=self.device) # 新增：重建损失初始化
        masked_t_pos = None
        masked_i_pos = None
        
        
        # 融合：非 DecAlign 路径下 fusion_proj；DecAlign 路径下对齐支路为 text / image / skipgram，时间在融合后拼接
        if self.fusion_required and not self.use_decalign:
            embs = [event_embeddings]
            if self.use_text:
                embs.append(self.encode_texts(batch['texts']))
            if self.use_image:
                embs.append(self.encode_images(batch['image_paths']))
            if self.use_skipgram:
                sk = batch['skipgram_embeddings'].to(self.device).to(event_embeddings.dtype)
                embs.append(self.skipgram_proj(sk))
            event_embeddings = self.fusion_proj(torch.cat(embs, dim=-1))

        type_raw = batch['event_embeddings']
        decalign_dec = torch.tensor(0.0, device=self.device)
        decalign_hete = torch.tensor(0.0, device=self.device)
        decalign_homo = torch.tensor(0.0, device=self.device)
        decalign_recon = torch.tensor(0.0, device=self.device)

        if self.use_decalign:
            lam_dec, lam_hete, lam_homo, lam_recon, eff_loss_ratio = self._align_schedule(epoch)
        else:
            lam_dec = lam_hete = lam_homo = lam_recon = 0.0
            eff_loss_ratio = self.loss_ratio

        if self.use_decalign:
            if self.tem_enc_type != "TimePositionEncoding":
                raise RuntimeError("use_decalign requires TimePositionEncoding.")
            dtype = type_raw.dtype
            text_emb, img_emb, skipgram_emb = self._modality_inputs_for_align(batch, bsz, seq_len, dtype)
            tem_emb = self.tem_enc(time_delta_seqs.to(dtype))
            pad_m = batch_non_pad_mask.bool()
            event_emb = torch.zeros(bsz, seq_len, self.hidden_size, device=self.device, dtype=dtype)
            idx_map = defaultdict(list)
            for i, did in enumerate(dataset_ids):
                idx_map[did].append(i)
            wtot = float(bsz)
            s_dec = s_hete = s_homo = s_recon = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            for did, idxs in idx_map.items():
                ii = torch.tensor(idxs, device=self.device, dtype=torch.long)
                da = self.align_fusion[did](
                    text_emb.index_select(0, ii),
                    img_emb.index_select(0, ii),
                    skipgram_emb.index_select(0, ii),
                    pad_m.index_select(0, ii),
                )
                partial = da["event_partial"]
                te = tem_emb.index_select(0, ii)
                merged = self.decalign_post_time(torch.cat([partial, te], dim=-1))
                event_emb.index_copy_(0, ii, merged)
                wg = float(len(idxs))
                s_dec = s_dec + da["dec_loss"].to(torch.float32) * wg
                s_hete = s_hete + da["hete_loss"].to(torch.float32) * wg
                s_homo = s_homo + da["homo_loss"].to(torch.float32) * wg
                s_recon = s_recon + da["recon_loss"].to(torch.float32) * wg
            decalign_dec = s_dec / wtot
            decalign_hete = s_hete / wtot
            decalign_homo = s_homo / wtot
            decalign_recon = s_recon / wtot
            outputs = self.llm(
                inputs_embeds=event_emb,
                output_hidden_states=True,
                attention_mask=attention_mask,
            )
        elif self.tem_enc_type == "TimePositionEncoding":
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
                position_ids=position_ids,
            )
        last_hidden = outputs.hidden_states[-1]
        
        

        
        
        # ================== 总损失汇总 ==================
        total_loss = (
            lam_dec * decalign_dec
            + lam_hete * decalign_hete
            + lam_homo * decalign_homo
            + lam_recon * decalign_recon
        )
        
        
        
        
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

            RCA_loss = torch.tensor(0.0, device=self.device)

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

            total_loss += eff_loss_ratio * (type_loss*1 + time_loss) + self.RCA_ratio * RCA_loss + self.JEPA_ratio * JEPA_loss
            
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
        
        # 融合与 LLM（与 forward 同路径）
        if self.fusion_required and not self.use_decalign:
            embs = [event_embeddings]
            if self.use_text:
                embs.append(self.encode_texts(batch['texts']))
            if self.use_image:
                embs.append(self.encode_images(batch['image_paths']))
            if self.use_skipgram:
                sk = batch['skipgram_embeddings'].to(self.device).to(event_embeddings.dtype)
                embs.append(self.skipgram_proj(sk))
            event_embeddings = self.fusion_proj(torch.cat(embs, dim=-1))

        type_raw = batch['event_embeddings']
        bsz, seq_len = batch['time_delta_seqs'].shape

        if self.use_decalign:
            if self.tem_enc_type != "TimePositionEncoding":
                raise RuntimeError("use_decalign requires TimePositionEncoding.")
            dtype = type_raw.dtype
            text_emb, img_emb, skipgram_emb = self._modality_inputs_for_align(batch, bsz, seq_len, dtype)
            tem_emb = self.tem_enc(time_delta_seqs.to(dtype))
            pad_m = batch_non_pad_mask.bool()
            event_emb = torch.zeros(bsz, seq_len, self.hidden_size, device=self.device, dtype=dtype)
            idx_map = defaultdict(list)
            for i, did in enumerate(dataset_ids):
                idx_map[did].append(i)
            for did, idxs in idx_map.items():
                ii = torch.tensor(idxs, device=self.device, dtype=torch.long)
                da = self.align_fusion[did](
                    text_emb.index_select(0, ii),
                    img_emb.index_select(0, ii),
                    skipgram_emb.index_select(0, ii),
                    pad_m.index_select(0, ii),
                )
                partial = da["event_partial"]
                te = tem_emb.index_select(0, ii)
                merged = self.decalign_post_time(torch.cat([partial, te], dim=-1))
                event_emb.index_copy_(0, ii, merged)
            outputs = self.llm(
                inputs_embeds=event_emb,
                output_hidden_states=True,
                attention_mask=attention_mask,
            )
        elif self.tem_enc_type == "TimePositionEncoding":
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
                position_ids=position_ids,
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