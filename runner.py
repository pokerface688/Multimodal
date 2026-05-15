import os, warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_WARNINGS"] = "1"
warnings.filterwarnings("ignore")

import torch, yaml, time, math
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model import EventPredictionModel
from dataset import build_multiloader
from utils import set_seed, init_save_dir
from config import Config
from logger import setup_logging, ProjectLogger

logger = ProjectLogger(__name__).get_logger()


class Runner:
    def __init__(self, args):
        # ---------- config ----------
        full_data_config = Config.build_from_yaml_file(args.data_config_path)
        self.data_config = {name: full_data_config[name] for name in args.dataset}
        self.model_config = Config.build_from_yaml_file(args.model_config_path)
        set_seed(args.seed)

        # cmd override
        for attr in ['gpu', 'model_name', 'load_model_path', 'evaluate_only', 'batch_size', 'epoch', 'peft_type', 
                    'use_mixlognormal', 'use_prompt', 'type_embeddings_path', 'time_scale', 'patience', 'model_path',
                    'lora_lr', 'opt_lr', 'train_subset_ratio', 'loss_ratio', 'RCA_ratio', 'JEPA_ratio', 'tem_enc_type', 'use_image', 'use_text',
                    'use_skipgram', 'skipgram_embeddings_path',
                    'RCA_type', 'lambda_decalign_dec', 'lambda_decalign_hete', 'lambda_decalign_homo',
                    'lambda_decalign_recon', 'align_pretrain_epochs',
                    'lambda_decalign_dec_warm', 'lambda_decalign_hete_warm', 'lambda_decalign_homo_warm', 'lambda_decalign_recon_warm',
                    'lambda_decalign_dec_main', 'lambda_decalign_hete_main', 'lambda_decalign_homo_main', 'lambda_decalign_recon_main',
                    'loss_ratio_warm', 'loss_ratio_main',
                    'decalign_d_model', 'decalign_num_heads', 'decalign_nlevels', 'decalign_num_prototypes',
                    'decalign_conv1d_kernel_size', 'decalign_attn_dropout', 'decalign_attn_dropout_a', 'decalign_attn_dropout_v',
                    'decalign_lambda_ot', 'decalign_ot_num_iters']:
            if getattr(args, attr, None) is not None:
                setattr(self.model_config, attr, getattr(args, attr))

        # ---------- logger ----------
        self.save_dir = init_save_dir()
        backup_config = Config({'model': self.model_config, 'data': self.data_config})
        backup_config.to_yaml_file(Path(self.save_dir) / 'config.yaml')
        setup_logging(log_dir=self.save_dir)

        # ---------- tokenizer ----------
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_path + self.model_config.model_name)
        
        # ---------- 类型嵌入路径 ----------
        dataset_name = "_".join(args.dataset)
        self.type_embeddings_path = Path(self.model_config.type_embeddings_path) / f'type_embeddings.pt'

        # ---------- data ----------
        if self.model_config.RCA_ratio > 0:
            self.use_root_cause = True
        else:
            self.use_root_cause = False

        self.train_loader = build_multiloader(
            self.data_config, 'train',
            self.model_config.batch_size, num_workers=0,
            train_subset_ratio=self.model_config.train_subset_ratio,
            type_embeddings_path=self.type_embeddings_path,
            use_prompt=self.model_config.use_prompt,
            use_root_cause=self.use_root_cause,
            skipgram_embeddings_path=getattr(self.model_config, "skipgram_embeddings_path", None),
            use_skipgram=bool(getattr(self.model_config, "use_skipgram", False)),
        )
        # dev/test
        self.dev_loaders = {
            name: build_multiloader(
                {name: spec}, 'dev', self.model_config.batch_size,
                type_embeddings_path=self.type_embeddings_path,
                use_prompt=self.model_config.use_prompt,
                use_root_cause=self.use_root_cause,
                skipgram_embeddings_path=getattr(self.model_config, "skipgram_embeddings_path", None),
                use_skipgram=bool(getattr(self.model_config, "use_skipgram", False)),
            )
            for name, spec in self.data_config.items()
        }
        self.test_loaders = {
            name: build_multiloader(
                {name: spec}, 'test', self.model_config.batch_size,
                type_embeddings_path=self.type_embeddings_path,
                use_prompt=self.model_config.use_prompt,
                use_root_cause=self.use_root_cause,
                skipgram_embeddings_path=getattr(self.model_config, "skipgram_embeddings_path", None),
                use_skipgram=bool(getattr(self.model_config, "use_skipgram", False)),
            )
            for name, spec in self.data_config.items()
        }
        logger.critical("Successfully load multi-dataset.")

        # ---------- model ----------
        self.device = torch.device("cuda", self.model_config.gpu)
        self.model = EventPredictionModel(self.model_config, self.data_config, self.type_embeddings_path)
        if args.load_model_path:
            self.load(args.load_model_path)
            logger.critical(f"Load checkpoint from {args.load_model_path}")

        # ---------- optimizer ----------
        lora_params, head_params = [], []
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                if 'lora_' in n:          # A/B 矩阵
                    lora_params.append(p)
                else:                     # 时间头 / 类型头
                    head_params.append(p)

        self.opt = torch.optim.AdamW([
                {'params': head_params, 'lr': self.model_config.opt_lr},
                {'params': lora_params, 'lr': self.model_config.lora_lr}
            ], lr=self.model_config.opt_lr, weight_decay=self.model_config.weight_decay)
        
        # 调度器
        total_steps = self.model_config.epoch * len(self.train_loader)
        warmup_steps = int(0.1 * total_steps)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(self.opt, total_iters=warmup_steps)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=total_steps - warmup_steps)
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.opt, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps])

        logger.critical("Initialized model.")
        logger.info(f"Num of trainable parameters {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

    # ======================== save ========================
    def save(self):
        trainable = {name for name, p in self.model.named_parameters() if p.requires_grad}
        ckpt = {k: v for k, v in self.model.state_dict().items() if k in trainable}
        logger.debug(f"save {len(ckpt)} parameters.")
        torch.save(ckpt, Path(self.save_dir) / "model_weights.pth")

    # ======================== load ========================
    def load(self, save_path):
        ckpt = torch.load(save_path, weights_only=True, map_location='cpu')
        self.model.load_state_dict(ckpt, strict=False)

    # ======================== train ========================
    def train(self):
        best_metric = -1e8
        best_epoch = 0
        epoch_times = []
        
        for epoch in range(1, self.model_config.epoch + 1):
            tick = time.perf_counter()

            self.model.train()
            running_loss = 0.0
            total_batches = len(self.train_loader)

            with logging_redirect_tqdm():
                pbar = tqdm(enumerate(self.train_loader), total=total_batches,
                            desc=f"Epoch {epoch}", leave=False, ncols=100)
                for idx, batch in pbar:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items()}
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        loss = self.model(batch, epoch=epoch)
                    self.opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.opt.step()
                    running_loss += loss.item()
                    
                    pbar.set_postfix(loss=running_loss / (idx + 1))

            epoch_loss = running_loss / len(self.train_loader)
            logger.info(f'Epoch {epoch} train loss {epoch_loss:.4f}')

            epoch_times.append(time.perf_counter() - tick)

            if epoch % self.model_config.valid_freq == 0:
                metrics = self.predict(self.dev_loaders, "Dev")
                if self.model_config.RCA_ratio != 0:
                    avg_RCA = np.mean([m['RCA_acc'] for m in metrics.values()])
                    cur_metric = avg_RCA
                elif self.model_config.loss_ratio != 0:
                    # 用平均 acc - rmse 做 early stopping
                    avg_acc = np.mean([m['acc'] for m in metrics.values()])
                    avg_rmse = np.mean([m['rmse'] for m in metrics.values()])
                    cur_metric = avg_acc - avg_rmse
                else:
                    cur_metric = -round(epoch_loss, 4)
                    
                if cur_metric > best_metric:
                    best_metric = cur_metric
                    best_epoch = epoch
                    self.save()
                    logger.info(f"Best updated at epoch {epoch}")
                if epoch - best_epoch >= self.model_config.patience:
                    logger.critical("Early stopping.")
                    break
                    
            self.scheduler.step()
            
        if best_epoch > 0:
            self.load(Path(self.save_dir) / "model_weights.pth")
            logger.critical(f'Load best model state from epoch {best_epoch}')
        logger.info('Training finished!')

        if best_epoch == 0:
            self.save()
        avg_time = sum(epoch_times) / len(epoch_times)
        logger.info(f'Average epoch time: {avg_time:.2f}s')

    # ======================== predict ========================
    @torch.no_grad()
    def predict(self, loaders: dict, phase: str):
        """
        loaders: dict{name: DataLoader}
        return dict{name: {'rmse':xx, 'acc':xx}}
        """
        self.model.eval()
        ret = {}
        for name, loader in loaders.items():
            l2, correct, total, cRCA, tRCA = 0, 0, 0, 0, 0
            for batch in loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    l2b, cb, tb, cRCAb, tRCAb = self.model.predict(batch)
                l2 += l2b
                correct += cb
                total += tb
                cRCA += cRCAb
                tRCA += tRCAb
                
            rmse = math.sqrt(l2 / total)
            acc = correct / total * 100
            RCA_acc = cRCA / tRCA * 100
            ret[name] = {'rmse': rmse, 'acc': acc, 'RCA_acc': RCA_acc}
            logger.critical(f"{phase} {name}: rmse={rmse:.4f}  acc={acc:.2f}%  RCA_acc={RCA_acc:.2f}%")
        return ret
    # ======================== run ========================
    def run(self):
        if not self.model_config.evaluate_only:
            self.train()
        self.predict(self.test_loaders, "Test")