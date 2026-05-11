import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import datetime
from pathlib import Path
import json
from typing import Dict, Optional


def init_save_dir():
    current_date = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%Y%m%d-%H%M")
    process_id = os.getpid()
    save_dir = f"./result/{current_date}-{process_id}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    return save_dir

def set_seed(seed=1037):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def clamp_preserve_gradients(x: torch.Tensor, min: float, max: float) -> torch.Tensor:
    """Clamp the tensor while preserving gradients in the clamped region."""
    return x + (x.clamp(min, max) - x).detach()

def compute_type_loss(logits, label, batch_non_pad_mask):
    """
    Compute neg loglike loss for type_predictor.
    Args:
        logits [batch_size, seq_len-1, num_types]: direct output of Linear layer.
        label [batch_size, seq_len-1]: ground truth label.
        batch_non_pad_mask [batch_size, seq_len-1]: 1 for non-pad, 0 for pad.
    Description:
        We use negative log p(k_i) as our loss function. i.e. type part of the loglike loss.
    """
    log_p_k = F.log_softmax(logits, dim=-1)
    index = torch.where(batch_non_pad_mask, label, 0).unsqueeze(2)
    log_p_k_i = torch.gather(log_p_k, dim=2, index=index).squeeze(-1)
    loss = log_p_k_i * batch_non_pad_mask.float()
    return -loss.sum()

def compute_time_loss(log_p_t, batch_non_pad_mask):
    """
    Compute neg loglike loss for time_predictor.
    Args:
        log_p_t [batch_size, seq_len-1]: log probability of ground truth delta time under MixLogNormal.
        batch_non_pad_mask [batch_size, seq_len-1]: 1 for non-pad, 0 for pad.
    Description:
        We use negative log p_t as our loss function. i.e. time part of the loglike loss.
    """        
    loss = log_p_t * batch_non_pad_mask.float()
    return -loss.sum()

def compute_type_acc(logits, label, batch_non_pad_mask):
    """
    Compute accuracy for type_predictor.
    Args:
        logits [batch_size, seq_len-1, num_types]: direct output of Linear layer.
        label [batch_size, seq_len-1]: ground truth label.
        batch_non_pad_mask [batch_size, seq_len-1]: 1 for non-pad, 0 for pad.
    """   
    pred = torch.argmax(logits, dim=-1)
    correct = ((pred == label) * batch_non_pad_mask).sum()
    total = batch_non_pad_mask.sum()
    return correct.item(), total.item()

def compute_time_rmse(hat_t, t, batch_non_pad_mask):
    """
    Compute rmse for time_predictor.
    Args:
        hat_t [batch_size, seq_len-1]: predicted delta time of MixLogNormal.
        t [batch_size, seq_len-1]: ground truth delta time.
        batch_non_pad_mask [batch_size, seq_len-1]: 1 for non-pad, 0 for pad.
    """
    L2 = (hat_t - t) ** 2
    L2_sum = (L2 * batch_non_pad_mask).sum()
    count = batch_non_pad_mask.sum()
    return L2_sum.item(), count.item()

def compute_time_mae(hat_t, t, batch_non_pad_mask):
    """
    Compute mse for time_predictor.
    Args:
        hat_t [batch_size, seq_len-1]: predicted delta time of MixLogNormal.
        t [batch_size, seq_len-1]: ground truth delta time.
        batch_non_pad_mask [batch_size, seq_len-1]: 1 for non-pad, 0 for pad.
    """
    L1 = torch.abs(hat_t - t)
    L1_sum = (L1 * batch_non_pad_mask).sum()
    count = batch_non_pad_mask.sum()
    return L1_sum.item(), count.item()

def compute_RCA_loss(RCA_logits, label):
    logits = RCA_logits.unsqueeze(0)          # (1, num_classes)
    target = label.unsqueeze(0)          # (1,)
    loss = F.cross_entropy(logits, target)
    return loss

def compute_JEPA_loss(pred_hidden, hidden):
    criterion = torch.nn.MSELoss()
    loss = criterion(pred_hidden, hidden)
    return loss