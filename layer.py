import torch
from torch import nn
import torch.nn.functional as F
import math
from utils import clamp_preserve_gradients

class Time2Vec(nn.Module):
    """
    Time2Vec时间编码方案
    τ = [Δt/T, sin(ω1Δt), cos(ω1Δt), ..., sin(ωkΔt), cos(ωkΔt)]
    其中T=1秒，ω_i可学习
    """
    def __init__(self, hidden_dim: int, k: int = 32):
        super().__init__()
        self.k = k // 2
        
        # 可学习的频率参数 ω_i
        self.freq = nn.Parameter(torch.randn(self.k) * 0.01)
        
    def forward(self, delta_times: torch.Tensor):
        """
        Args:
            delta_times: [batch_size, seq_len] 秒级时间差
        Returns:
            time_embeddings: [batch_size, seq_len, hidden_dim]
        """
        # 计算Time2Vec分量 [B, S, 2k+1]
        batch_size, seq_len = delta_times.shape
        dt = delta_times.unsqueeze(-1).to(delta_times.device)  # [B, S, 1]
        freq = self.freq.to(delta_times.device)
        
        # 正弦余弦分量
        angles = dt * freq.unsqueeze(0).unsqueeze(0)  # [B, S, k]
        sin_comp = torch.sin(angles)
        cos_comp = torch.cos(angles)
        
        # 拼接
        time_vec = torch.cat([sin_comp, cos_comp], dim=-1)  # [B, S, k]
        
        # MLP映射
        return time_vec


class TimePositionalEncoding(nn.Module):
    """Temporal encoding in THP, ICML 2020
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        i = torch.arange(0, d_model, 1)
        div_term = (2 * torch.div(i, 2, rounding_mode='trunc').float() * -(math.log(10000.0) / d_model)).exp()
        self.register_buffer('div_term', div_term)

    def forward(self, x):
        """Compute time positional encoding defined in Equation (2) in THP model.

        Args:
            x (tensor): time_seqs, [batch_size, seq_len]

        Returns:
            temporal encoding vector, [batch_size, seq_len, model_dim]

        """
        result = x.unsqueeze(-1) * self.div_term
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result


class MixLogNormal(nn.Module):
    """
    Given {(t_0, k_0), ..., (t_{i-1}, k_{i-1})}, predict t_i. 
    Implementation of Mixture LogNormal.
    """
    def __init__(self, hid_dim, num_mixture):
        super().__init__()
        
        self.linear_weight = nn.Sequential(
            nn.Linear(hid_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_mixture)
        )
        self.linear_mean = nn.Sequential(
            nn.Linear(hid_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_mixture)
        )
        self.linear_std = nn.Sequential(
            nn.Linear(hid_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_mixture)
        )

        nn.init.xavier_normal_(self.linear_weight[0].weight)
        nn.init.xavier_normal_(self.linear_weight[2].weight)
        nn.init.xavier_normal_(self.linear_mean[0].weight)
        nn.init.xavier_normal_(self.linear_mean[2].weight)
        nn.init.xavier_normal_(self.linear_std[0].weight)
        nn.init.xavier_normal_(self.linear_std[2].weight)

    def forward(self, embedding, time_delta_seqs):
        """
        Args: 
            embedding [batch_size, seq_len-1, hid_dim]: history embedding.
            time_delta_seqs [batch_size, seq_len-1]: ground truth delta time.
        Description:
            We compute the log probability of delta time under mix log normal distributions.
            We use softplus to keep σ > 0.
        """
        # [batch_size, seq_len, num_mixture]
        weight = self.linear_weight(embedding)
        mean = self.linear_mean(embedding)
        std = self.linear_std(embedding)
        std = clamp_preserve_gradients(std, -3, 2)
        std = F.softplus(std)

        log_p_t = self._compute_log_p_t(weight, mean, std, time_delta_seqs)
        return log_p_t

    def predict(self, embedding, time_delta_seqs):
        """
        Args: 
            embedding [batch_size, seq_len-1, hid_dim]: history embedding.
            time_delta_seqs [batch_size, seq_len-1]: ground truth delta time.
        Description:
            Predict expected delta time according to history embedding, and compute log likelihood of ground truth time.
        """
        # [batch_size, seq_len, num_mixture]
        weight = self.linear_weight(embedding)
        mean = self.linear_mean(embedding)
        std = self.linear_std(embedding)
        std = clamp_preserve_gradients(std, -3, 2)
        std = F.softplus(std)

        predict_time_delta = self._compute_expectation(weight, mean, std)
        log_p_t = self._compute_log_p_t(weight, mean, std, time_delta_seqs)

        return predict_time_delta, log_p_t

    def _compute_log_p_t(self, weight, mean, std, time_delta_seqs):
        """
        Description:
            Compute log probability of given time_delta_seqs under the distributions described by weight, 
                mean and std.
            We use softmax to keep weight be positive and sum to 1.
        Args:
            weight: [batch_size, seq_len, num_mixture]
            mean: [batch_size, seq_len, num_mixture]
            std: [batch_size, seq_len, num_mixture]
            time_delta_seqs: [batch_size, seq_len]
        Algorithm:
            PDF of single log normal distribution:
                p(t | μ, σ) = 1 / (t * σ * sqrt(2π)) * exp(-(ln(x)-μ)^2/(2σ^2))
            PDF of mix log normal distribution:
                p(t) = \sum_{k=1}^{num_mix} weight_k * p(t | μ_k, σ_k)
            We first compute log weighted probability of each component:
                log_component_k = log(weight_k) + log(p(t | μ_k, σ_k))
                                = log_softmax(weight) - log(t) - log(σ) - 0.5*log(2π) - (ln(x)-μ)^2/(2σ^2)
            Then we use logsumexp to compute final log probability:
                log_p_t = logsumexp(log_component_k, dim=-1)
        """
        log_t = torch.log(time_delta_seqs + 1e-8).unsqueeze(2)
        exponent = -0.5 * torch.square((log_t - mean) / (std))
        log_component_k = F.log_softmax(weight, dim=-1) - log_t - torch.log(std) - \
                            0.5 * math.log(2 * math.pi) + exponent

        log_p_t = torch.logsumexp(log_component_k, dim=-1)
        return log_p_t

    def _compute_expectation(self, weight, mean, std):
        """
        Description:
            Compute expectation of the distributions described by weight, mean and std.
            We use softmax to keep weight be positive and sum to 1.
        Args:
            weight: [batch_size, seq_len, num_mixture]
            mean: [batch_size, seq_len, num_mixture]
            std: [batch_size, seq_len, num_mixture]
        Algorithm:
            Expectation of mix log normal distribution:
                E(t) = sum_{k=1}^{num_mix} weight_k * exp(μ_k + 0.5*σ_k^2)
        """
        weight = F.softmax(weight, dim=-1)
        exponent = mean + 0.5 * torch.pow(std, 2)
        predict_time_delta = torch.sum(weight * torch.exp(exponent), dim=-1)
        return predict_time_delta


class TypeHead(nn.Module):
    """
    事件类型预测头（无参数版本）
    logits = h @ E_type.T
    """
    def __init__(self, type_embeddings: dict[str, torch.Tensor], temperature: float = 1.0):
        super().__init__()
        # 将嵌入字典注册为buffer（不可训练）
        self.temperature = nn.Parameter(torch.tensor(temperature))
        for name, emb in type_embeddings.items():
            self.register_buffer(f"type_emb_{name}", emb)
        self.type_names = list(type_embeddings.keys())

    def forward(self, hidden_states: torch.Tensor, dataset_id: str):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            dataset_id: 数据集名称
        Returns:
            logits: [batch, seq_len, num_types]
        """
        # 获取对应数据集的嵌入矩阵 [num_types, hidden_dim]
        type_emb = getattr(self, f"type_emb_{dataset_id}")
        
        # 计算logits: h @ E_type.T / τ
        logits = torch.matmul(hidden_states, type_emb.T) / self.temperature
        return logits


class TimeHead(nn.Module):
    """
    时间预测头（单线性层）
    pred_dt = linear_time(h)
    """
    def __init__(self, hid_dim: int):
        super().__init__()
        self.linear = nn.Linear(hid_dim, 1, bias=True)
        nn.init.xavier_normal_(self.linear.weight)
        
    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        Returns:
            pred_dt: [batch, seq_len, 1]
        """
        return self.linear(hidden_states).squeeze(-1)  # [B, S]


class TimeHeadMLP(nn.Module):
    """
    时间预测头（MLP）
    """
    def __init__(self, hid_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(hid_dim, 128, bias=True)
        self.linear2 = nn.Linear(128, 1, bias=True)
        self.relu = nn.ReLU()
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        
    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        Returns:
            pred_dt: [batch, seq_len, 1]
        """
        return self.linear2(self.relu(self.linear1(hidden_states))).squeeze(-1)  # [B, S]