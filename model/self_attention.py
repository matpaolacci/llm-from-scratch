import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()

        # Projection matrices
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """

        # Linear projection
        Q = self.W_Q(x)      # (B, N, d_k)
        K = self.W_K(x)      # (B, N, d_k)
        V = self.W_V(x)      # (B, N, d_v)

        # Query-Key similarity
        scores = torch.matmul(Q, K.transpose(-2, -1))
        # (B, N, N)

        # Scaling
        scores = scores / (K.size(-1) ** 0.5)

        # Casual mask
        scores = Q @ K.transpose(-2, -1)   # QK^T
        scores = scores / sqrt(d_k)

        # Softmax (B, N, N)
        attention = F.softmax(scores, dim=-1)

        # Somma pesata dei Value
        output = attention @ V
        # (B, N, d_v)

        return output, attention