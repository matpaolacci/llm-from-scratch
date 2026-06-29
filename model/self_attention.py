import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()

        # Matrici di proiezione
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """

        # 1. Proiezioni lineari
        Q = self.W_Q(x)      # (B, N, d_k)
        K = self.W_K(x)      # (B, N, d_k)
        V = self.W_V(x)      # (B, N, d_v)

        # 2. Similarità Query-Key
        scores = torch.matmul(Q, K.transpose(-2, -1))
        # (B, N, N)

        # 3. Scaling
        scores = scores / (K.size(-1) ** 0.5)

        # 4. Softmax
        attention = F.softmax(scores, dim=-1)
        # (B, N, N)

        # 5. Somma pesata dei Value
        output = torch.matmul(attention, V)
        # (B, N, d_v)

        return output, attention