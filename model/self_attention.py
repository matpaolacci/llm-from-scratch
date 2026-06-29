import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, max_len):
        super().__init__()

        # Projection matrices
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)

        # Precomputed casual mask (more efficent)
        self.register_buffer(
            "attention_mask",
            torch.tril(torch.ones(max_len, max_len)).bool()
        )

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        B, N, _ = x.shape

        # Linear projection
        Q = self.W_Q(x)      # (B, N, d_k)
        K = self.W_K(x)      # (B, N, d_k)
        V = self.W_V(x)      # (B, N, d_v)

        # Query-Key similarity --> (B, N, N)
        scores = Q @ K.transpose(-2, -1)

        # Scaling
        scores = scores / torch.sqrt(K.size(-1))

        # Causal masking: blocks attention to future positions by setting upper-triangular
        # scores to -inf before softmax, enforcing left-to-right autoregressive attention.
        scores = scores.masked_fill(
            ~self.attention_mask[:N, :N],
            float("-inf")
        )

        # Softmax --> (B, N, N)
        attention = F.softmax(scores, dim=-1)

        # Somma pesata dei Value --> (B, N, d_v)
        output = attention @ V

        return output, attention