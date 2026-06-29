import torch
from models.self_attention import SelfAttention


def test_self_attention_forward_shapes():
    """
    Tests that the forward pass produces outputs with the correct shapes.

    Ensures:
    - output has shape (B, N, d_v)
    - attention matrix has shape (B, N, N)
    """
    torch.manual_seed(0)

    B, N, d_model = 2, 5, 16
    d_k, d_v = 8, 8

    x = torch.randn(B, N, d_model)

    attn = SelfAttention(d_model, d_k, d_v, max_len=10)
    output, weights = attn(x)

    assert output.shape == (B, N, d_v)
    assert weights.shape == (B, N, N)


def test_self_attention_row_sum():
    """
    Tests that attention weights form valid probability distributions.

    Ensures:
    - Each row of the attention matrix sums to 1 (softmax correctness)
    """
    torch.manual_seed(0)

    B, N, d_model = 1, 6, 16
    d_k, d_v = 8, 8

    x = torch.randn(B, N, d_model)

    attn = SelfAttention(d_model, d_k, d_v, max_len=10)
    _, weights = attn(x)

    row_sums = weights.sum(dim=-1)

    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_causal_mask_enforcement():
    """
    Tests that causal masking correctly prevents attention to future tokens.

    Ensures:
    - Upper triangular part of attention matrix is ~0
    - No token can attend to future positions
    """
    torch.manual_seed(0)

    B, N, d_model = 1, 6, 16
    d_k, d_v = 8, 8

    x = torch.randn(B, N, d_model)

    attn = SelfAttention(d_model, d_k, d_v, max_len=10)
    _, weights = attn(x)

    # Future positions mask (upper triangle)
    upper_tri = torch.triu(torch.ones(N, N), diagonal=1).bool()

    future_weights = weights[:, upper_tri]

    assert torch.all(future_weights < 1e-6)


def test_backward_pass():
    """
    Tests that the module supports backpropagation correctly.

    Ensures:
    - Gradients are computed for input tensor
    - No NaN values appear in gradients
    - Computational graph is valid
    """
    torch.manual_seed(0)

    B, N, d_model = 2, 5, 16
    d_k, d_v = 8, 8

    x = torch.randn(B, N, d_model, requires_grad=True)

    attn = SelfAttention(d_model, d_k, d_v, max_len=10)

    output, _ = attn(x)

    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()