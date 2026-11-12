import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rope(q, k):
    B, H, T, D = q.shape
    assert D % 2 == 0

    positions = torch.arange(T, device=q.device)

    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, D, 2, device=q.device).float() / D)
    )

    angles = torch.outer(positions, inv_freq)

    cos = angles.cos()[None, None, :, :].to(q.dtype)
    sin = angles.sin()[None, None, :, :].to(q.dtype)

    def rotate(x):
        x = x.reshape(B, H, T, D // 2, 2)

        x1 = x[..., 0]
        x2 = x[..., 1]

        rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

        return rotated.flatten(-2)

    return rotate(q), rotate(k)


class XSAAttention(nn.Module):

    def __init__(self, d=72, heads=4, lam=1):
        super().__init__()
        assert d % heads == 0

        self.heads = heads
        self.head_dim = d // heads
        self.lam = lam

        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.out = nn.Linear(d, d)

    def forward(self, x):

        B, T, D = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # [B, T, D] -> [B, heads, T, head_dim]
        q = q.reshape(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to queries and keys
        q, k = apply_rope(q, k)
        v = v.reshape(B, T, self.heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # XSA
        v_norm = torch.nn.functional.normalize(v, dim=-1)
        z = y - self.lam * (y * v_norm).sum(dim=-1, keepdim=True) * v_norm

        # [B, heads, T, head_dim] -> [B, T, D]
        z = z.transpose(1, 2).contiguous().reshape(B, T, D) 

        return self.out(z)

class Block(nn.Module):

    def __init__(self, d, heads, lam):
        super().__init__()

        self.norm1 = nn.LayerNorm(d)
        self.attn = XSAAttention(d, heads, lam)

        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Transformer(nn.Module):

    def __init__(self, vocab_size, d=72, layers=2, heads=4, context=128, lam=1):
        super().__init__()

        self.context = context

        self.token_emb = nn.Embedding(vocab_size, d)

        self.blocks = nn.ModuleList([
            Block(d, heads, lam) for _ in range(layers)
        ])

        self.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        assert T <= self.context

        x = self.token_emb(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits

if __name__ == "__main__":
    model = Transformer(vocab_size=100, d=72, layers=2, heads=4, context=128, lam=1)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
