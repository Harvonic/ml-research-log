
import json
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, vocab_size, d=72, layers=2, heads=4, context=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(context, d)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=heads,
            dim_feedforward=d * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True
        )

        self.blocks = nn.TransformerEncoder(
            layer, 
            num_layers=layers,
            enable_nested_tensor=False
        )

        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)

        mask = torch.ones(T, T, device=x.device, dtype=torch.bool).triu(1)
        h = self.blocks(h, mask=mask)

        return self.head(self.norm(h))


# Load vocabulary (list or token-to-ID dictionary)
with open("vocab.json") as f:
    vocab = json.load(f)

tokens = sorted(vocab, key=vocab.get)

vocab_null = {t: i for i, t in enumerate(tokens)}
vocab_normal = {
    t: i for i, t in enumerate(t for t in tokens if t != "<NULL>")
}

model_null = Transformer(len(vocab_null))
model_normal = Transformer(len(vocab_normal))

for name, model in [("NULL", model_null), ("Normal", model_normal)]:
    params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {params:,} parameters")
