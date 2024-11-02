import torch
import torch.nn as nn
from .attention import ScaledDotProductAttention
from .positional_encoding import PositionalEncoding

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion):
        super().__init__()
        self.attention = ScaledDotProductAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)
        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        return self.norm2(forward + x)

class Transformer(nn.Module):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, vocab_size, max_length):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = PositionalEncoding(embed_size, max_length)

        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, forward_expansion) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        out = self.word_embedding(x) + self.position_embedding(x)
        for layer in self.layers:
            out = layer(out, out, out)
        return self.fc_out(out)
