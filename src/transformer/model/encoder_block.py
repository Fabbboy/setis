from torch import nn
from transformer.layer.normalization import LayerNorm
from transformer.layer.pwff import PositionWiseFeedForward
from transformer.layer.multihead import MultiHeadAttention


class EncoderBlock(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop=0.1):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop)

        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden, drop)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop)

    def forward(self, x, src_mask):
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        x = self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)

        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
