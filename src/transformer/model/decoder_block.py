from torch import nn

from transformer.layer.normalization import LayerNorm
from transformer.layer.multihead import MultiHeadAttention
from transformer.layer.pwff import PositionWiseFeedForward


class DecoderBlock(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop=0.1):
        super(DecoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop)

        self.src_attention = MultiHeadAttention(d_model, n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop)

        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden, drop)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        _x = x
        x = self.self_attention(q=x, k=x, v=x, mask=tgt_mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.src_attention(q=x, k=enc_out, v=enc_out, mask=src_mask)
        x = self.dropout2(x)
        x = self.norm2(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x
