from torch import nn

from transformer.model.decoder_block import DecoderBlock
from transformer.embedding.embedding import Embedding


class Decoder(nn.Module):
    def __init__(
        self,
        voc_size,
        max_len,
        d_model,
        ffn_hidden,
        n_head,
        n_layers,
        drop_prob,
        device,
    ):
        super(Decoder, self).__init__()
        self.embedding = Embedding(voc_size, d_model, max_len, drop_prob, device)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(d_model, ffn_hidden, n_head, drop_prob)
                for _ in range(n_layers)
            ]
        )

        self.fc = nn.Linear(d_model, voc_size)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.fc(x)