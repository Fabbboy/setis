from torch import nn

from transformer.model.encoder_block import EncoderBlock
from transformer.embedding.embedding import Embedding


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        d_model,
        ffn_hidden,
        n_head,
        n_layers,
        drop_prob,
        device,
    ):
        super(Encoder, self).__init__()
        self.embedding = Embedding(vocab_size, d_model, max_len, drop_prob, device)
        self.layers = nn.ModuleList(
            [
                EncoderBlock(d_model, ffn_hidden, n_head, drop_prob)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return x
