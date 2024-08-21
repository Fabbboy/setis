from torch import nn

from transformer.embedding.positional import PositionalEncoding

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(Embedding, self).__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop(tok_emb + pos_emb)