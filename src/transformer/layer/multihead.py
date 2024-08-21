from torch import nn
from transformer.layer.sdpa import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.attention = ScaledDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor)
        return tensor.permute(0, 2, 1, 3)

    def concat(self, tensor):
        batch_size, n_head, length, d_tensor = tensor.size()
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        return tensor.view(batch_size, length, d_tensor * n_head)
    
    def forward(self, q, k, v, mask=None):
        q,k,v = self.w_q(q), self.w_k(k), self.w_v(v)

        # split by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        out, _ = self.attention(q, k, v, mask)

        # concat heads
        out = self.concat(out)
        out = self.w_concat(out)

        return out