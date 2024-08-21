import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_k, dtype=torch.float32)
        )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        score = self.softmax(scores)
        output = torch.matmul(score, v)
        return output, score
