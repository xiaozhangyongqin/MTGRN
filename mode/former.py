import torch
import torch.nn as nn
# from torchinfo import summary

class AttentionLayer(nn.Module):
    """"
    Attention、Multi-Head Attention
    """
    def __init__(self, model_dim, num_heads=4, mask=False):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads

        # QKV
        self.fc_Q = nn.Linear(model_dim, model_dim)
        self.fc_K = nn.Linear(model_dim, model_dim)
        self.fc_V = nn.Linear(model_dim, model_dim)
        # out

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q(b, ..., tg_len, model_dim)
        # K, V(b, ..., src_len, model_dim)
        batch_size = query.shape[0]
        tgt_len = query.shape[-2]
        src_len = key.shape[-2]

        query = self.fc_Q(query)
        key = self.fc_K(key)
        value = self.fc_V(value)

        # 多头拆分
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        # 转置
        key = key.transpose(-1, -2)     # (num_heads * batch_size, ..., head_dim, src_len)
        att_score = query @ key / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_len, src_len)

        if self.mask:
            mask = torch.ones(tgt_len, src_len, dtype=torch.bool, device=query.device).tril()   #
            att_score.masked_fill_(~mask, -torch.inf)  # 
        att_score = torch.softmax(att_score, dim=-1)
        out = att_score @ value     # (num_heads * batch_size, ..., tgt_len, head_dim)
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)  # (batch_size, ..., tgt_len, head_dim * num_heads)
        out = self.out_proj(out)

        return out

class SelfAttentionLayer(nn.Module):
    def __init__(self, model_dim, feed_forward_dim=2048, num_heads=4, dropout=0.1, mask=False):
        super().__init__()
        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        # x.shape() = (batch_size, ..., seq_len, model_dim)
        x = x.transpose(dim, -2)
        residual = x
        out = self.attn(x, x, x)
        out = self.dropout1(out)
        out = self.ln1(out + residual)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(out + residual)

        out = out.transpose(dim, -2)
        return out

if __name__ == '__main__':
    input = torch.randn(64, 12, 170, 48)

    selfAttention = SelfAttentionLayer(model_dim=48, feed_forward_dim=2048, num_heads=4, dropout=0.1, mask=False)
    # time
    out_time = selfAttention(input, dim=1)
    # space
    out_s = selfAttention(input, dim=2)
    print(out_time.shape)
    print(out_s.shape)
    # print(summary(selfAttention, input_size=(64, 12, 170, 48)))
