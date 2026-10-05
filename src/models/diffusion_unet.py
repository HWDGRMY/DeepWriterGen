import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(t_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.drop = nn.Dropout(dropout)
        self.act = nn.SiLU()
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(t)[:, :, None]
        h = self.act(self.norm2(self.conv2(h)))
        return self.skip(x) + self.drop(h)


class CrossAttn1D(nn.Module):
    """修复版 CrossAttention。"""
    def __init__(self, dim, ctx_dim, heads=4, dim_head=64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner = heads * dim_head
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(ctx_dim, inner, bias=False)
        self.to_v = nn.Linear(ctx_dim, inner, bias=False)
        self.to_out = nn.Linear(inner, dim)

    def forward(self, x, ctx):
        B, C, L = x.shape
        S = ctx.shape[1]
        q = self.to_q(x.permute(0, 2, 1))
        k = self.to_k(ctx)
        v = self.to_v(ctx)
        q = q.view(B, L, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(B, S, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(B, S, self.heads, self.dim_head).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.to_out(out).permute(0, 2, 1)


class DiffusionUNet1D(nn.Module):
    def __init__(self, in_ch=2, out_ch=2, style_dim=512,
                 num_chars=2000, text_dim=64, text_hidden=128,
                 base=64, mults=(1, 2, 4, 8), n_res=2,
                 t_dim=256, dropout=0.1, attn_heads=4):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 4), nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim))

        self.char_embed = nn.Embedding(num_chars, text_dim, padding_idx=0)
        self.text_lstm = nn.LSTM(text_dim, text_hidden,
                                 batch_first=True, bidirectional=True)

        self.cond = nn.Sequential(
            nn.Linear(style_dim + text_hidden * 2, t_dim), nn.SiLU(),
            nn.Linear(t_dim, t_dim))

        self.init = nn.Conv1d(in_ch, base, 3, padding=1)
        self.down = nn.ModuleList()
        chs = [base]
        cur = base
        for i, m in enumerate(mults):
            oc = base * m
            for _ in range(n_res):
                self.down.append(ResBlock1D(cur, oc, t_dim, dropout))
                cur = oc
                chs.append(cur)
            if i != len(mults) - 1:
                self.down.append(nn.Conv1d(cur, cur, 3, 2, 1))
                chs.append(cur)

        self.mid1 = ResBlock1D(cur, cur, t_dim, dropout)
        self.mid_a = CrossAttn1D(cur, t_dim, heads=attn_heads, dim_head=64)
        self.mid2 = ResBlock1D(cur, cur, t_dim, dropout)

        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            oc = base * m
            for _ in range(n_res + 1):
                self.up.append(ResBlock1D(cur + chs.pop(), oc, t_dim, dropout))
                cur = oc
            if i != 0:
                self.up.append(nn.ConvTranspose1d(cur, cur, 4, 2, 1))

        self.out = nn.Sequential(
            nn.GroupNorm(8, cur), nn.SiLU(),
            nn.Conv1d(cur, out_ch, 3, padding=1))

    def forward(self, x, t, style_vec, text_ids):
        t_emb = self.time_emb(t)
        emb = self.char_embed(text_ids)
        _, (h, _) = self.text_lstm(emb)
        txt = torch.cat([h[0], h[1]], dim=-1)
        cond = self.cond(torch.cat([style_vec, txt], dim=-1)) + t_emb

        h = self.init(x)
        skips = [h]
        for l in self.down:
            h = l(h, cond) if isinstance(l, ResBlock1D) else l(h)
            skips.append(h)

        h = self.mid1(h, cond)
        h = self.mid_a(h, cond[:, None, :])
        h = self.mid2(h, cond)

        for l in self.up:
            if isinstance(l, ResBlock1D):
                h = torch.cat([h, skips.pop()], 1)
                h = l(h, cond)
            else:
                h = l(h)
        return self.out(h)