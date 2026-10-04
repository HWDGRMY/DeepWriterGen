import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = h + self.time_mlp(t_emb)[:, :, None]
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        return h + self.skip(x)


class CrossAttention1D(nn.Module):
    def __init__(self, dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = heads * dim_head
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context):
        # x: (B, C, L), context: (B, S, C_ctx)
        B, C, L = x.shape
        q = self.to_q(x.permute(0, 2, 1))  # (B, L, inner)
        k = self.to_k(context)  # (B, S, inner)
        v = self.to_v(context)
        q = q.view(B, L, self.heads, -1).transpose(1, 2)
        k = k.view(B, -1, self.heads, -1).transpose(1, 2)
        v = v.view(B, -1, self.heads, -1).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, L, -1)
        out = self.to_out(out)
        return out.permute(0, 2, 1)  # (B, C, L)


class DiffusionUNet1D(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, style_dim=512, char_embed_dim=128,
                 bbox_dim=4, base_channels=64, channel_mults=(1, 2, 4, 8), num_res_blocks=2,
                 time_emb_dim=256, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        self.char_embed = nn.Embedding(5000, char_embed_dim)  # 假设字符数5000
        self.bbox_proj = nn.Linear(bbox_dim, char_embed_dim)
        # 条件向量拼接
        self.cond_proj = nn.Linear(style_dim + char_embed_dim + char_embed_dim, time_emb_dim)

        self.init_conv = nn.Conv1d(in_channels, base_channels, 3, padding=1)

        # Down blocks
        self.downs = nn.ModuleList()
        channels = [base_channels]
        current_ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResBlock1D(current_ch, out_ch, time_emb_dim, dropout))
                current_ch = out_ch
                channels.append(current_ch)
            if i != len(channel_mults) - 1:
                self.downs.append(nn.Conv1d(current_ch, current_ch, 3, stride=2, padding=1))
                channels.append(current_ch)

        # Middle
        self.mid1 = ResBlock1D(current_ch, current_ch, time_emb_dim, dropout)
        self.mid_attn = CrossAttention1D(current_ch, time_emb_dim)
        self.mid2 = ResBlock1D(current_ch, current_ch, time_emb_dim, dropout)

        # Up blocks
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.ups.append(ResBlock1D(current_ch + channels.pop(), out_ch, time_emb_dim, dropout))
                current_ch = out_ch
            if i != 0:
                self.ups.append(nn.ConvTranspose1d(current_ch, current_ch, 4, stride=2, padding=1))

        self.out = nn.Sequential(
            nn.GroupNorm(8, current_ch),
            nn.SiLU(),
            nn.Conv1d(current_ch, out_channels, 3, padding=1)
        )

    def forward(self, x, t, style_vec, char_ids, bboxes):
        """
        x: (B, 2, L) 噪声轨迹
        t: (B,) 时间步
        style_vec: (B, style_dim)
        char_ids: (B,) 字符ID
        bboxes: (B, 4) bbox
        """
        B, C, L = x.shape
        t_emb = self.time_emb(t)  # (B, time_emb_dim)
        char_emb = self.char_embed(char_ids)  # (B, char_embed_dim)
        bbox_emb = self.bbox_proj(bboxes)  # (B, char_embed_dim)
        cond = torch.cat([style_vec, char_emb, bbox_emb], dim=-1)
        cond_emb = self.cond_proj(cond)  # (B, time_emb_dim)
        cond_emb = cond_emb + t_emb  # 融合

        h = self.init_conv(x)
        skips = [h]

        # Down
        for layer in self.downs:
            if isinstance(layer, ResBlock1D):
                h = layer(h, cond_emb)
            else:
                h = layer(h)
            skips.append(h)

        # Middle
        h = self.mid1(h, cond_emb)
        h = self.mid_attn(h, cond_emb[:, None, :])  # 将cond作为context
        h = self.mid2(h, cond_emb)

        # Up
        for layer in self.ups:
            if isinstance(layer, ResBlock1D):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = layer(h, cond_emb)
            else:
                h = layer(h)

        return self.out(h)