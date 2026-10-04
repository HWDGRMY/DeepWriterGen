import torch
import torch.nn as nn

class LayoutGenerator(nn.Module):
    def __init__(self, num_chars, char_embed_dim=128, hidden_dim=256, bbox_dim=4):
        super().__init__()
        self.char_embed = nn.Embedding(num_chars, char_embed_dim)
        self.lstm = nn.LSTM(char_embed_dim + bbox_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, bbox_dim)

    def forward(self, char_ids, bbox_prefix=None):
        """
        char_ids: (B, N) 字符ID序列
        bbox_prefix: (B, K, 4) 风格参考的bbox前缀，可选
        返回: (B, N, 4) 预测的bbox
        """
        B, N = char_ids.shape
        emb = self.char_embed(char_ids)  # (B, N, char_embed_dim)
        if bbox_prefix is not None:
            # 将前缀拼接到输入序列前
            prefix_emb = torch.zeros(B, bbox_prefix.size(1), emb.size(2), device=emb.device)
            # 这里简化：前缀不参与embedding，直接拼接? 实际应使用前缀的bbox作为输入
            pass
        # 简化：自回归生成，使用教师强制
        # 输入为字符嵌入 + 前一个bbox（初始为0）
        inputs = torch.cat([emb, torch.zeros(B, N, 4, device=emb.device)], dim=-1)  # (B, N, char_embed_dim+4)
        lstm_out, _ = self.lstm(inputs)  # (B, N, hidden)
        bboxes = self.fc(lstm_out)  # (B, N, 4)
        return bboxes