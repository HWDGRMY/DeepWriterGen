"""
行级数据集。
每个样本 = 一行。
归一化：坐标 ÷ 该页字宽 → 一个字跨度 ≈ 1.0。
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from src.data.loader import load_wptt_page_with_structure


def resample(points, seq_len=256):
    pts = points[points[:, 0] != -1]
    if len(pts) < 3:
        return np.zeros((seq_len, 2), dtype=np.float32)
    diffs = np.diff(pts, axis=0)
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    cum = np.concatenate([[0], np.cumsum(dists)])
    total = cum[-1]
    if total == 0:
        return np.tile(pts[0:1], (seq_len, 1)).astype(np.float32)
    t = np.linspace(0, total, seq_len)
    return np.stack([np.interp(t, cum, pts[:, 0]),
                     np.interp(t, cum, pts[:, 1])],
                    axis=1).astype(np.float32)


def encode(text, char2id, max_len):
    ids = [char2id.get(ch, 1) for ch in text[:max_len]]
    ids += [0] * (max_len - len(ids))
    return np.array(ids, dtype=np.int64)


class LineDataset(Dataset):
    def __init__(self, metadata_file, split, char_map_file,
                 style_vectors, max_line_len=32, seq_len=256, min_len=2):
        self.max_line_len = max_line_len
        self.seq_len = seq_len
        self.style_vectors = style_vectors

        with open(char_map_file, 'r', encoding='utf-8') as f:
            self.char2id = json.load(f)

        import pandas as pd
        df = pd.read_csv(metadata_file, encoding='utf-8-sig')
        df['writer_id'] = df['writer_id'].astype(str)
        df = df[df['split'] == split].reset_index(drop=True)

        self.samples = []
        for _, row in df.iterrows():
            try:
                data = load_wptt_page_with_structure(row['file'])
            except Exception:
                continue
            wid = row['writer_id']

            # 整页 bbox + 平均字宽
            all_pts, texts = [], []
            for l in data['lines']:
                if len(l['points']) > 0:
                    all_pts.append(l['points'])
                if len(l['text']) > 0:
                    texts.append(l['text'])
            if not all_pts or not texts:
                continue
            all_pts = np.concatenate(all_pts, axis=0)
            m = all_pts[:, 0] != -1
            if m.sum() < 3:
                continue

            x_min = all_pts[m, 0].min()
            y_min = all_pts[m, 1].min()
            page_w = max(all_pts[m, 0].max() - x_min, 1.0)
            avg_chars = np.mean([len(t) for t in texts])
            char_w = page_w / max(avg_chars, 1.0)

            for l in data['lines']:
                text = l['text']
                pts = l['points']
                if len(text) < min_len or len(pts) < 20:
                    continue
                valid = [c for c in text if c in self.char2id
                         and c not in ('<pad>', '<unk>')]
                if len(valid) < min_len:
                    continue

                pts_n = pts.copy()
                mm = pts_n[:, 0] != -1
                pts_n[mm, 0] = (pts[mm, 0] - x_min) / char_w
                pts_n[mm, 1] = (pts[mm, 1] - y_min) / char_w

                self.samples.append({
                    'text': text,
                    'pts_norm': pts_n,
                    'writer_id': wid,
                })
        print(f"{split}: {len(self.samples)} 行")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'text_ids': torch.from_numpy(
                encode(s['text'], self.char2id, self.max_line_len)),
            'line_points': torch.from_numpy(
                resample(s['pts_norm'], self.seq_len)),
            'style_vec': self.style_vectors.get(s['writer_id'],
                                                torch.zeros(512)),
            'writer_id': s['writer_id'],
            'text': s['text'],
        }