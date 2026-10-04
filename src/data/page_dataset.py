import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from src.data.loader import load_wptt_page_with_structure
from src.data.layout_utils import resample_trajectory, compute_bbox_from_points

class PageDataset(Dataset):
    """
    整页轨迹数据集。每个样本是一页，包含：
    - style_vec: 该书写者的风格向量（预计算）
    - char_ids: 目标页的字符ID序列
    - target_chars: 目标页每个字符的轨迹点序列（重采样后）
    - target_bboxes: 目标页每个字符的 bbox
    """
    def __init__(self, metadata_file, split='Train', char_map_file='data/features/char_map.json',
                 seq_len=100, style_vectors=None):
        """
        style_vectors: dict {writer_id: torch.Tensor(512)}，预计算的风格向量。
                       若为 None，则使用零向量占位。
        """
        self.seq_len = seq_len
        self.style_vectors = style_vectors or {}

        import pandas as pd
        self.df = pd.read_csv(metadata_file, encoding='utf-8-sig')
        self.df['writer_id'] = self.df['writer_id'].astype(str)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.writer_ids = sorted(self.df['writer_id'].unique())

        with open(char_map_file, 'r', encoding='utf-8') as f:
            self.char2id = json.load(f)

        # 加载所有页的结构
        self.pages = []
        for _, row in self.df.iterrows():
            file_path = row['file']
            try:
                data = load_wptt_page_with_structure(file_path)
            except Exception as e:
                print(f"加载失败: {file_path}, {e}")
                continue
            char_instances = data['char_instances']
            if len(char_instances) == 0:
                continue
            valid = []
            for ci in char_instances:
                ch = ci['char']
                if ch in self.char2id:
                    valid.append(ci)
            if len(valid) == 0:
                continue
            target_chars = []
            target_bboxes = []
            for ci in valid:
                pts = ci['points']
                resampled = resample_trajectory(pts, self.seq_len)
                target_chars.append(resampled)
                h, w, v_center = compute_bbox_from_points(pts)
                h_offset = ci['bbox'][3]
                target_bboxes.append([h, w, v_center, h_offset])
            self.pages.append({
                'writer_id': row['writer_id'],
                'page': row['page'],
                'chars': [ci['char'] for ci in valid],
                'target_chars': np.array(target_chars, dtype=np.float32),  # (N, seq_len, 2)
                'target_bboxes': np.array(target_bboxes, dtype=np.float32),  # (N, 4)
            })

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, idx):
        page = self.pages[idx]
        writer_id = page['writer_id']
        style_vec = self.style_vectors.get(writer_id, torch.zeros(512))
        chars = page['chars']
        char_ids = torch.tensor([self.char2id[ch] for ch in chars], dtype=torch.long)
        target_chars = torch.from_numpy(page['target_chars'])  # (N, seq_len, 2)
        target_bboxes = torch.from_numpy(page['target_bboxes'])  # (N, 4)
        return {
            'style_vec': style_vec,
            'char_ids': char_ids,
            'target_chars': target_chars,
            'target_bboxes': target_bboxes,
            'writer_id': writer_id,
        }