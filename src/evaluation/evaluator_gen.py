"""
评估流程：
  对测试集每位书写者 B：
    1. 取 B 的风格向量 z_B
    2. 取 B 写的一段文本对应的真实轨迹 Y_true
    3. 用模型生成 Y_hat（给定 z_B + 相同文本）
    4. 计算 DTW / Content Score / Style Score
"""
import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from src.models.layout_generator import LayoutGenerator
from src.models.diffusion_unet import DiffusionUNet1D
from src.models.style_adapter import StyleAdapter
from src.models.diffusion_sampler import ddim_sample
from src.data.loader import load_wptt_page_with_structure


def generate_for_text(text, style_vec, layout, unet, adapter, char2id,
                      device, line_spacing=1.5, steps=50):
    # Layout 分行
    ids = [char2id.get(c, 1) for c in text[:128]]
    ids += [0] * (128 - len(ids))
    ids_t = torch.tensor(ids).unsqueeze(0).to(device)
    sv = style_vec.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = layout(ids_t, sv)[0]
        probs = torch.sigmoid(logits).cpu().numpy()

    lines, cur = [], []
    for i, ch in enumerate(text):
        if i >= len(probs):
            break
        cur.append(ch)
        if probs[i] > 0.5:
            lines.append(''.join(cur)); cur = []
    if cur:
        lines.append(''.join(cur))

    # 逐行生成
    pts_all = []
    y_cursor = 0.0
    adapted = adapter(sv)
    for lt in lines:
        if not lt:
            y_cursor += line_spacing
            continue
        line_ids = [char2id.get(c, 1) for c in lt[:32]]
        line_ids += [0] * (32 - len(line_ids))
        lid = torch.tensor(line_ids).unsqueeze(0).to(device)
        traj = ddim_sample(unet, (1, 2, 256), adapted, lid,
                           device=device, steps=steps)[0].cpu().numpy().T
        traj[:, 1] += y_cursor
        pts_all.append(traj)
        pts_all.append(np.array([[-1.0, 0.0]], dtype=np.float32))
        y_cursor += line_spacing
    return np.concatenate(pts_all, axis=0)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    char2id = json.load(open('data/features/char_map.json', encoding='utf-8'))
    n = len(char2id)

    layout = LayoutGenerator(num_chars=n).to(device)
    unet = DiffusionUNet1D(style_dim=512, num_chars=n).to(device)
    adapter = StyleAdapter().to(device)
    layout.load_state_dict(torch.load(
        'outputs/checkpoints/layout_best.pth', map_location=device)['layout'])
    ckpt = torch.load('outputs/checkpoints/gen_best.pth', map_location=device)
    unet.load_state_dict(ckpt['diffusion_unet'])
    adapter.load_state_dict(ckpt['style_adapter'])
    layout.eval(); unet.eval(); adapter.eval()

    # 风格向量（含测试集）
    sv_dict = torch.load('data/features/style_vectors.pt', map_location='cpu')

    # 从测试集取书写者
    import pandas as pd
    df = pd.read_csv('data/features/metadata_gen.csv', encoding='utf-8-sig')
    df['writer_id'] = df['writer_id'].astype(str)
    test_df = df[df['split'] == 'Test']

    results = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="评估"):
        wid = row['writer_id']
        file = row['file']
        if wid not in sv_dict:
            continue

        # 取真实页的文本和轨迹
        try:
            data = load_wptt_page_with_structure(file)
        except Exception:
            continue
        full_text = ''.join(
            ''.join(c for c in l['text'] if c in char2id)
            for l in data['lines'])[:128]
        if not full_text:
            continue

        # 真实轨迹
        true_pts = np.concatenate(
            [l['points'] for l in data['lines']], axis=0)

        # 生成
        gen_pts = generate_for_text(
            full_text, sv_dict[wid], layout, unet, adapter,
            char2id, device)

        # TODO: 计算 DTW / Content Score / Style Score
        # 这里先占位
        results.append({
            'writer_id': wid,
            'text': full_text,
            'n_true': len(true_pts),
            'n_gen': len(gen_pts),
        })

    print(f"\n完成 {len(results)} 位测试书写者")
    print(f"示例: {results[0] if results else '无'}")