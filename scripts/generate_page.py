"""
整页生成：用户指定文本行（或每行字数），模型逐行生成。

用法:
    python scripts/generate_page.py
"""
import os
import sys
import json
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from src.models.diffusion_unet import DiffusionUNet1D
from src.models.style_adapter import StyleAdapter
from src.models.diffusion_sampler import ddim_sample


def encode(text, char2id, max_len=32):
    ids = [char2id.get(c, 1) for c in text[:max_len]]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids).unsqueeze(0)


def generate_line(unet, adapter, text, style_vec, char2id, device,
                  seq_len=256, steps=50):
    """生成一行归一化轨迹。"""
    ids = encode(text, char2id, 32).to(device)
    sv = style_vec.unsqueeze(0).to(device)
    adapted = adapter(sv)
    traj = ddim_sample(unet, (1, 2, seq_len), adapted, ids,
                       device=device, steps=steps)[0].cpu().numpy()
    return traj.T   # (L, 2)


def auto_wrap(text, chars_per_line, break_chars='，。！？；：'):
    """按每行字数硬分，尽量在标点后换行。"""
    lines, cur = [], ''
    for ch in text:
        cur += ch
        if ch in break_chars and len(cur) >= chars_per_line * 0.7:
            lines.append(cur)
            cur = ''
        elif len(cur) >= chars_per_line:
            lines.append(cur)
            cur = ''
    if cur:
        lines.append(cur)
    return lines


def compose_page(lines, style_vec, unet, adapter, char2id, device,
                 line_spacing=1.5, steps=50):
    """逐行生成，按顺序拼接成整页。"""
    pts_all = []
    y_cursor = 0.0
    for lt in lines:
        if not lt:
            y_cursor += line_spacing
            continue
        pts = generate_line(unet, adapter, lt, style_vec, char2id,
                            device, steps=steps)
        pts[:, 1] += y_cursor
        pts_all.append(pts)
        pts_all.append(np.array([[-1.0, 0.0]], dtype=np.float32))
        y_cursor += line_spacing
    return np.concatenate(pts_all, axis=0)


def visualize(points, out_path):
    """把点序列画成图（方便肉眼检查）。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    except ImportError:
        print("matplotlib 未安装，跳过可视化")
        return

    # 拆分成笔画
    segments = []
    cur = []
    for p in points:
        if p[0] == -1:
            if cur:
                segments.append(np.array(cur))
            cur = []
        else:
            cur.append(p)
    if cur:
        segments.append(np.array(cur))

    fig, ax = plt.subplots(figsize=(8, 12))
    for seg in segments:
        ax.plot(seg[:, 0], -seg[:, 1], 'k-', linewidth=1)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化: {out_path}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    char2id = json.load(open('data/features/char_map.json', encoding='utf-8'))
    n = len(char2id)

    # 加载模型
    unet = DiffusionUNet1D(style_dim=512, num_chars=n).to(device)
    adapter = StyleAdapter().to(device)
    ckpt = torch.load('outputs/checkpoints/gen_best.pth', map_location=device)
    unet.load_state_dict(ckpt['diffusion_unet'])
    adapter.load_state_dict(ckpt['style_adapter'])
    unet.eval()
    adapter.eval()
    print(f"已加载模型 (epoch={ckpt.get('epoch', '?')})")

    # 风格向量
    sv_dict = torch.load('data/features/style_vectors.pt', map_location='cpu')
    wid = list(sv_dict.keys())[0]
    style_vec = sv_dict[wid]
    print(f"使用风格: 书写者 {wid}")

    # ============ 用户输入 ============
    # 方式 1：直接给分好的行
    lines = [
        '落霞与孤鹜齐飞',
        '秋水共长天一色',
    ]

    # 方式 2：给一段文本 + 每行字数（自动分行）
    # text = '落霞与孤鹜齐飞秋水共长天一色'
    # lines = auto_wrap(text, chars_per_line=7)
    # ===================================

    print(f"\n分行方案 ({len(lines)} 行):")
    for i, lt in enumerate(lines):
        print(f"  行{i+1}: {lt}")

    # 生成
    page_pts = compose_page(
        lines, style_vec, unet, adapter, char2id, device,
        line_spacing=1.5, steps=50)

    # 保存
    out_dir = 'outputs/generated'
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'page.npy'), page_pts)
    print(f"\n保存 {len(page_pts)} 个点到 {out_dir}/page.npy")

    # 可视化
    visualize(page_pts, os.path.join(out_dir, 'page.png'))