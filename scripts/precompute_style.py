"""
6 进程并行 + 流式处理。

关键改动：
  1. N_WORKERS = 6
  2. worker 返回后立刻 GPU 前向，不累积原始 feats
  3. 每次缓冲 20 页做一次批处理，平衡内存和 GPU 利用率
"""
import os
import sys
import random
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import get_context

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from src.models.backbone import ConvNeXtBackbone
from src.data.loader import load_wptt_page_with_structure
from src.preprocessing.corner import detect_corners
from src.preprocessing.segmentation import pseudo_segment
from src.features.path_signature import (
    render_trajectory_to_bitmap, generate_path_signature_features)


N_CHARS_PER_PAGE = 30
N_WORKERS = 12
BATCH_PAGES = 20            # 缓冲多少页做一次 GPU 前向


# ============ Worker（模块级函数，可 pickle） ============
def worker_process_page(args):
    file_path, n_sample = args
    try:
        data = load_wptt_page_with_structure(file_path)
        page_pts = [l['points'] for l in data['lines'] if len(l['points']) > 0]
        if not page_pts:
            return file_path, None
        page_pts = np.concatenate(page_pts, axis=0)
        if len(page_pts) < 50:
            return file_path, None

        corners = detect_corners(page_pts, k=2, threshold=3)
        pseudo_chars = pseudo_segment(page_pts, corners)
        if len(pseudo_chars) == 0:
            return file_path, None

        rng = random.Random(hash(file_path) & 0xffffffff)
        if len(pseudo_chars) > n_sample:
            pseudo_chars = rng.sample(pseudo_chars, n_sample)

        feats = []
        for char_pts in pseudo_chars:
            try:
                bitmap, _ = render_trajectory_to_bitmap(char_pts)
                sig_maps = generate_path_signature_features(char_pts)
                b3 = np.expand_dims(bitmap, axis=-1)
                feat = np.concatenate([b3, sig_maps], axis=-1)
                feat = np.pad(feat, ((21, 21), (21, 21), (0, 0)),
                              mode='constant')
                feats.append(feat.transpose(2, 0, 1))
            except Exception:
                continue

        if not feats:
            return file_path, None
        return file_path, np.stack(feats, axis=0).astype(np.float32)
    except Exception:
        return file_path, None


# ============ 主流程 ============
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    model = ConvNeXtBackbone(
        num_classes=1019, in_chans=64, feature_dim=512, pretrained=False
    ).to(device)
    ckpt = torch.load('outputs/checkpoints/convnext_best.pth',
                      map_location=device)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.eval()
    print("已加载风格编码器")

    df = pd.read_csv('data/features/metadata_gen.csv', encoding='utf-8-sig')
    df['writer_id'] = df['writer_id'].astype(str)

    file_to_writer = dict(zip(df['file'], df['writer_id']))
    all_files = df['file'].tolist()
    print(f"共 {len(all_files)} 个页面, {df['writer_id'].nunique()} 位书写者")

    tasks = [(f, N_CHARS_PER_PAGE) for f in all_files]
    ctx = get_context('spawn')

    print(f"启动 {N_WORKERS} 进程（流式处理）...")
    file_to_vec = {}   # file_path -> (512,) tensor

    # 缓冲池：每次积累 BATCH_PAGES 页做一次 GPU 前向
    buffer_paths = []
    buffer_feats = []

    def flush():
        """把 buffer 里的页送 GPU，前向并平均。"""
        if not buffer_paths:
            return
        sizes = [f.shape[0] for f in buffer_feats]
        stacked = np.concatenate(buffer_feats, axis=0)
        with torch.no_grad():
            input_t = torch.from_numpy(stacked).to(device)
            vecs = model(input_t, return_features=True).cpu()
        idx = 0
        for p, n in zip(buffer_paths, sizes):
            file_to_vec[p] = vecs[idx:idx + n].mean(dim=0)
            idx += n
        buffer_paths.clear()
        buffer_feats.clear()
        del stacked, vecs, input_t
        torch.cuda.empty_cache()

    with ctx.Pool(processes=N_WORKERS) as pool:
        for file_path, feats in tqdm(
                pool.imap_unordered(worker_process_page, tasks, chunksize=4),
                total=len(tasks), desc="预处理+前向"):
            if feats is None:
                continue
            buffer_paths.append(file_path)
            buffer_feats.append(feats)

            if len(buffer_paths) >= BATCH_PAGES:
                flush()

        # 处理剩余
        flush()

    print(f"完成 {len(file_to_vec)} 个页面的特征提取")

    # 按书写者聚合
    writer_vecs = {}
    for file_path, vec in file_to_vec.items():
        wid = file_to_writer[file_path]
        writer_vecs.setdefault(wid, []).append(vec)

    style_vectors = {wid: torch.stack(v).mean(dim=0)
                     for wid, v in writer_vecs.items()}

    # 补齐缺失
    for wid in df['writer_id'].unique():
        if wid not in style_vectors:
            style_vectors[wid] = torch.zeros(512)

    out = 'data/features/style_vectors.pt'
    torch.save(style_vectors, out)
    print(f"已保存 {len(style_vectors)} 个书写者风格向量: {out}")

    # 覆盖检查
    for split in ['Train', 'Val', 'Test']:
        sub_writers = df[df['split'] == split]['writer_id'].unique()
        covered = sum(1 for w in sub_writers if w in style_vectors)
        print(f"  {split}: {covered}/{len(sub_writers)} 位风格向量已提取")


if __name__ == '__main__':
    main()