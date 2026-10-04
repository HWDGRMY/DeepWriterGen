"""
预计算所有书写者的风格向量，保存为 style_vectors.pt。
使用已训练好的 ConvNeXtBackbone 提取特征。
"""
import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.backbone import ConvNeXtBackbone
from src.data.loader import load_wptt_page_with_structure
from src.preprocessing.corner import detect_corners
from src.preprocessing.segmentation import pseudo_segment
from src.features.path_signature import render_trajectory_to_bitmap, generate_path_signature_features

def extract_style_vector(model, page_points, device):
    """从一页轨迹中提取风格向量（平均所有伪字符的特征）"""
    corners = detect_corners(page_points, k=2, threshold=3)
    pseudo_chars = pseudo_segment(page_points, corners)
    if len(pseudo_chars) == 0:
        return torch.zeros(512)
    feats = []
    for char_pts in pseudo_chars:
        # 生成特征图 (64, 96, 96)
        bitmap, norm_points = render_trajectory_to_bitmap(char_pts)
        sig_maps = generate_path_signature_features(char_pts)
        # 拼接位图和签名特征
        bitmap_3d = np.expand_dims(bitmap, axis=-1)  # (54,54,1)
        feat = np.concatenate([bitmap_3d, sig_maps], axis=-1)  # (54,54,64)
        # 填充到 96x96
        feat = np.pad(feat, ((21,21),(21,21),(0,0)), mode='constant')
        feat_tensor = torch.from_numpy(feat).permute(2,0,1).float().unsqueeze(0).to(device)
        with torch.no_grad():
            vec = model(feat_tensor, return_features=True)  # (1, 512)
        feats.append(vec.cpu())
    return torch.stack(feats).mean(dim=0).squeeze(0)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvNeXtBackbone(num_classes=1019, in_chans=64, feature_dim=512, pretrained=False).to(device)
    ckpt = torch.load('outputs/checkpoints/convnext_best.pth', map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    metadata = 'data/features/metadata.csv'
    df = pd.read_csv(metadata, encoding='utf-8-sig')
    df['writer_id'] = df['writer_id'].astype(str)
    writer_pages = df.groupby('writer_id')['file'].apply(list).to_dict()

    style_vectors = {}
    for wid, files in tqdm(writer_pages.items(), desc="提取风格向量"):
        vecs = []
        for f in files[:3]:  # 只用前3页作为风格参考
            try:
                data = load_wptt_page_with_structure(f)
                vec = extract_style_vector(model, data['all_points'], device)
                vecs.append(vec)
            except:
                continue
        if vecs:
            style_vectors[wid] = torch.stack(vecs).mean(dim=0)
        else:
            style_vectors[wid] = torch.zeros(512)

    torch.save(style_vectors, 'data/features/style_vectors.pt')
    print(f"已保存 {len(style_vectors)} 个书写者的风格向量")