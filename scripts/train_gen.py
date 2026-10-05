"""
训练入口：只训练 Line Diffusion。
"""
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from src.data.line_dataset import LineDataset
from src.training.trainer_gen import train_gen


def load_and_flatten_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    flat = {}
    for key in ['model', 'data', 'training', 'checkpoint', 'logging']:
        flat.update(cfg.get(key, {}))

    int_fields = ['seq_len', 'max_line_len', 'batch_size', 'num_workers',
                  'line_epochs', 'eval_frequency', 'patience']
    float_fields = ['line_lr', 'weight_decay']
    for k in int_fields:
        if k in flat:
            flat[k] = int(flat[k])
    for k in float_fields:
        if k in flat:
            flat[k] = float(flat[k])
    return flat


def main():
    config_path = os.path.join(BASE_DIR, 'configs', 'diffusion.yaml')
    flat = load_and_flatten_config(config_path)

    print("=" * 60)
    print("配置确认：")
    for k in ['seq_len', 'max_line_len', 'batch_size', 'num_workers',
              'line_epochs', 'line_lr', 'weight_decay',
              'eval_frequency', 'patience', 'line_save_path']:
        print(f"  {k} = {flat.get(k)}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    style_vectors = torch.load(
        os.path.join(BASE_DIR, 'data', 'features', 'style_vectors.pt'),
        map_location='cpu')
    print(f"加载 {len(style_vectors)} 位书写者风格向量")

    metadata_file = os.path.join(BASE_DIR, 'data', 'features', 'metadata_gen.csv')
    char_map_file = os.path.join(BASE_DIR, 'data', 'features', 'char_map.json')

    print("\n构建 Line 数据集...")
    line_train = LineDataset(
        metadata_file=metadata_file, split='Train',
        char_map_file=char_map_file, style_vectors=style_vectors,
        max_line_len=flat['max_line_len'], seq_len=flat['seq_len'])
    line_val = LineDataset(
        metadata_file=metadata_file, split='Val',
        char_map_file=char_map_file, style_vectors=style_vectors,
        max_line_len=flat['max_line_len'], seq_len=flat['seq_len'])

    num_chars = len(line_train.char2id)
    print(f"字符表大小: {num_chars}")

    bs, nw = flat['batch_size'], flat['num_workers']
    kw = dict(num_workers=nw, pin_memory=True)
    line_train_loader = DataLoader(line_train, bs, shuffle=True, **kw)
    line_val_loader = DataLoader(line_val, bs, shuffle=False, **kw)

    train_gen(flat, line_train_loader, line_val_loader, num_chars, device)


if __name__ == '__main__':
    main()