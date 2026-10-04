import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.page_dataset import PageDataset
from src.models.layout_generator import LayoutGenerator
from src.models.diffusion_unet import DiffusionUNet1D
from src.models.style_adapter import StyleAdapter
from src.training.trainer_gen import train_gen

if __name__ == '__main__':
    with open('configs/diffusion.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载预计算的风格向量
    style_vectors = torch.load('data/features/style_vectors.pt', map_location='cpu')

    train_dataset = PageDataset('data/features/metadata.csv', split='Train',
                                char_map_file='data/features/char_map.json',
                                style_vectors=style_vectors)
    val_dataset = PageDataset('data/features/metadata.csv', split='Test',
                              char_map_file='data/features/char_map.json',
                              style_vectors=style_vectors)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'], pin_memory=True)

    train_gen(config, train_loader, val_loader, len(train_dataset.char2id), device)