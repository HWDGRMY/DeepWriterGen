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
from src.evaluation.evaluator_gen import evaluate_generation

if __name__ == '__main__':
    with open('configs/diffusion.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    style_vectors = torch.load('data/features/style_vectors.pt', map_location='cpu')

    val_dataset = PageDataset('data/features/metadata.csv', split='Test',
                              char_map_file='data/features/char_map.json',
                              style_vectors=style_vectors)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'])

    num_chars = len(val_dataset.char2id)
    layout_gen = LayoutGenerator(num_chars=num_chars).to(device)
    diffusion_unet = DiffusionUNet1D(style_dim=512, char_embed_dim=128).to(device)
    style_adapter = StyleAdapter().to(device)

    ckpt = torch.load('outputs/checkpoints/gen_best.pth', map_location=device)
    layout_gen.load_state_dict(ckpt['layout_gen'])
    diffusion_unet.load_state_dict(ckpt['diffusion_unet'])
    style_adapter.load_state_dict(ckpt['style_adapter'])

    mse = evaluate_generation({'layout_gen': layout_gen, 'diffusion_unet': diffusion_unet,
                               'style_adapter': style_adapter}, val_loader, device)
    print(f"生成轨迹 MSE: {mse:.6f}")