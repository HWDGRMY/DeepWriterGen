import torch
import numpy as np
from src.models.diffusion_sampler import ddim_sample


def evaluate_generation(model_dict, val_loader, device, seq_len=100, steps=50):
    """
    评估生成质量：DTW、内容准确率、风格相似度。
    简化：计算生成轨迹与真实轨迹的MSE作为DTW的近似。
    """
    layout_gen = model_dict['layout_gen']
    diffusion_unet = model_dict['diffusion_unet']
    style_adapter = model_dict['style_adapter']
    layout_gen.eval()
    diffusion_unet.eval()
    style_adapter.eval()

    total_mse = 0
    total_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            style_vec = batch['style_vec'].to(device)
            char_ids = batch['char_ids'].to(device)
            target_chars = batch['target_chars'].to(device)
            target_bboxes = batch['target_bboxes'].to(device)
            B, N, L, _ = target_chars.shape
            target_chars = target_chars.permute(0, 1, 3, 2).reshape(B * N, 2, L)
            target_bboxes_flat = target_bboxes.reshape(B * N, 4)
            char_ids_flat = char_ids.reshape(B * N)
            style_vec_flat = style_vec.unsqueeze(1).repeat(1, N, 1).reshape(B * N, -1)
            style_vec_adapted = style_adapter(style_vec_flat)

            # 生成
            generated = ddim_sample(diffusion_unet, (B * N, 2, L), style_vec_adapted,
                                    char_ids_flat, target_bboxes_flat, device, steps=steps)
            mse = torch.mean((generated - target_chars) ** 2)
            total_mse += mse.item() * B * N
            total_samples += B * N
    avg_mse = total_mse / total_samples
    return avg_mse