import torch


def ddim_sample(model, shape, style_vec, char_ids, bboxes, device, steps=50, eta=0.0):
    """
    DDIM 采样生成轨迹。
    """
    B, C, L = shape
    x = torch.randn(shape, device=device)
    # 时间步序列
    timesteps = torch.linspace(0, 1, steps + 1, device=device)
    # 简化实现：使用线性beta调度
    betas = torch.linspace(1e-4, 0.02, 1000, device=device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    for i in reversed(range(steps)):
        t = timesteps[i + 1] * 1000
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        pred_noise = model(x, t_tensor, style_vec, char_ids, bboxes)
        alpha = alphas_cumprod[t_tensor.long()]
        alpha_prev = alphas_cumprod[torch.clamp(t_tensor - 1, min=0).long()]
        # DDIM 更新
        pred_x0 = (x - torch.sqrt(1 - alpha)[:, None, None] * pred_noise) / torch.sqrt(alpha)[:, None, None]
        pred_x0 = torch.clamp(pred_x0, -1, 1)
        if i > 0:
            noise = torch.randn_like(x) if eta > 0 else 0
            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
            x = torch.sqrt(alpha_prev)[:, None, None] * pred_x0 + \
                torch.sqrt(1 - alpha_prev - sigma ** 2)[:, None, None] * pred_noise + \
                sigma[:, None, None] * noise
        else:
            x = pred_x0
    return x