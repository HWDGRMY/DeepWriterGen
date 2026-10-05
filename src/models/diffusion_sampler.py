import torch


def ddim_sample(model, shape, style_vec, text_ids,
                device='cuda', steps=50, eta=0.0):
    betas = torch.linspace(1e-4, 0.02, 1000, device=device)
    alphas = 1 - betas
    acp = torch.cumprod(alphas, dim=0)
    x = torch.randn(shape, device=device)
    ts = torch.linspace(999, 0, steps + 1).long()

    for i in range(steps):
        t = ts[i].item()
        tn = ts[i + 1].item()
        tt = torch.full((shape[0],), t, device=device, dtype=torch.long)
        pred = model(x, tt, style_vec, text_ids)
        a = acp[t]
        ap = acp[tn] if tn > 0 else torch.tensor(1.0, device=device)
        x0 = (x - torch.sqrt(1 - a) * pred) / torch.sqrt(a)
        x0 = torch.clamp(x0, -1, 1)
        if tn > 0:
            if eta > 0:
                noise = torch.randn_like(x)
                sigma = eta * torch.sqrt((1 - ap) / (1 - a) *
                                         (1 - a / ap))
            else:
                noise = 0
                sigma = 0
            x = (torch.sqrt(ap) * x0 +
                 torch.sqrt(1 - ap - sigma ** 2) * pred + sigma * noise)
        else:
            x = x0
    return x