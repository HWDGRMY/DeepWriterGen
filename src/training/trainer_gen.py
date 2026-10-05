import torch
import torch.nn as nn
from tqdm import tqdm
from src.models.diffusion_unet import DiffusionUNet1D
from src.models.style_adapter import StyleAdapter


def train_line(unet, adapter, train_loader, val_loader,
               epochs, lr, wd, eval_freq, patience, device, save_path):
    opt = torch.optim.AdamW(
        list(unet.parameters()) + list(adapter.parameters()),
        lr=lr, weight_decay=wd)
    mse = nn.MSELoss()
    betas = torch.linspace(1e-4, 0.02, 1000, device=device)
    acp = torch.cumprod(1 - betas, dim=0)
    best = float('inf')
    wait = 0

    for epoch in range(1, epochs + 1):
        unet.train()
        adapter.train()
        total = 0.0
        for b in tqdm(train_loader, desc=f"[Line] E{epoch}"):
            sv = b['style_vec'].to(device)
            ids = b['text_ids'].to(device)
            pts = b['line_points'].to(device)          # (B, L, 2)
            x0 = pts.permute(0, 2, 1).contiguous()     # (B, 2, L)
            B = x0.size(0)
            t = torch.randint(0, 1000, (B,), device=device).long()
            noise = torch.randn_like(x0)
            x_t = (torch.sqrt(acp[t])[:, None, None] * x0 +
                   torch.sqrt(1 - acp[t])[:, None, None] * noise)
            pred = unet(x_t, t, adapter(sv), ids)
            loss = mse(pred, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"[Line] E{epoch} train: {total / len(train_loader):.4f}")

        if epoch % eval_freq == 0:
            unet.eval()
            adapter.eval()
            v = 0.0
            with torch.no_grad():
                for b in val_loader:
                    sv = b['style_vec'].to(device)
                    ids = b['text_ids'].to(device)
                    x0 = b['line_points'].to(device).permute(0, 2, 1).contiguous()
                    B = x0.size(0)
                    t = torch.randint(0, 1000, (B,), device=device).long()
                    noise = torch.randn_like(x0)
                    x_t = (torch.sqrt(acp[t])[:, None, None] * x0 +
                           torch.sqrt(1 - acp[t])[:, None, None] * noise)
                    pred = unet(x_t, t, adapter(sv), ids)
                    v += mse(pred, noise).item()
            v /= len(val_loader)
            print(f"[Line] E{epoch} val:   {v:.4f}")
            if v < best:
                best = v
                wait = 0
                torch.save({
                    'diffusion_unet': unet.state_dict(),
                    'style_adapter': adapter.state_dict(),
                    'epoch': epoch,
                }, save_path)
                print(f"[Line] 保存最佳模型 (val={v:.4f})")
            else:
                wait += 1
                if wait >= patience:
                    print("[Line] 早停")
                    break


def train_gen(config, line_train_loader, line_val_loader,
              num_chars, device):
    print("=" * 60)
    print("训练 Line Generator (1D U-Net Diffusion)")
    print("=" * 60)

    unet = DiffusionUNet1D(style_dim=512, num_chars=num_chars).to(device)
    adapter = StyleAdapter().to(device)

    total_params = sum(p.numel() for p in unet.parameters())
    print(f"Line 模型参数量: {total_params / 1e6:.2f}M")

    train_line(
        unet, adapter,
        line_train_loader, line_val_loader,
        epochs=config['line_epochs'],
        lr=config['line_lr'],
        wd=config['weight_decay'],
        eval_freq=config['eval_frequency'],
        patience=config['patience'],
        device=device,
        save_path=config['line_save_path'],
    )