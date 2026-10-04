import torch
import torch.nn as nn
from tqdm import tqdm
from src.models.layout_generator import LayoutGenerator
from src.models.diffusion_unet import DiffusionUNet1D
from src.models.style_adapter import StyleAdapter

def train_gen(config, train_loader, val_loader, num_chars, device):
    layout_gen = LayoutGenerator(num_chars=num_chars).to(device)
    diffusion_unet = DiffusionUNet1D(style_dim=512, char_embed_dim=128).to(device)
    style_adapter = StyleAdapter().to(device)

    optimizer = torch.optim.AdamW(list(layout_gen.parameters()) +
                                  list(diffusion_unet.parameters()) +
                                  list(style_adapter.parameters()),
                                  lr=config['lr'], weight_decay=config['weight_decay'])
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, config['epochs']+1):
        layout_gen.train()
        diffusion_unet.train()
        style_adapter.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            style_vec = batch['style_vec'].to(device)
            char_ids = batch['char_ids'].to(device)
            target_chars = batch['target_chars'].to(device)
            target_bboxes = batch['target_bboxes'].to(device)

            B, N, L, _ = target_chars.shape
            target_chars = target_chars.permute(0,1,3,2).reshape(B*N, 2, L)
            target_bboxes_flat = target_bboxes.reshape(B*N, 4)
            char_ids_flat = char_ids.reshape(B*N)
            style_vec_flat = style_vec.unsqueeze(1).repeat(1, N, 1).reshape(B*N, -1)
            style_vec_adapted = style_adapter(style_vec_flat)

            # 布局损失
            pred_bboxes = layout_gen(char_ids)
            loss_layout = l1_loss(pred_bboxes, target_bboxes)

            # 扩散损失
            t = torch.randint(0, 1000, (B*N,), device=device).long()
            noise = torch.randn_like(target_chars)
            betas = torch.linspace(1e-4, 0.02, 1000, device=device)
            alphas = 1 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            sqrt_alpha = torch.sqrt(alphas_cumprod[t])[:,None,None]
            sqrt_one_minus = torch.sqrt(1 - alphas_cumprod[t])[:,None,None]
            x_t = sqrt_alpha * target_chars + sqrt_one_minus * noise
            pred_noise = diffusion_unet(x_t, t, style_vec_adapted, char_ids_flat, target_bboxes_flat)
            loss_diff = mse_loss(pred_noise, noise)

            loss = loss_diff + 0.1 * loss_layout
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} train loss: {avg_loss:.4f}")

        # 验证
        if epoch % config['eval_frequency'] == 0:
            layout_gen.eval()
            diffusion_unet.eval()
            style_adapter.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    style_vec = batch['style_vec'].to(device)
                    char_ids = batch['char_ids'].to(device)
                    target_chars = batch['target_chars'].to(device)
                    target_bboxes = batch['target_bboxes'].to(device)
                    B, N, L, _ = target_chars.shape
                    target_chars = target_chars.permute(0,1,3,2).reshape(B*N, 2, L)
                    target_bboxes_flat = target_bboxes.reshape(B*N, 4)
                    char_ids_flat = char_ids.reshape(B*N)
                    style_vec_flat = style_vec.unsqueeze(1).repeat(1, N, 1).reshape(B*N, -1)
                    style_vec_adapted = style_adapter(style_vec_flat)
                    t = torch.randint(0, 1000, (B*N,), device=device).long()
                    noise = torch.randn_like(target_chars)
                    betas = torch.linspace(1e-4, 0.02, 1000, device=device)
                    alphas = 1 - betas
                    alphas_cumprod = torch.cumprod(alphas, dim=0)
                    sqrt_alpha = torch.sqrt(alphas_cumprod[t])[:,None,None]
                    sqrt_one_minus = torch.sqrt(1 - alphas_cumprod[t])[:,None,None]
                    x_t = sqrt_alpha * target_chars + sqrt_one_minus * noise
                    pred_noise = diffusion_unet(x_t, t, style_vec_adapted, char_ids_flat, target_bboxes_flat)
                    val_loss += mse_loss(pred_noise, noise).item()
            val_loss /= len(val_loader)
            print(f"Epoch {epoch} val loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'layout_gen': layout_gen.state_dict(),
                    'diffusion_unet': diffusion_unet.state_dict(),
                    'style_adapter': style_adapter.state_dict(),
                }, config['save_best'])
                print("保存最佳模型")
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    print("早停")
                    break