"""Train the multimodal Gated Cross-Attention price model (notebook cell 1).

Reference port of the original Colab training run that produced
``artifacts/advanced_price_model.pt``. Expects a CSV with ``image_encoding``
and ``text_encoding`` columns (JSON-encoded 512-dim vectors) plus ``price``.

Usage:
    python scripts/train_multimodal.py --train-csv path/to/train_combined.csv
"""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pricing.models import AdvancedPriceModel

REPO_ROOT = Path(__file__).resolve().parents[1]


def safe_parse(x):
    if pd.isna(x):
        return None
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            try:
                return ast.literal_eval(x)
            except Exception:
                return None
    if isinstance(x, (list, np.ndarray)):
        return x
    return None


class PriceDataset(Dataset):
    def __init__(self, X_img, X_txt, y=None, augment=False):
        self.X_img = torch.FloatTensor(X_img)
        self.X_txt = torch.FloatTensor(X_txt)
        self.y = torch.FloatTensor(y) if y is not None else None
        self.augment = augment

    def __len__(self):
        return len(self.X_img)

    def __getitem__(self, idx):
        img, txt = self.X_img[idx], self.X_txt[idx]

        # Gaussian-noise augmentation for robustness to encoding perturbations
        if self.augment and self.y is not None:
            if np.random.rand() > 0.5:
                img = img + torch.randn_like(img) * 0.01
                txt = txt + torch.randn_like(txt) * 0.01

        if self.y is not None:
            return img, txt, self.y[idx]
        return img, txt


class AdaptiveLoss(nn.Module):
    """Weighted blend of MSE and MAE (alpha = 0.7)."""

    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, pred, target):
        return self.alpha * self.mse(pred, target) + (1 - self.alpha) * self.mae(pred, target)


def msape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-10
    return 100 / len(y_true) * np.sum(np.abs(y_true - y_pred) / denominator)


def train_model_advanced(model, train_loader, val_loader, device, epochs=250, lr_max=3e-4,
                         patience=40, warmup_epochs=15, log_interval=5, min_epochs=80):
    criterion = AdaptiveLoss(alpha=0.7)
    optimizer = optim.AdamW(model.parameters(), lr=lr_max, weight_decay=5e-5, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-7
    )

    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = min_epochs
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=1e-5)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_msape = float("inf")
    best_model_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": [], "val_msape": [], "lr": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        if epoch < warmup_epochs:
            current_lr = lr_max * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

        for X_img_batch, X_txt_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            X_img_batch = X_img_batch.to(device)
            X_txt_batch = X_txt_batch.to(device)
            y_batch = y_batch.to(device)

            # Mixup augmentation (randomly blend samples)
            if np.random.rand() > 0.7 and epoch > warmup_epochs:
                lam = np.random.beta(0.2, 0.2)
                rand_index = torch.randperm(X_img_batch.size(0)).to(device)
                X_img_batch = lam * X_img_batch + (1 - lam) * X_img_batch[rand_index]
                X_txt_batch = lam * X_txt_batch + (1 - lam) * X_txt_batch[rand_index]
                y_batch = lam * y_batch + (1 - lam) * y_batch[rand_index]

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(X_img_batch, X_txt_batch)
                loss = criterion(outputs, y_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_img_batch, X_txt_batch, y_batch in val_loader:
                X_img_batch = X_img_batch.to(device)
                X_txt_batch = X_txt_batch.to(device)
                y_batch = y_batch.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(X_img_batch, X_txt_batch)
                    loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(y_batch.cpu().numpy())

        val_loss /= len(val_loader)
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_msape_score = msape(np.expm1(val_targets), np.expm1(val_preds))

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_msape"].append(val_msape_score)
        history["lr"].append(current_lr)

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Val MSAPE={val_msape_score:.4f}, LR={current_lr:.6f}")

        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        elif epoch >= warmup_epochs:
            scheduler.step()

        if val_msape_score < best_val_msape and epoch >= min_epochs:
            best_val_msape = val_msape_score
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            print(f"  New best model! MSAPE: {best_val_msape:.4f}")
        else:
            wait += 1
            if wait >= patience and epoch >= min_epochs:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
    print(f"\nBest model MSAPE: {best_val_msape:.4f}")
    return model, swa_model, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "advanced_price_model.pt"))
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    df = pd.read_csv(args.train_csv).dropna()
    df["image_encoding"] = df["image_encoding"].apply(safe_parse)
    df["text_encoding"] = df["text_encoding"].apply(safe_parse)
    df = df.dropna(subset=["image_encoding", "text_encoding"])

    X_image = np.array(list(df["image_encoding"].values), dtype=np.float32)
    X_text = np.array(list(df["text_encoding"].values), dtype=np.float32)

    scaler_img = RobustScaler()
    scaler_txt = RobustScaler()
    X_image = scaler_img.fit_transform(X_image)
    X_text = scaler_txt.fit_transform(X_text)

    y_log = np.log1p(df["price"].values.astype(np.float32))

    X_img_train, X_img_val, X_txt_train, X_txt_val, y_train, y_val = \
        train_test_split(X_image, X_text, y_log, test_size=0.12, random_state=42, shuffle=True)

    batch_size = 128
    train_loader = DataLoader(
        PriceDataset(X_img_train, X_txt_train, y_train, augment=True),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        PriceDataset(X_img_val, X_txt_val, y_val),
        batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=True
    )

    config = {
        "dim_img": X_image.shape[1],
        "dim_txt": X_text.shape[1],
        "hidden_dims": [1536, 768, 384],
        "dropout": 0.15,
        "num_heads": 12,
    }
    model = AdvancedPriceModel(**config).to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    model, swa_model, history = train_model_advanced(model, train_loader, val_loader, device)

    torch.save({
        "model_state_dict": model.state_dict(),
        "swa_model_state_dict": swa_model.state_dict(),
        "scaler_img": scaler_img,
        "scaler_txt": scaler_txt,
        "history": history,
        "config": config,
    }, args.out)
    print(f"Model saved to {args.out}")


if __name__ == "__main__":
    main()
