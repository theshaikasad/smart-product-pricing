"""Stacked ensemble training + inference (notebook cell 11) — the best variant
from the challenge submission (validation MSAPE 37.21).

Builds image-only and text-only MLP heads, generates 5-fold OOF predictions
together with the pretrained multimodal model, engineers meta-features, trains
Ridge / GradientBoosting / meta-MLP stackers, blends them with inverse-MSAPE
weights, and runs inference on the test CSV.

Usage:
    python scripts/train_ensemble.py --train-csv train_combined.csv --test-csv test_combined.csv
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import dump
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pricing.models import AdvancedPriceModel

REPO_ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

CONFIG = {
    "epochs_uni": 100,
    "epochs_meta": 600,
    "lr_uni": 1e-3,
    "wd_uni": 1e-4,
    "lr_meta": 3e-4,
    "wd_meta": 5e-4,
    "batch_size": 256,
    "meta_dropout": 0.4,
    "patience": 15,
}


def msape(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return 100.0 * np.mean(np.abs(y_true - y_pred) / denom)


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


class ImprovedUniMLP(nn.Module):
    """Single-modality price head: MLP over one 512-dim embedding."""

    def __init__(self, dim, hidden=[1024, 512, 256]):
        super().__init__()
        layers = []
        prev = dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.2)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class EnhancedMetaMLP(nn.Module):
    """Stacker MLP over the engineered meta-features."""

    def __init__(self, input_dim, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout * 0.8),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout * 0.6),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_unimodal(model, Xtr, ylog_tr, Xva, ylog_va, epochs=100, lr=1e-3, wd=1e-4):
    tr = TensorDataset(torch.tensor(Xtr).float(), torch.tensor(ylog_tr).float())
    va = TensorDataset(torch.tensor(Xva).float(), torch.tensor(ylog_va).float())
    tl = DataLoader(tr, batch_size=CONFIG["batch_size"], shuffle=True)
    vl = DataLoader(va, batch_size=512, shuffle=False)

    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    best, best_s = float("inf"), None
    patience_cnt = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vals = []
        with torch.no_grad():
            for xb, yb in vl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vals.append(loss_fn(model(xb), yb).item())

        vm = float(np.mean(vals))
        if vm < best - 1e-6:
            best, best_s = vm, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= CONFIG["patience"]:
                break

    if best_s:
        model.load_state_dict(best_s)
    return model


@torch.no_grad()
def predict_uni_log(model, X):
    model.eval()
    outs = []
    X_t = torch.tensor(X).float()
    for i in range(0, len(X), 512):
        outs.append(model(X_t[i:i + 512].to(DEVICE)).cpu().numpy())
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def predict_multi_log(model, Xi, Xt):
    model.eval()
    outs = []
    Xi_t = torch.tensor(Xi).float()
    Xt_t = torch.tensor(Xt).float()
    for i in range(0, len(Xi), 512):
        outs.append(model(Xi_t[i:i + 512].to(DEVICE), Xt_t[i:i + 512].to(DEVICE)).cpu().numpy())
    return np.concatenate(outs, axis=0)


def create_advanced_meta_features(img_pred, txt_pred, mul_pred):
    """Meta-features: predictions, disagreements, log stubs, ratios, and
    order statistics across the three base heads (17 features)."""
    meta = [img_pred, txt_pred, mul_pred]

    meta.append(np.abs(img_pred - mul_pred))
    meta.append(np.abs(txt_pred - mul_pred))
    meta.append(np.abs(img_pred - txt_pred))

    meta.append(np.log1p(mul_pred))
    meta.append(np.log1p(np.abs(img_pred)))
    meta.append(np.log1p(np.abs(txt_pred)))

    safe_mul = np.maximum(np.abs(mul_pred), 1e-6)
    meta.append(img_pred / safe_mul)
    meta.append(txt_pred / safe_mul)

    all_preds = np.stack([img_pred, txt_pred, mul_pred], axis=1)
    meta.append(np.median(all_preds, axis=1))
    meta.append(np.var(all_preds, axis=1))
    meta.append(np.min(all_preds, axis=1))
    meta.append(np.max(all_preds, axis=1))
    meta.append(np.max(all_preds, axis=1) - np.min(all_preds, axis=1))

    return np.column_stack(meta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv")
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "artifacts" / "advanced_price_model.pt"))
    parser.add_argument("--artifact-dir", default=str(REPO_ROOT / "artifacts" / "ensemble"))
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print("Loading training data...")
    df = pd.read_csv(args.train_csv).dropna()
    drop_cols = [c for c in ["sample_id", "catalog_content", "image_link", "download_failed"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    df["image_encoding"] = df["image_encoding"].apply(safe_parse)
    df["text_encoding"] = df["text_encoding"].apply(safe_parse)

    X_img = np.asarray(df["image_encoding"].tolist(), dtype=np.float32)
    X_txt = np.asarray(df["text_encoding"].tolist(), dtype=np.float32)
    y_log = np.log1p(df["price"].values.astype(np.float32))

    print("Loading pretrained multimodal model...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    scaler_img = ckpt["scaler_img"]
    scaler_txt = ckpt["scaler_txt"]

    Xi_s = scaler_img.transform(X_img)
    Xt_s = scaler_txt.transform(X_txt)

    multi = AdvancedPriceModel(**config).to(DEVICE)
    multi.load_state_dict(ckpt["model_state_dict"])
    multi.eval()

    Xi_tr, Xi_va, Xt_tr, Xt_va, ylog_tr, ylog_va = train_test_split(
        Xi_s, Xt_s, y_log, test_size=0.15, random_state=SEED, shuffle=True
    )

    print("\nGenerating OOF predictions (5-fold)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_img, oof_txt, oof_mul, oof_y = [], [], [], []

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(Xi_tr)):
        print(f"  Fold {fold_idx + 1}/5...")
        img_m = train_unimodal(ImprovedUniMLP(Xi_tr.shape[1]), Xi_tr[tr_idx], ylog_tr[tr_idx],
                               Xi_tr[va_idx], ylog_tr[va_idx],
                               epochs=CONFIG["epochs_uni"], lr=CONFIG["lr_uni"], wd=CONFIG["wd_uni"])
        txt_m = train_unimodal(ImprovedUniMLP(Xt_tr.shape[1]), Xt_tr[tr_idx], ylog_tr[tr_idx],
                               Xt_tr[va_idx], ylog_tr[va_idx],
                               epochs=CONFIG["epochs_uni"], lr=CONFIG["lr_uni"], wd=CONFIG["wd_uni"])

        oof_img.append(np.expm1(predict_uni_log(img_m, Xi_tr[va_idx])))
        oof_txt.append(np.expm1(predict_uni_log(txt_m, Xt_tr[va_idx])))
        oof_mul.append(np.expm1(predict_multi_log(multi, Xi_tr[va_idx], Xt_tr[va_idx])))
        oof_y.append(np.expm1(ylog_tr[va_idx]))

    img_oof = np.concatenate(oof_img)
    txt_oof = np.concatenate(oof_txt)
    mul_oof = np.concatenate(oof_mul)
    y_oof = np.concatenate(oof_y)

    print("\nCreating meta features...")
    meta_X = create_advanced_meta_features(img_oof, txt_oof, mul_oof)
    meta_scaler = RobustScaler()
    meta_X_scaled = meta_scaler.fit_transform(meta_X)

    print("Training Ridge ensemble...")
    ridge = Ridge(alpha=100.0, fit_intercept=True)
    ridge.fit(meta_X_scaled, y_oof)
    print(f"OOF Ridge MSAPE: {msape(y_oof, ridge.predict(meta_X_scaled)):.3f}")

    print("Training GradientBoosting ensemble...")
    gb = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8,
        min_samples_split=5, min_samples_leaf=2, random_state=SEED,
        validation_fraction=0.1, n_iter_no_change=50
    )
    gb.fit(meta_X_scaled, y_oof)
    print(f"OOF GradientBoosting MSAPE: {msape(y_oof, gb.predict(meta_X_scaled)):.3f}")

    print("Training Meta-MLP...")
    meta_mlp = EnhancedMetaMLP(meta_X_scaled.shape[1], dropout=CONFIG["meta_dropout"]).to(DEVICE)
    opt = torch.optim.AdamW(meta_mlp.parameters(), lr=CONFIG["lr_meta"], weight_decay=CONFIG["wd_meta"])
    crit = nn.SmoothL1Loss()
    Xt_meta = torch.tensor(meta_X_scaled, dtype=torch.float32, device=DEVICE)
    yt_meta = torch.tensor(y_oof, dtype=torch.float32, device=DEVICE)

    best_loss, best_state, patience_cnt = float("inf"), None, 0
    for _ in tqdm(range(CONFIG["epochs_meta"]), desc="Train Meta-MLP"):
        opt.zero_grad()
        loss = crit(meta_mlp(Xt_meta).squeeze(-1), yt_meta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(meta_mlp.parameters(), 1.0)
        opt.step()
        if loss.item() < best_loss - 1e-7:
            best_loss = loss.item()
            best_state = {k: v.detach().cpu().clone() for k, v in meta_mlp.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt > 100:
                break
    if best_state:
        meta_mlp.load_state_dict(best_state)

    print("\nEvaluating on validation split...")
    img_full = train_unimodal(ImprovedUniMLP(Xi_s.shape[1]), Xi_s, y_log, Xi_va, ylog_va,
                              epochs=CONFIG["epochs_uni"], lr=CONFIG["lr_uni"], wd=CONFIG["wd_uni"])
    txt_full = train_unimodal(ImprovedUniMLP(Xt_s.shape[1]), Xt_s, y_log, Xt_va, ylog_va,
                              epochs=CONFIG["epochs_uni"], lr=CONFIG["lr_uni"], wd=CONFIG["wd_uni"])

    img_va = np.expm1(predict_uni_log(img_full, Xi_va))
    txt_va = np.expm1(predict_uni_log(txt_full, Xt_va))
    mul_va = np.expm1(predict_multi_log(multi, Xi_va, Xt_va))
    y_val = np.expm1(ylog_va)

    meta_val_s = meta_scaler.transform(create_advanced_meta_features(img_va, txt_va, mul_va))
    ridge_val = ridge.predict(meta_val_s)
    gb_val = gb.predict(meta_val_s)
    mlp_val = meta_mlp(torch.tensor(meta_val_s, dtype=torch.float32, device=DEVICE)).detach().cpu().numpy().squeeze()

    ms_ridge, ms_gb, ms_mlp = msape(y_val, ridge_val), msape(y_val, gb_val), msape(y_val, mlp_val)
    print(f"Holdout MSAPE — Ridge: {ms_ridge:.3f}, GB: {ms_gb:.3f}, Meta-MLP: {ms_mlp:.3f}")

    weights = np.array([1.0 / (ms_ridge + 1e-6), 1.0 / (ms_gb + 1e-6), 1.0 / (ms_mlp + 1e-6)])
    weights /= weights.sum()
    ensemble_val = weights[0] * ridge_val + weights[1] * gb_val + weights[2] * mlp_val
    print(f"Weighted Ensemble MSAPE: {msape(y_val, ensemble_val):.3f}")

    print("\nSaving artifacts...")
    torch.save(img_full.state_dict(), artifact_dir / "image_only_model.pt")
    torch.save(txt_full.state_dict(), artifact_dir / "text_only_model.pt")
    dump(meta_scaler, artifact_dir / "meta_scaler.joblib")
    dump(ridge, artifact_dir / "ridge_model.joblib")
    dump(gb, artifact_dir / "gb_model.joblib")
    torch.save(meta_mlp.state_dict(), artifact_dir / "meta_mlp.pt")
    with open(artifact_dir / "ensemble_config.json", "w") as f:
        json.dump({
            "weights": weights.tolist(),
            "validation_scores": {
                "ridge": float(ms_ridge), "gb": float(ms_gb), "mlp": float(ms_mlp),
                "weighted": float(msape(y_val, ensemble_val)),
            },
        }, f, indent=2)

    if not args.test_csv:
        print("No --test-csv given; done.")
        return

    print("\nRunning inference on test set...")
    test_df = pd.read_csv(args.test_csv)
    sample_ids = test_df["sample_id"].values if "sample_id" in test_df.columns else np.arange(len(test_df))
    test_df["image_encoding"] = test_df["image_encoding"].apply(safe_parse)
    test_df["text_encoding"] = test_df["text_encoding"].apply(safe_parse)

    def fill_missing(x, dim):
        return np.zeros(dim) if x is None else x

    img_dim = len(test_df["image_encoding"].dropna().iloc[0])
    txt_dim = len(test_df["text_encoding"].dropna().iloc[0])
    Xi_te = np.asarray([fill_missing(x, img_dim) for x in test_df["image_encoding"]], dtype=np.float32)
    Xt_te = np.asarray([fill_missing(x, txt_dim) for x in test_df["text_encoding"]], dtype=np.float32)

    Xi_te_s = scaler_img.transform(Xi_te)
    Xt_te_s = scaler_txt.transform(Xt_te)

    img_p = np.expm1(predict_uni_log(img_full, Xi_te_s))
    txt_p = np.expm1(predict_uni_log(txt_full, Xt_te_s))
    mul_p = np.expm1(predict_multi_log(multi, Xi_te_s, Xt_te_s))

    meta_test_s = meta_scaler.transform(create_advanced_meta_features(img_p, txt_p, mul_p))
    final = (weights[0] * ridge.predict(meta_test_s)
             + weights[1] * gb.predict(meta_test_s)
             + weights[2] * meta_mlp(torch.tensor(meta_test_s, dtype=torch.float32, device=DEVICE))
                 .detach().cpu().numpy().squeeze())
    final = np.maximum(final, 0.0)

    out_csv = artifact_dir / "ensemble_final_predictions.csv"
    pd.DataFrame({"sample_id": sample_ids, "price": final}).to_csv(out_csv, index=False)
    print(f"Predictions saved to {out_csv}")


if __name__ == "__main__":
    main()
