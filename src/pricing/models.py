"""Multimodal price-prediction network from the Amazon ML Challenge 2025 solution.

Architecture is kept identical (including attribute names) to the training
notebook so that ``advanced_price_model.pt`` checkpoints load without key
remapping. See docs/AMAZON_ML.pdf for the full writeup.
"""

import torch
import torch.nn as nn


class GatedCrossAttention(nn.Module):
    """Multi-head attention where image features attend to text embeddings;
    a sigmoid gate regulates how much fused signal flows forward."""

    def __init__(self, dim_img, dim_txt, dim_hidden, num_heads=8, dropout=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.dim_hidden = dim_hidden
        self.head_dim = dim_hidden // num_heads

        self.query_img = nn.Linear(dim_img, dim_hidden)
        self.key_txt = nn.Linear(dim_txt, dim_hidden)
        self.value_txt = nn.Linear(dim_txt, dim_hidden)
        self.key_img = nn.Linear(dim_img, dim_hidden)
        self.value_img = nn.Linear(dim_img, dim_hidden)

        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        self.gate = nn.Sequential(
            nn.Linear(dim_img + dim_txt, dim_hidden),
            nn.Sigmoid()
        )

        self.out_proj = nn.Linear(dim_hidden * 2, dim_hidden)
        self.layer_norm = nn.LayerNorm(dim_hidden)

    def forward(self, img, txt):
        batch_size = img.size(0)

        # Cross-attention: image queries attend to text
        Q_img = self.query_img(img).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K_txt = self.key_txt(txt).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V_txt = self.value_txt(txt).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_txt = torch.softmax(torch.matmul(Q_img, K_txt.transpose(-2, -1)) * self.scale, dim=-1)
        attn_txt = self.dropout(attn_txt)
        cross_txt = torch.matmul(attn_txt, V_txt).transpose(1, 2).contiguous().view(batch_size, self.dim_hidden)

        # Self-attention on image
        K_img = self.key_img(img).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V_img = self.value_img(img).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_img = torch.softmax(torch.matmul(Q_img, K_img.transpose(-2, -1)) * self.scale, dim=-1)
        attn_img = self.dropout(attn_img)
        self_img = torch.matmul(attn_img, V_img).transpose(1, 2).contiguous().view(batch_size, self.dim_hidden)

        # Gated fusion
        gate_input = torch.cat([img, txt], dim=-1)
        gate_weights = self.gate(gate_input)

        combined = torch.cat([cross_txt, self_img], dim=-1)
        fused = self.out_proj(combined)
        fused = gate_weights * fused

        return self.layer_norm(fused)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel recalibration."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # Squeeze
        w = torch.mean(x, dim=0, keepdim=True)
        # Excitation
        w = torch.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class ResidualBlock(nn.Module):
    """Dense block with GELU, LayerNorm, Dropout and SE recalibration."""

    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )
        self.se = SEBlock(dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = self.net(x)
        x = self.se(x)
        return self.layer_norm(x + residual)


class AdvancedPriceModel(nn.Module):
    """Gated cross-attention fusion of image/text embeddings with SE residual
    blocks and progressive downscaling; predicts log1p(price)."""

    def __init__(self, dim_img, dim_txt, hidden_dims=[1536, 768, 384], dropout=0.15, num_heads=12):
        super().__init__()

        self.img_proj = nn.Sequential(
            nn.Linear(dim_img, hidden_dims[0] // 2),
            nn.LayerNorm(hidden_dims[0] // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        self.txt_proj = nn.Sequential(
            nn.Linear(dim_txt, hidden_dims[0] // 2),
            nn.LayerNorm(hidden_dims[0] // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        self.cross_attn = GatedCrossAttention(
            hidden_dims[0] // 2,
            hidden_dims[0] // 2,
            hidden_dims[0],
            num_heads=num_heads,
            dropout=dropout
        )

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[0], dropout) for _ in range(3)
        ])

        self.down_blocks = nn.ModuleList()
        prev_dim = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            self.down_blocks.append(nn.Sequential(
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
            prev_dim = h_dim

        self.skip_connections = nn.ModuleList([
            nn.Linear(hidden_dims[0], hidden_dims[-1])
        ])

        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dims[-1] * 2, hidden_dims[-1]),
            nn.LayerNorm(hidden_dims[-1]),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.LayerNorm(hidden_dims[-1] // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )

    def forward(self, img, txt):
        img_feat = self.img_proj(img)
        txt_feat = self.txt_proj(txt)

        x = self.cross_attn(img_feat, txt_feat)
        skip = x

        for block in self.residual_blocks:
            x = block(x)

        for down in self.down_blocks:
            x = down(x)

        skip_transformed = self.skip_connections[0](skip)
        x = torch.cat([x, skip_transformed], dim=-1)

        out = self.prediction_head(x)
        return out.squeeze()
