import os
from typing import Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Global configuration -------------------------------------------------------
# ---------------------------------------------------------------------------

C: int = 12  # Sparsity factor
ANGLE_STEP: int = 360 // C  # Angular distance between successive projections

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Attention with positional encodings (β, θ) --------------------------------
# ---------------------------------------------------------------------------

class AttentionPos(nn.Module):

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int = 4,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
    ) -> None:  # noqa: D401 – (docstring ends with a period above)
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # Projections ----------------------------------------------------------------
        self.qk = nn.Linear(dim * 4, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        # Feed‑forward network --------------------------------------------------------
        self.act = nn.GELU()
        self.proj1 = nn.Linear(dim, dim * 2)
        self.proj2 = nn.Linear(dim * 2, dim)

        # Layer normalisation ---------------------------------------------------------
        self.norm_attn = nn.LayerNorm(dim, eps=1e-6)
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)

        # Dropouts -------------------------------------------------------------------
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Positional encodings --------------------------------------------------------
        beta = torch.atan(
            torch.arange(-178, 179, dtype=torch.float32) / 360 * torch.pi
        )  # (357,)
        self.register_buffer("beta", beta.repeat(360, 1), persistent=False)  # (360, 357)

        theta = torch.sin(torch.arange(0, 360, dtype=torch.float32) / 360 * torch.pi)
        theta = torch.rot90(theta.repeat(357, 1))  # (360, 357)
        self.register_buffer("theta", theta, persistent=False)

        mask = torch.zeros(360, 357, dtype=torch.float32)
        mask[0::C, :] = 1.0  # Keep only the sparse projection angles
        self.register_buffer("mask", mask, persistent=False)

    # ---------------------------------------------------------------------
    # Forward pass ---------------------------------------------------------
    # ---------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, C)
        b, n, c = x.shape

        # Concatenate learned features with positional information ------------
        qk_in = torch.cat(
            (
                x,
                self.theta.expand(b, -1, -1),
                self.beta.expand(b, -1, -1),
                self.mask.expand(b, -1, -1),
            ),
            dim=2,
        )  # (B, N, 4*C)

        # Query / Key ---------------------------------------------------------
        qk = (
            self.qk(qk_in)
            .reshape(b, n, 2, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k = qk[0], qk[1]  # Each is (B, heads, N, C//heads)

        # Value ---------------------------------------------------------------
        v = (
            self.v(x)
            .reshape(b, n, self.num_heads, c // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        # Scaled dot‑product attention ---------------------------------------
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x_attn = (attn @ v).transpose(1, 2).reshape(b, n, c)

        # Residual & normalisation -------------------------------------------
        x = self.norm_attn(x_attn + x)

        # Feed‑forward --------------------------------------------------------
        x_ffn = self.proj2(self.act(self.proj1(x)))
        x = self.norm_ffn(x + x_ffn)
        return x


# ---------------------------------------------------------------------------
# Standard multi‑head attention (no positional encoding) --------------------
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Vanilla multi‑head self‑attention used as a building block."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int = 7,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
    ) -> None:  # noqa: D401
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_out = nn.Linear(dim, dim)

        self.proj1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(dim * 2, dim)

        self.norm_attn = nn.LayerNorm(dim, eps=1e-6)
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, C)
        b, n, c = x.shape

        # QKV projection ------------------------------------------------------
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention -----------------------------------------------------------
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x_attn = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.norm_attn(x_attn + x)

        # Feed‑forward --------------------------------------------------------
        x_ffn = self.proj2(self.act(self.proj1(x)))
        x = self.norm_ffn(x + x_ffn)
        return x


# ---------------------------------------------------------------------------
# Sinogram processing block --------------------------------------------------
# ---------------------------------------------------------------------------


class SinogramBlock(nn.Module):
    """Stack of attention layers operating on sinogram data."""

    def __init__(self, num_sensor: int, *, angle_step: int, num_heads: int = 7):
        super().__init__()
        self.sample = 360 // angle_step  # Number of projections

        self.attn_pos = AttentionPos(num_sensor, num_heads=num_heads)
        self.attn_blocks = nn.Sequential(
            Attention(num_sensor, num_heads=num_heads),
            Attention(num_sensor, num_heads=num_heads),
            Attention(num_sensor, num_heads=num_heads),
            Attention(num_sensor, num_heads=num_heads),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, C)
        residual = x
        x = self.attn_pos(x)
        x = self.attn_blocks(x)
        x = self.act(x + residual)
        return x


# ---------------------------------------------------------------------------
# Top‑level model -----------------------------------------------------------
# ---------------------------------------------------------------------------


class ViewTrans(nn.Module):
    """Whole model: sinogram attention → filtered back‑projection (FBP)."""

    def __init__(self) -> None:  # noqa: D401
        super().__init__()
        from utils import FbpLayer  # Local import to avoid circular deps

        self.sin = SinogramBlock(num_sensor=357, angle_step=ANGLE_STEP)
        self.fbp = FbpLayer()
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 360, 357) raw sinogram data.
        Returns:
            (B, 1, H, W) reconstructed CT image.
        """
        sin_processed = self.sin(x)
        ct_out = self.fbp(sin_processed).permute(0, 3, 2, 1)
        return self.act(ct_out)


# ---------------------------------------------------------------------------
# Convenience factory -------------------------------------------------------
# ---------------------------------------------------------------------------


