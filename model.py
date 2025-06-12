
import os
import torch
import torch.nn as nn
from torch import device

from utils import FbpLayer

C: int = 12  # Sparsity factor
ANGLE_STEP: int = 360 // C  # Angular distance between successive projections


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Attention_pos(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop_ratio=0.1, proj_drop_ratio=0.1):
        super(Attention_pos, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qk = nn.Linear(dim * 4, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj1 = nn.Linear(dim, dim * 2)
        self.proj2 = nn.Linear(dim * 2, dim)
        self.normal1 = nn.LayerNorm(357, eps=1e-6)
        self.normal2 = nn.LayerNorm(357, eps=1e-6)

        # Positional encoding for the receiver (using Tan function)
        beta = torch.squeeze(torch.tensor(range(-178, 179, 1)) / 360 * torch.pi)
        beta = beta.repeat(360, 1)
        self.beta = torch.atan(beta).to(device)

        # Positional encoding for the angle (using Cos function)
        angle = torch.squeeze(torch.tensor(range(0, 360, 1)) / 360 * torch.pi)
        angle = angle.repeat(357, 1)
        angle = torch.sin(angle)
        self.angle = torch.rot90(angle).to(device)

        # Mask used to hide specific angles and receivers
        self.mask = torch.zeros(360, 357).to(device)
        self.mask[0::C, :] = 1

    def forward(self, x1):
        B, N, C = x1.shape
        # Expand angle, beta, and mask to match the batch dimension of x1
        beta = self.beta.repeat(B, 1, 1).to(device)
        angle = self.angle.repeat(B, 1, 1).to(device)
        mask = self.mask.repeat(B, 1, 1).to(device)

        # Normalize input x1, concatenate with angle, beta, and mask
        x2 = self.normal1(x1)
        qk = torch.cat((x2, angle, beta, mask), dim=2)
        #         plt.imshow(qk[0].cpu().detach().numpy())
        #         plt.show()

        # Generate Query (Q), Key (K), and Value (V)
        qk = self.qk(qk).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 2,3,7,360,51
        q, k = qk[0], qk[1]
        v = self.v(x2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # Compute similarity between Q and K
        attn = attn.softmax(dim=-1)  # Apply softmax

        x2 = (attn @ v).transpose(1, 2).reshape(B, N, C)  # Attention-weighted output

        x2 = self.proj(x2)
        x2 = x2 + x1
        x = self.normal2(x2)
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x)
        x = x + x2
        return x2

# Attention without positional encoding
class Attention(nn.Module):
    def __init__(self, dim, num_heads=7, qkv_bias=False, qk_scale=None, attn_drop_ratio=0.1, proj_drop_ratio=0.1):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(dim * 2, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.normal1 = nn.LayerNorm(357, eps=1e-6)
        self.normal2 = nn.LayerNorm(357, eps=1e-6)

    def forward(self, x1):
        B, N, C = x1.shape
        x2 = self.normal1(x1)
        qkv = self.qkv(x2).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x2 = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x2 = self.proj(x2)

        x2 = x2 + x1
        x = self.normal2(x2)
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x)
        x = x + x2
        return x

# Processing sinogram
class sin_angle(nn.Module):
    def __init__(self, num_sensor, angle, num_heads=7):  # num_sensor: number of sensors per angle, angle: downsampled degrees
        super().__init__()

        self.sample = int(360 / angle)
        self.attn_pos = Attention_pos(num_sensor, num_heads=num_heads)
        self.attn1 = Attention(num_sensor, num_heads=num_heads)
        self.attn2 = Attention(num_sensor, num_heads=num_heads)
        self.attn3 = Attention(num_sensor, num_heads=num_heads)
        self.attn4 = Attention(num_sensor, num_heads=num_heads)

        self.act = nn.ReLU()

    # Combine x_in and angle positional embedding pos_embed (element-wise addition)
    def connet(self, x_in, pos_embed):
        B, channel, angle, sensor = x_in.shape
        x_cat = torch.zeros((B, channel, angle, sensor))

        for i in range(self.sample - 1):
            x_cat[:, i + 1, :, :] = x_in[:, i + 1, :, :] + pos_embed[:, i * angle:i * angle + angle, :]
        return x_cat

    def forward(self, x_in):
        # B,angle,sensor = x_in.shape
        # x_end = torch.zeros((B,self.sample-1,int(angle/self.sample),sensor)).to(device)

        x_i = x_in
        x_i = self.attn_pos(x_i)
        x_i = self.attn1(x_i)
        x_i = self.attn2(x_i)
        x_i = self.attn3(x_i)
        x_i = self.attn4(x_i) + x_in
        x_i = self.act(x_i)
        #         for i in range(self.sample-1):
        #             x_end[:,i,:,:] = x_i[:,i+1::self.sample,:]

        return x_i



angle = int(360 / C)

class mymodel(nn.Module):
    def __init__(self, ):
        super(mymodel, self).__init__()

        self.sin = sin_angle(num_sensor=357, angle=angle, num_heads=7)
        self.fbp = FbpLayer()
        self.act = nn.ReLU()


    def forward(self, x):

        sin1 = self.sin(x)
        ct_out = self.fbp(sin1).permute(0, 3, 2, 1)

        return ct_out