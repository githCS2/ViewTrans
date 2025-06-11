import os
import sys
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import re

from model import C
# --- Device Setup ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Dataset Class ---
class MyDataSet(Dataset):
    def __init__(self, sin_in, label):
        self.sin_in = sin_in
        self.label = label

    def __len__(self):
        return len(self.sin_in)

    def __getitem__(self, item):
        sin_in = self.sin_in[item]
        label = self.label[item]
        return sin_in, label

    @staticmethod
    def collate_fn(batch):
        sin_in, label = tuple(zip(*batch))
        sin_in = torch.stack(sin_in, dim=0)
        label = torch.stack(label, dim=0)
        return sin_in, label


# --- Reshaping Function ---
def reshape(x):
    B, channel, angle, sensor = x.shape
    x_end = x[:, 0, :, :]
    for i in range(channel - 1):
        x_end = torch.cat((x_end, x[:, i + 1, :, :]), dim=2)
    x_end = torch.reshape(x_end, (B, angle * channel, sensor))
    return x_end


# --- Interpolation Function ---
def inter(data):

    n, angle, sensor = data.shape
    org = data[:, 0::C, :]

    ex = torch.cat((org, org[:, 0, :].unsqueeze(1)), dim=1)
    sin_in = torch.zeros((n, C, int(360 / C), sensor))

    for i in range(C):
        if i == 0:
            sin_in[:, i, :, :] = data[:, i::C, :]
        else:
            sin_in[:, i, :, :] = ((C - i) * org + (i) * ex[:, 1:, :]) / C

    return sin_in


# --- Data Loading ---
def load_data(trainDataDir):
    data = np.load(trainDataDir)
    sine357 = torch.tensor(data['sinogram'])
    sin_in = reshape(inter(sine357))
    ct = torch.tensor(data['ct_images']).permute(0, 3, 1, 2)
    data_set = MyDataSet(sin_in, ct)
    return data_set


# --- FBP Layer ---
class FbpLayer(nn.Module):
    def __init__(self):
        super(FbpLayer, self).__init__()

        _rawAT = np.load('./Matrix_A.npz')
        indice = _rawAT['indice'].astype('int32')
        data = _rawAT['data'].astype('float32')
        self.cos = torch.tensor(_rawAT['cos'].astype('float32').transpose()).to(device)

        shape = (65536, 128520)
        indice = torch.tensor(indice.transpose())
        data = torch.tensor(data).reshape(-1)

        A = torch.sparse_coo_tensor(indice, data, shape)
        self.A_Matrix = A.to(device)

        _out_sz = round(np.sqrt(float(self.A_Matrix.shape[0])))
        self.out_shape = (_out_sz, _out_sz)

        fbp_filter_weight = torch.tensor(_rawAT['filt'].astype('float32')).to(device)
        self.fbp_filter_weight = nn.Parameter(fbp_filter_weight.reshape(1, 1, 1, -1)).to(device)
        self.fbp_filter = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(713, 1), stride=(1, 1), padding='same')
        self.fbp_filter.weight.data = self.fbp_filter_weight
        self.fbp_filter.bias.data = torch.tensor([0.])

    def forward(self, sin_fan):
        sin_fan = sin_fan.unsqueeze(1)
        sin_sz = sin_fan.shape[1] * sin_fan.shape[2] * sin_fan.shape[3]
        r = sin_fan
        r = r * self.cos

        sin_fan_flt = self.fbp_filter(r.to(device)).permute(0, 2, 3, 1)
        sin_fan_flt = torch.reshape(sin_fan_flt, [-1, sin_sz]).transpose(1, 0)
        fbpOut = torch.sparse.mm(self.A_Matrix, sin_fan_flt).transpose(1, 0)
        fbpOut = torch.reshape(fbpOut, [-1, self.out_shape[0], self.out_shape[1], 1])

        fbpOut = fbpOut.clamp(0, 1)
        return fbpOut


# --- Checkpoint Loading and Saving ---
def load_checkpoint(model, optimizer, checkpoint_dir, epoch_start=0):
    checkpoint_list = os.listdir(checkpoint_dir)
    if len(checkpoint_list) > 0:
        checkpoint_list.sort(key=lambda x: int(re.findall(r"epoch-(\d+).pth", x)[0]))
        last_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_list[-1])
        print('Loading checkpoint from: %s' % last_checkpoint_path)
        checkpoint = torch.load(last_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
    return model, optimizer, epoch_start


def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch-{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")


# --- Visualize Sinogram ---
def visualize_sinogram(sinogram, title="Sinogram", xlabel="Detector Samples", ylabel="Projection Angles"):
    plt.imshow(sinogram.squeeze(), cmap='gray', aspect='auto')  # Remove extra dimensions and display in grayscale
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


# --- PSNR Calculation ---
def computeloss(predict, target):
    n, c, x, y = predict.shape
    psrn_all = 0
    for i in range(n):
        img1 = predict[i].cpu()
        img2 = target[i].cpu()

        ma = torch.max(img2) - torch.min(img2)
        psnr_pix = sum(sum(sum((img1 - img2) ** 2))) / (x * y)
        psrn_all += 10 * np.log((ma ** 2) / psnr_pix.cpu().detach().numpy()) / np.log(10)
    return psrn_all / n


def compute_rrmse(I: torch.Tensor, I_ref: torch.Tensor) -> float:
    """Compute relative RMSE (%) between two images."""

    # Frobenius norm (L2 norm) of the difference
    diff = I - I_ref
    norm_diff = np.linalg.norm(diff, "fro")

    # Frobenius norm of the reference image
    norm_I_ref = np.linalg.norm(I_ref, "fro")

    # Relative RMSE (percentage)
    return (norm_diff / norm_I_ref) * 100