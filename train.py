import os
import sys
import torch
from torch import nn
from tqdm import tqdm
from model import ViewTrans
from utils import load_data, load_checkpoint, save_checkpoint, visualize_sinogram, computeloss
from pytorch_msssim import ssim  # Using pytorch_msssim for SSIM calculation

# Setting up GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load training and validation data (only once)
train_data = load_data("./data/train_data.npz")
val_data = load_data("./data/test_data.npz")

# Visualize the first sinogram
sin_in, _ = train_data[0]
visualize_sinogram(sin_in, title="First Sinogram from train_data", xlabel="Detector Samples (357)", ylabel="Projection Angles (360)")

# Define batch size and data loaders (only once)
batch_size = 3

train_dataset = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    collate_fn=train_data.collate_fn
)

val_dataset = torch.utils.data.DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    collate_fn=val_data.collate_fn
)

# Initialize the model
model = ViewTrans().to(device)

# Freeze FBP layer parameters
for p in model.fbp.parameters():
    p.requires_grad = False

# Get trainable parameters
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.33)


checkpoint_dir = './checkpoints'  # Directory to save checkpoints
epoch_start = 0

# Load previously saved checkpoint if available
model, optimizer, epoch_start = load_checkpoint(model, optimizer, checkpoint_dir, epoch_start)


psnrmax = 0  # Variable to track the best PSNR during training

# Training loop
for epoch in range(epoch_start, 500):
    loss_all = 0.  # Total loss for the epoch
    model.train()  # Set model to training mode

    # Use tqdm for progress bar
    train_dataset_tqdm = tqdm(train_dataset, file=sys.stdout)

    for step, data in enumerate(train_dataset_tqdm):
        sin_in, label = data

        ct = model(sin_in.to(device).to(torch.float32))

        # Calculate loss using MSE
        loss = nn.MSELoss()(ct, label.to(device).to(torch.float32))
        loss_all += loss

        # Calculate PSNR for the batch
        ct_psrn = computeloss(ct, label.to(device))

        train_dataset_tqdm.desc = f"epoch:{epoch}, loss: {loss_all:.3f}, ct_psrn: {ct_psrn:.3f}"

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Save checkpoint every 3 epochs
    if (epoch + 1) % 3 == 0:
        save_checkpoint(model, optimizer, epoch + 1, checkpoint_dir)

    # Validation phase
    with torch.no_grad():
        model.eval()  # Set model to evaluation mode
        ct_psnr = 0
        ct_ssim = 0

        for step, data in enumerate(val_dataset):
            sin_in, label = data
            sin_in = sin_in.to(device).float()

            # Forward pass through the model
            ct = model(sin_in.to(device).to(torch.float32))
            ct = ct.clamp(0, 1)  # Clamp the output between 0 and 1

            # Calculate SSIM and PSNR for the batch
            ct_ssim += ssim(ct, label.to(device), data_range=1.0, size_average=True)
            ct_psnr += computeloss(ct, label.to(device))

        # Averaging PSNR and SSIM over the entire validation set
        ct_psnr = ct_psnr / len(val_dataset)
        ct_ssim = ct_ssim / len(val_dataset)

        print("PSNR: ", ct_psnr)
        print("SSIM: ", ct_ssim)

        # Save the model if PSNR improves
        if ct_psnr > psnrmax:
            psnrmax = ct_psnr
            print("PSNR improved, saving model")
            torch.save(model.state_dict(), f"./weights/ct_predict_{epoch + 1}.pth")
