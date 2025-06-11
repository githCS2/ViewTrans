import torch
import matplotlib.pyplot as plt
import time

from model import ViewTrans
from utils import load_data, computeloss, compute_rrmse
from pytorch_msssim import ssim

# Load training and validation datasets
train_data = load_data("./data/train_data.npz")
val_data = load_data("./data/test_data.npz")

batch_size = 3

train_dataset = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    collate_fn=train_data.collate_fn,
)

val_dataset = torch.utils.data.DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    collate_fn=val_data.collate_fn,
)

# Select GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ViewTrans().to(device)

# Load pre‑trained weights
model_weights = torch.load("./weights/ct_predict.pth")
model.load_state_dict(model_weights)
model.eval()  # Switch to evaluation mode

# -----------------------------------------------------------------------------
# Evaluation loop
# -----------------------------------------------------------------------------

with torch.no_grad():
    total_psnr = 0.0
    total_ssim = 0.0
    total_inference_time = 0.0  # Accumulate inference time per batch

    for step, data in enumerate(val_dataset):
        sin_in, label = data
        start_time = time.time()  # Start timing

        # Image reconstruction
        ct = model(sin_in.to(device).to(torch.float32))

        end_time = time.time()  # Stop timing
        inference_time = end_time - start_time
        total_inference_time += inference_time

        ct = ct.clamp(0, 1)  # Limit output to [0, 1]

        # SSIM & PSNR metrics
        total_ssim += ssim(ct, label.to(device), data_range=1.0, size_average=True)
        total_psnr += computeloss(ct, label.to(device))

        # Show & save the first reconstructed image of the first batch
        if step == 0:
            label = label.to(device)

            reconstructed_image = ct[0][0].cpu().detach().numpy()
            ground_truth_image = label[0][0].cpu().detach().numpy()  # Third label image (example)

            rrmse = compute_rrmse(
                torch.tensor(reconstructed_image), torch.tensor(ground_truth_image)
            )
            print(f"rRMSE: {rrmse:.3f}%")

            # Residual plot
            residual = reconstructed_image - ground_truth_image
            plt.figure(figsize=(8, 6))
            plt.imshow(residual.T, cmap="gray")
            plt.title("Residual Plot (Reconstructed - Ground Truth)")
            plt.show()

    # Average metrics over all batches
    avg_psnr = total_psnr / (step + 1)
    avg_ssim = total_ssim / (step + 1)

    # Average inference time per batch
    avg_inference_time = total_inference_time / (step + 1)
    print(f"Average Inference Time per Batch: {avg_inference_time:.6f} seconds")
    print(f"Average PSNR: {avg_psnr:.3f}")
    print(f"Average SSIM: {avg_ssim:.3f}")
