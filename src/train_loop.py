import os, glob, time
from tqdm.auto import tqdm

from SimCLR import SimCLR
from loss import NTXentLoss
from data_aug import DataAugmentation
from timer import EMAMeter, format_time

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class MyImageDataset(Dataset):
    def __init__(self, img_paths, img_size):
        self.img_paths = img_paths
        self.transform = DataAugmentation(img_size)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        x_i, x_j = self.transform(img) # two augmented views
        return x_i, x_j


def train(model, train_loader, optimizer, loss_fn, device, start_epoch, epochs):
    total_iters = len(train_loader) * (epochs - start_epoch + 1)
    ema_time = EMAMeter(beta=0.9)
    global_start = time.time()

    model.train()
    for epoch in range(start_epoch-1, epochs):
        epoch_start = time.time()
        pbar = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{epochs}",
                    leave=False)
        
        total_loss = 0
        for step, (x_i, x_j) in pbar:
            iter_start = time.time()

            x_i, x_j = x_i.to(device), x_j.to(device)

            # forward pass
            z_i, z_j = model(x_i, x_j)

            # compute loss
            loss = loss_fn(z_i, z_j)

            # backward pass and update
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # calculate total loss
            total_loss += loss.item()

            # update EMA timer
            iter_time = time.time() - iter_start
            ema_time.update(iter_time)

            iters_done = epoch * len(train_loader) + (step + 1)
            iters_left = total_iters - iters_done
            eta_sec = ema_time.avg * iters_left

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                bt=f"{iter_time*1000:.0f}ms",
                eta=format_time(eta_sec)
            )

        torch.save({"epoch": epoch,
              "model_state_dict": model.state_dict(),
              "optimizer_state_dict": optimizer.state_dict(),
              "loss": loss}, f"/content/drive/MyDrive/Colab_Notebooks/SimCLR/saved_model/checkpoint{epoch+1}.pth")

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{epochs} finished in {format_time(epoch_time)} "
              f"| avg batch ~ {ema_time.avg*1000:.0f} ms")

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

    total_time = time.time() - global_start
    print(f"Training done in {format_time(total_time)}")


# === Main function ===
PATH  = "img/unlabeled/"
CHECKPOINT = -1
def main():
    # prepare dataset and dataloader
    img_paths = glob.glob(os.path.join(PATH, "*.png"))
    dataset = MyImageDataset(img_paths, img_size=96)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # prepare model, loss function, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimCLR().to(device)
    loss_fn = NTXentLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    start_epoch = 0

    if CHECKPOINT >= 0:
        checkpoint = torch.load(f"/content/drive/MyDrive/Colab_Notebooks/SimCLR/saved_model/checkpoint{CHECKPOINT}.pth", map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    # start training
    train(model, loader, optimizer, loss_fn, device, start_epoch=start_epoch, epochs=20)


if __name__ == "__main__":
    main()