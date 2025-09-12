import os, glob
import torch
from SimCLR import SimCLR
from loss import NTXentLoss
from data_aug import DataAugmentation
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


def train(model, train_loader, optimizer, loss_fn, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_i, x_j in train_loader:
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

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")


Path  = "img/unlabeled/"
def main():
    # prepare dataset and dataloader
    img_paths = glob.glob(os.path.join(Path, "*.png"))
    dataset = MyImageDataset(img_paths, img_size=96)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # prepare model, loss function, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimCLR().to(device)
    loss_fn = NTXentLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # start training
    train(model, loader, optimizer, loss_fn, device, epochs=2)


if __name__ == "__main__":
    main()