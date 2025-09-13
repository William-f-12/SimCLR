import os
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from SimCLR import SimCLR
from torch.utils.data import Dataset, DataLoader


class EvalImageDataset(Dataset):
    def __init__(self, root_dir, per_class=200, transform=None):
        self.img_paths = []
        self.labels = []
        self.per_class = per_class
        self.transform = transform if transform else transforms.ToTensor()
        for label in range(1, 11):
            class_dir = os.path.join(root_dir, str(label))
            for fname in os.listdir(class_dir):
                if fname.endswith('.png'):
                    self.img_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        img = self.transform(img)
        label = self.labels[idx]
        return img, label
    

def linear_eval(features, labels, device, num_classes=10, epochs=20):
    # 1. define linear classifier
    classifier = nn.Linear(features.size(1), num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)

    # 2. move data to device
    features = features.to(device)
    labels = labels.to(device)

    # 3. train linear classifier
    for epoch in range(epochs):
        classifier.train()
        optimizer.zero_grad()
        outputs = classifier(features)
        loss = criterion(outputs, labels-1)
        loss.backward()
        optimizer.step()
        # print loss every 5 epochs
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # 4. evaluate accuracy
    classifier.eval()
    with torch.no_grad():
        outputs = classifier(features)
        preds = outputs.argmax(dim=1)
        acc = (preds == (labels-1)).float().mean().item()
    print(f"Linear evaluation accuracy: {acc:.4f}")

    return acc


def evaluate(model_path, eval_loader, device):
    # 1. load model
    model = SimCLR()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    # 2. feature extraction
    features = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in eval_loader:
            imgs = imgs.to(device)
            feats = model.encode(imgs)
            features.append(feats.cpu())
            labels.append(lbls)
    features = torch.cat(features)
    labels = torch.cat(labels)

    # 3. Downstream classifier evaluation
    acc = linear_eval(features, labels, device=device, num_classes=10, epochs=30)

    # 4. Output results
    print(f"Evaluation finished. Final linear evaluation accuracy: {acc * 100:.2f}%")


# === Main function ===
PATH = "img/"
PER_CLASS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "saved_model/simclr_epoch6.pth"

def main():
    eval_dataset = EvalImageDataset(PATH, per_class=PER_CLASS)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    evaluate(MODEL_PATH, eval_loader, device=DEVICE)

    
if __name__ == "__main__":
    main()