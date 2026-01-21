import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import EMGDataset
from model import MLP
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = "db"
WINDOW = 50
STRIDE = 5
BATCH = 256
EPOCHS = 80
LR = 1e-3

TRAIN_VOL = list(range(1, 9))
TEST_VOL = [9, 10]

train_set = EMGDataset(
    ROOT, TRAIN_VOL, WINDOW, STRIDE, compute_norm=True
)

test_set = EMGDataset(
    ROOT, TEST_VOL, WINDOW, STRIDE,
    mean=train_set.mean,
    std=train_set.std
)

train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH)

num_classes = len(set(train_set.y))
model = MLP(train_set.X.shape[1], num_classes).to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

best_acc = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    correct = total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1} | Loss {total_loss:.4f} | Test Acc {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_mlp_emg.pth")

print(f"✅ Melhor acurácia: {best_acc:.2f}%")

import numpy as np

np.save("mean.npy", train_set.mean)
np.save("std.npy", train_set.std)

