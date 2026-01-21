import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import HandGestureDataset
from model import CNN1DGesture

# ======================
# Configurações
# ======================
DATA_DIR = "db"
WINDOW_SIZE = 15
BATCH_SIZE = 128
EPOCHS = 40
LR = 1e-3

TRAIN_VOLUNTARIES = list(range(1, 9))   # 001–008
TEST_VOLUNTARIES = [9, 10]               # 009–010

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Datasets
# ======================
train_dataset = HandGestureDataset(
    data_dir=DATA_DIR,
    voluntaries=TRAIN_VOLUNTARIES,
    window_size=WINDOW_SIZE
)

test_dataset = HandGestureDataset(
    data_dir=DATA_DIR,
    voluntaries=TEST_VOLUNTARIES,
    window_size=WINDOW_SIZE
)

# Compartilha o label_map
test_dataset.label_map = train_dataset.label_map

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ======================
# Modelo
# ======================
model = CNN1DGesture(num_classes=len(train_dataset.label_map)).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ======================
# Treinamento
# ======================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # ======================
    # Avaliação
    # ======================
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {acc:.2f}%")

torch.save(model.state_dict(), "cnn1d_hand_gesture.pth")
print("✅ Modelo salvo com sucesso")
