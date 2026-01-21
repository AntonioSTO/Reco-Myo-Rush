import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import HandGestureDataset
from model import CNN_GRU_Gesture

# ======================
# Configurações ótimas
# ======================
DATA_DIR = "db"
WINDOW_SIZE = 25
STRIDE = 3
BATCH_SIZE = 128
EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 7

TRAIN_VOLUNTARIES = list(range(1, 9))
TEST_VOLUNTARIES = [9, 10]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Datasets
# ======================
train_dataset = HandGestureDataset(
    DATA_DIR,
    TRAIN_VOLUNTARIES,
    WINDOW_SIZE,
    STRIDE
)

test_dataset = HandGestureDataset(
    DATA_DIR,
    TEST_VOLUNTARIES,
    WINDOW_SIZE,
    STRIDE,
    label_map=train_dataset.label_map
)

train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, BATCH_SIZE)

# ======================
# Modelo
# ======================
model = CNN_GRU_Gesture(len(train_dataset.label_map)).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

# ======================
# Early stopping
# ======================
best_acc = 0
epochs_no_improve = 0

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
    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Test Acc: {acc:.2f}%")

    # ======================
    # Early stopping logic
    # ======================
    if acc > best_acc:
        best_acc = acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_cnn_gru_hand_gesture.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("⏹️ Early stopping acionado")
            break

print(f"🏆 Melhor Test Acc: {best_acc:.2f}%")
