import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import EMGDataset
from model import MLP
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = EMGDataset(
    "db", [9,10], 50, 5,
    mean=np.load("mean.npy"),
    std=np.load("std.npy")
)

loader = DataLoader(dataset, batch_size=256)
model = MLP(dataset.X.shape[1], len(set(dataset.y)))
model.load_state_dict(torch.load("best_mlp_emg.pth"))
model.to(DEVICE)
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for x, y in loader:
        x = x.to(DEVICE)
        pred = model(x).argmax(1).cpu().numpy()
        y_pred.extend(pred)
        y_true.extend(y.numpy())

print(classification_report(y_true, y_pred))
cm = confusion_matrix(y_true, y_pred)

plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.show()
