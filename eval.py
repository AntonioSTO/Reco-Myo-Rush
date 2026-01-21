import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from dataset import EMGDataset
from model import MLP
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

dataset = EMGDataset(
    "db",
    [9, 10],
    50,
    5,
    label_map=label_map,
    mean=np.load("mean.npy"),
    std=np.load("std.npy")
)

loader = DataLoader(dataset, batch_size=256)

model = MLP(dataset.X.shape[1], len(label_map))
model.load_state_dict(torch.load("best_mlp_emg.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for x, y in loader:
        x = x.to(DEVICE)
        preds = model(x).argmax(1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y.numpy())

print(classification_report(y_true, y_pred))
cm = confusion_matrix(y_true, y_pred)

plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.show()
