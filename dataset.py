import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from features import extract_features

class EMGDataset(Dataset):
    def __init__(
        self,
        root,
        volunteers,
        window_size=50,
        stride=5,
        mean=None,
        std=None,
        compute_norm=False
    ):
        self.X = []
        self.y = []

        for v in volunteers:
            folder = os.path.join(root, f"volunteer_{v:02d}")
            for file in os.listdir(folder):
                if not file.endswith(".csv"):
                    continue

                df = pd.read_csv(os.path.join(folder, file))
                data = df.iloc[:, :-1].values  # canais
                labels = df.iloc[:, -1].values

                for i in range(0, len(data) - window_size, stride):
                    window = data[i:i+window_size]
                    label = labels[i+window_size//2]
                    feats = extract_features(window)

                    self.X.append(feats)
                    self.y.append(label)

        self.X = np.stack(self.X)
        self.y = np.array(self.y)

        if compute_norm:
            self.mean = self.X.mean(axis=0)
            self.std = self.X.std(axis=0) + 1e-8
        else:
            self.mean = mean
            self.std = std

        self.X = (self.X - self.mean) / self.std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long)
        )
