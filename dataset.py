import os
import re
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

        for file in os.listdir(root):
            if not file.endswith(".csv"):
                continue

            # Extrai ID do voluntário: voluntary_001_center.csv → 1
            match = re.search(r"voluntary_(\d+)_", file)
            if match is None:
                continue

            subject_id = int(match.group(1))
            if subject_id not in volunteers:
                continue

            df = pd.read_csv(os.path.join(root, file))

            data = df.iloc[:, :-1].values  # canais EMG
            labels = df.iloc[:, -1].values  # labels

            for i in range(0, len(data) - window_size, stride):
                window = data[i:i+window_size]
                label = labels[i + window_size // 2]

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
