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
        window_size,
        stride,
        label_map=None,
        mean=None,
        std=None,
        compute_norm=False
    ):
        X_temp, y_temp = [], []
        label_set = set()

        for file in os.listdir(root):
            if not file.endswith(".csv"):
                continue

            match = re.search(r"voluntary_(\d+)_", file)
            if match is None:
                continue

            subject = int(match.group(1))
            if subject not in volunteers:
                continue

            df = pd.read_csv(os.path.join(root, file))

            data = df.iloc[:, :-1].values
            labels = df.iloc[:, -1].values

            for i in range(0, len(data) - window_size, stride):
                window = data[i:i + window_size]
                label = labels[i + window_size // 2]

                X_temp.append(extract_features(window))
                y_temp.append(label)
                label_set.add(label)

        # Criar label_map SOMENTE no treino
        if label_map is None:
            self.label_map = {l: i for i, l in enumerate(sorted(label_set))}
        else:
            self.label_map = label_map

        self.X = []
        self.y = []

        for x, y in zip(X_temp, y_temp):
            if y in self.label_map:
                self.X.append(x)
                self.y.append(self.label_map[y])

        self.X = np.stack(self.X)
        self.y = np.array(self.y, dtype=np.int64)

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
