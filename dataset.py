import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class HandGestureDataset(Dataset):
    def __init__(self, data_dir, voluntaries, window_size=15, stride=1):
        self.window_size = window_size
        self.stride = stride
        self.samples = []
        self.labels = []
        self.label_map = {}

        self._load_data(data_dir, voluntaries)

    def _normalize(self, x):
        return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)

    def _load_data(self, data_dir, voluntaries):
        label_counter = 0

        for file in sorted(os.listdir(data_dir)):
            if not file.endswith(".csv"):
                continue

            # extrai ID do voluntário: voluntary_001_center.csv → 1
            vol_id = int(file.split("_")[1])

            if vol_id not in voluntaries:
                continue

            path = os.path.join(data_dir, file)
            df = pd.read_csv(path)

            data = df[[f"CH_{i}" for i in range(1, 9)]].values
            labels = df["State"].values

            data = self._normalize(data)

            for i in range(0, len(data) - self.window_size, self.stride):
                window = data[i:i + self.window_size]
                window_labels = labels[i:i + self.window_size]

                # label majoritário
                label = max(set(window_labels), key=list(window_labels).count)

                if label not in self.label_map:
                    self.label_map[label] = label_counter
                    label_counter += 1

                self.samples.append(window)
                self.labels.append(self.label_map[label])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32).T  # (8, 15)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
