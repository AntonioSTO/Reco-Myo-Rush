import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class HandGestureDataset(Dataset):
    def __init__(
        self,
        data_dir,
        voluntaries,
        window_size=15,
        stride=1,
        label_map=None
    ):
        self.window_size = window_size
        self.stride = stride
        self.samples = []
        self.labels = []

        # Se label_map vier vazio ou None → treino
        self.label_map = {} if label_map is None else label_map

        self._load_data(data_dir, voluntaries)

    def _normalize(self, x):
        return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)

    def _load_data(self, data_dir, voluntaries):
        label_counter = len(self.label_map)

        for file in sorted(os.listdir(data_dir)):
            if not file.endswith(".csv"):
                continue

            vol_id = int(file.split("_")[1])
            if vol_id not in voluntaries:
                continue

            df = pd.read_csv(os.path.join(data_dir, file))

            data = df[[f"CH_{i}" for i in range(1, 9)]].values
            labels = df["State"].values

            data = self._normalize(data)

            for i in range(0, len(data) - self.window_size, self.stride):
                window = data[i:i + self.window_size]
                window_labels = labels[i:i + self.window_size]

                # Label majoritário
                label = max(set(window_labels), key=list(window_labels).count)

                # Só cria labels novas no treino
                if label not in self.label_map:
                    if self.label_map is not None:
                        self.label_map[label] = label_counter
                        label_counter += 1
                    else:
                        continue

                self.samples.append(window)
                self.labels.append(self.label_map[label])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32).T  # (8, 15)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
