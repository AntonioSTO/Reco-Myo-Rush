import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_GRU_Gesture(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # CNN local (feature extractor)
        self.cnn = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # GRU temporal
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            dropout=0.3
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, T, 8)
        x = x.permute(0, 2, 1)          # (B, 8, T)
        x = self.cnn(x)                 # (B, 64, T)
        x = x.permute(0, 2, 1)          # (B, T, 64)

        _, h = self.gru(x)              # h: (1, B, 64)
        h = h.squeeze(0)                # (B, 64)

        return self.classifier(h)
