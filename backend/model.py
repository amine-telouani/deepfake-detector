import torch
import torch.nn as nn
import torch.nn.functional as F

class DualBranchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.rgb_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.rgb_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.fft_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.fft_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fft_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128 * 4 * 4 * 2, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, rgb, fft):
        r = self.pool(F.relu(self.rgb_conv1(rgb)))
        r = self.pool(F.relu(self.rgb_conv2(r)))
        r = self.pool(F.relu(self.rgb_conv3(r)))
        r = r.view(r.size(0), -1)

        f = self.pool(F.relu(self.fft_conv1(fft)))
        f = self.pool(F.relu(self.fft_conv2(f)))
        f = self.pool(F.relu(self.fft_conv3(f)))
        f = f.view(f.size(0), -1)

        combined = torch.cat([r, f], dim=1)
        x = self.dropout(F.relu(self.fc1(combined)))
        x = self.fc2(x)
        return x