import torch
import torch.nn as nn

# ======================================================
# 1) SE-Attention Block (خفيف وفعال جداً)
# ======================================================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, length = x.size()
        y = self.avg_pool(x).view(batch, channels)
        y = self.fc(y).view(batch, channels, 1)
        return x * y
        

# ======================================================
# 2) Temporal Block (نفس TCN لكن بنضيف SEBlock)
# ======================================================

class TemporalBlock(nn.Module):
    def __init__(self, in_c, out_c, k, dilation, dropout=0.2):
        super().__init__()
        pad = (k - 1) * dilation

        self.conv1 = nn.Conv1d(in_c, out_c, k, padding=pad, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_c, out_c, k, padding=pad, dilation=dilation)
        self.relu2 = nn.ReLU()

        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else None
        
        # ← إضافة Attention هنا
        self.se = SEBlock(out_c)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.dropout(out)

        out = self.relu2(self.conv2(out))
        out = self.dropout(out)

        # Shortcut
        res = x if self.downsample is None else self.downsample(x)

        # Trim length mismatch
        out = out[:, :, :res.size(2)]

        # Apply SE Attention
        out = self.se(out)

        return torch.relu(out + res)


# ======================================================
# 3) الموديل الأساسي (SE-TCN)
# ======================================================

class SETCN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        channels = [32, 32, 32]
        layers = []

        for i in range(len(channels)):
            dilation = 2 ** i
            in_c = in_channels if i == 0 else channels[i - 1]
            out_c = channels[i]
            layers.append(TemporalBlock(in_c, out_c, 3, dilation))

        self.network = nn.Sequential(*layers)

        # Feature aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Stronger classifier head
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.network(x)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)
