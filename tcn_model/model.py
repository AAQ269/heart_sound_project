# model.py

import torch
import torch.nn as nn

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        if out.size(2) != res.size(2):
            out = out[:, :, :res.size(2)]  # Trim to match
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # ✅ FIX: تحقق من الشكل وعدله فقط إذا لزم الأمر
        if x.dim() == 3:
            # إذا كان الشكل (batch, length, channels)
            if x.size(2) == 1 or x.size(2) < x.size(1):
                x = x.transpose(1, 2)  # Convert to (batch, channels, length)
            # وإلا، الشكل صحيح بالفعل: (batch, channels, length)
        
        y = self.network(x)
        y = y[:, :, -1]  # Last time step
        return self.fc(y)

###################################
# تعريف موديلات بأحجام مختلفة
class LightweightTCN(TCN):
    def __init__(self, input_channels, num_classes):
        super(LightweightTCN, self).__init__(
            input_size=input_channels,
            output_size=num_classes,
            num_channels=[16, 16],
            kernel_size=3,
            dropout=0.1
        )

class MediumTCN(TCN):
    def __init__(self, input_channels, num_classes):
        super(MediumTCN, self).__init__(
            input_size=input_channels,
            output_size=num_classes,
            num_channels=[32, 32, 32],
            kernel_size=3,
            dropout=0.2
        )

class HeavyTCN(TCN):
    def __init__(self, input_channels, num_classes):
        super(HeavyTCN, self).__init__(
            input_size=input_channels,
            output_size=num_classes,
            num_channels=[64, 64, 64, 64],
            kernel_size=3,
            dropout=0.3
        )


# ═══════════════════════════════════════════════════
# Testing
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    print("Testing TCN models...")
    
    # Test with correct shape: (batch, channels, length)
    batch_size = 16
    input_channels = 1
    seq_length = 1024
    
    x = torch.randn(batch_size, input_channels, seq_length)
    print(f"\nInput shape: {x.shape}")
    
    models = [
        ("Lightweight", LightweightTCN(input_channels=1, num_classes=2)),
        ("Medium", MediumTCN(input_channels=1, num_classes=2)),
        ("Heavy", HeavyTCN(input_channels=1, num_classes=2))
    ]
    
    for name, model in models:
        try:
            out = model(x)
            params = sum(p.numel() for p in model.parameters())
            print(f"{name:12s} TCN: output={out.shape}, params={params:,}")
        except Exception as e:
            print(f"{name:12s} TCN: ERROR - {e}")