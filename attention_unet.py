import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        psi = torch.sigmoid(psi)
        return x * psi

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(AttentionUNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.ConvTranspose2d(features[-1], features[-1], kernel_size=2, stride=2)
        
        # Downsampling path
        self.encoders = nn.ModuleList()
        for feature in features:
            self.encoders.append(ConvBlock(in_channels, feature))
            in_channels = feature
        
        # Attention blocks
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(features[i], features[i + 1], features[i + 1] // 2) for i in range(len(features) - 1)
        ])
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[-1], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.pool(skip_connections[-1])
        
        # Decoder with attention
        for i in range(len(skip_connections) - 2, -1, -1):
            x = self.up(x)
            x = self.attention_blocks[i](x, skip_connections[i])
        
        # Final output
        return self.final_conv(x)

# Example instantiation
# model = AttentionUNet(in_channels=1, out_channels=1)
