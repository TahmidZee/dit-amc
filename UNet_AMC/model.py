"""
UNet-AMC: Unified Denoising + Classification Model for AMC

Architecture:
- U-Net Encoder: Extracts features with skip connections
- U-Net Decoder: Reconstructs clean signal (denoising path)
- MCLDNN-style Classifier: LSTM + Dense on bottleneck features

One model, joint training:
  L = L_cls + λ * L_denoise
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double conv block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsample: MaxPool -> ConvBlock"""
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool1d(2)
        self.conv = ConvBlock(in_ch, out_ch)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    """Upsample: ConvTranspose -> Concat skip -> ConvBlock"""
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch)  # *2 for skip connection
    
    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        if x.size(-1) != skip.size(-1):
            diff = skip.size(-1) - x.size(-1)
            x = F.pad(x, (diff // 2, diff - diff // 2))
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNetEncoder(nn.Module):
    """U-Net Encoder with skip connections"""
    
    def __init__(self, in_channels=2, base_channels=64, depth=4):
        super().__init__()
        self.depth = depth
        
        # Initial conv
        self.inc = ConvBlock(in_channels, base_channels)
        
        # Downsampling path
        self.downs = nn.ModuleList()
        ch = base_channels
        for i in range(depth):
            self.downs.append(DownBlock(ch, ch * 2))
            ch = ch * 2
        
        self.bottleneck_channels = ch
    
    def forward(self, x):
        """Returns bottleneck features and skip connections"""
        skips = []
        
        x = self.inc(x)
        skips.append(x)
        
        for down in self.downs:
            x = down(x)
            skips.append(x)
        
        # Last element is bottleneck, rest are skips
        bottleneck = skips.pop()
        skips = skips[::-1]  # Reverse for decoder
        
        return bottleneck, skips


class UNetDecoder(nn.Module):
    """U-Net Decoder for signal reconstruction"""
    
    def __init__(self, out_channels=2, base_channels=64, depth=4):
        super().__init__()
        self.depth = depth
        
        # Upsampling path
        self.ups = nn.ModuleList()
        ch = base_channels * (2 ** depth)
        for i in range(depth):
            self.ups.append(UpBlock(ch, ch // 2))
            ch = ch // 2
        
        # Final conv to output channels
        self.outc = nn.Conv1d(base_channels, out_channels, kernel_size=1)
    
    def forward(self, bottleneck, skips):
        """Reconstruct signal from bottleneck and skip connections"""
        x = bottleneck
        
        for i, up in enumerate(self.ups):
            x = up(x, skips[i])
        
        x = self.outc(x)
        return x


class MCLDNNClassifier(nn.Module):
    """MCLDNN-style classifier head: LSTM + Dense"""
    
    def __init__(self, in_channels, hidden_size=128, num_classes=11, dropout=0.5):
        super().__init__()
        
        # LSTM expects (batch, seq_len, features)
        # Our bottleneck is (batch, channels, length)
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Dense layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, length) bottleneck features
        Returns:
            logits: (batch, num_classes)
        """
        # Transpose for LSTM: (batch, length, channels)
        x = x.transpose(1, 2)
        
        # LSTM
        x, _ = self.lstm(x)
        
        # Take last timestep (or could use attention pooling)
        x = x[:, -1, :]
        
        # Dense
        logits = self.fc(x)
        return logits


class UNetAMC(nn.Module):
    """
    Unified Denoising + Classification Model
    
    Architecture:
        Noisy IQ -> Encoder -> Bottleneck -> Classifier -> Class
                                   |
                                   v
                              Decoder -> Denoised IQ
    """
    
    def __init__(
        self,
        in_channels=2,
        base_channels=64,
        depth=4,
        num_classes=11,
        lstm_hidden=128,
        dropout=0.5
    ):
        super().__init__()
        
        self.encoder = UNetEncoder(in_channels, base_channels, depth)
        self.decoder = UNetDecoder(in_channels, base_channels, depth)
        self.classifier = MCLDNNClassifier(
            in_channels=self.encoder.bottleneck_channels,
            hidden_size=lstm_hidden,
            num_classes=num_classes,
            dropout=dropout
        )
        
        self.base_channels = base_channels
        self.depth = depth
    
    def forward(self, x, return_denoised=True):
        """
        Args:
            x: (batch, 2, 128) noisy IQ signal
            return_denoised: whether to compute denoised output
        
        Returns:
            logits: (batch, num_classes) classification logits
            denoised: (batch, 2, 128) denoised signal (if return_denoised=True)
        """
        # Encode
        bottleneck, skips = self.encoder(x)
        
        # Classify from bottleneck
        logits = self.classifier(bottleneck)
        
        if return_denoised:
            # Decode for denoising
            denoised = self.decoder(bottleneck, skips)
            return logits, denoised
        else:
            return logits
    
    def classify_only(self, x):
        """Fast inference without denoising"""
        return self.forward(x, return_denoised=False)
    
    def denoise_then_classify(self, x, num_iterations=1):
        """
        Iterative denoising then classification.
        
        Phase 1: Denoise the signal (optionally multiple times)
        Phase 2: Classify the denoised signal
        """
        for _ in range(num_iterations):
            _, x = self.forward(x, return_denoised=True)
        
        # Final classification on denoised signal
        logits = self.forward(x, return_denoised=False)
        return logits, x


class UNetAMCWithMultiWindow(nn.Module):
    """
    UNet-AMC with multi-window (K-window) support.
    
    Processes K windows and aggregates for final decision.
    """
    
    def __init__(
        self,
        in_channels=2,
        base_channels=64,
        depth=4,
        num_classes=11,
        lstm_hidden=128,
        dropout=0.5,
        pool_method='mean'  # 'mean' or 'attn'
    ):
        super().__init__()
        
        self.base_model = UNetAMC(
            in_channels, base_channels, depth, num_classes, lstm_hidden, dropout
        )
        
        self.pool_method = pool_method
        if pool_method == 'attn':
            self.attn_pool = nn.Sequential(
                nn.Linear(num_classes, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
            )
        
        self.num_classes = num_classes
    
    def forward(self, x, return_denoised=True):
        """
        Args:
            x: (batch, K, 2, 128) K windows of IQ signal
            return_denoised: whether to return denoised signals
        
        Returns:
            logits: (batch, num_classes) aggregated logits
            denoised: (batch, K, 2, 128) denoised signals (if return_denoised)
        """
        B, K, C, L = x.shape
        
        # Flatten batch and K
        x_flat = x.view(B * K, C, L)
        
        # Process all windows
        if return_denoised:
            logits_flat, denoised_flat = self.base_model(x_flat, return_denoised=True)
            denoised = denoised_flat.view(B, K, C, L)
        else:
            logits_flat = self.base_model(x_flat, return_denoised=False)
            denoised = None
        
        # Reshape logits: (B, K, num_classes)
        logits_per_window = logits_flat.view(B, K, self.num_classes)
        
        # Aggregate across windows
        if self.pool_method == 'mean':
            logits = logits_per_window.mean(dim=1)
        elif self.pool_method == 'attn':
            # Attention weights
            attn_scores = self.attn_pool(logits_per_window)  # (B, K, 1)
            attn_weights = F.softmax(attn_scores, dim=1)     # (B, K, 1)
            logits = (logits_per_window * attn_weights).sum(dim=1)  # (B, num_classes)
        else:
            raise ValueError(f"Unknown pool method: {self.pool_method}")
        
        if return_denoised:
            return logits, denoised
        return logits


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = UNetAMC(in_channels=2, base_channels=64, depth=4, num_classes=11)
    print(f"UNetAMC parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(4, 2, 128)
    logits, denoised = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Denoised shape: {denoised.shape}")
    
    # Test multi-window version
    model_k = UNetAMCWithMultiWindow(base_channels=64, depth=4, pool_method='attn')
    print(f"\nUNetAMCWithMultiWindow parameters: {count_parameters(model_k):,}")
    
    x_k = torch.randn(4, 8, 2, 128)  # 8 windows
    logits_k, denoised_k = model_k(x_k)
    print(f"Multi-window input shape: {x_k.shape}")
    print(f"Aggregated logits shape: {logits_k.shape}")
    print(f"Denoised shape: {denoised_k.shape}")
