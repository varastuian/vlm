import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import Tuple

class ConvBlock(nn.Module):
    """
    Standard Convolutional Block: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpSampleBlock(nn.Module):
    """
    Upsampling Block: Transposed Conv -> Concatenation -> ConvBlock
    """
    def __init__(self, in_channels, out_channels, skip_channels):
        super(UpSampleBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # After upsample: in_channels // 2
        # After cat: (in_channels // 2) + skip_channels
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle padding issues if dimensions don't match exactly
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SiameseUNet(nn.Module):
    """
    Siamese U-Net for Change Detection.
    Uses a shared ResNet-34 backbone to extract features from T1 and T2 images.
    Features are then concatenated and decoded to produce a binary change mask.
    """
    def __init__(self, in_channels=4, out_channels=1): # Sentinel-2 RGB+NIR = 4 channels
        super(SiameseUNet, self).__init__()
        
        # --- Encoder (Shared Weights) ---
        # Using ResNet18 (Lighter model)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modify first layer to accept 'in_channels' (e.g., 4 bands) instead of 3
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        self.encoder1 = resnet.layer1 # 64 channels
        self.encoder2 = resnet.layer2 # 128 channels
        self.encoder3 = resnet.layer3 # 256 channels
        self.encoder4 = resnet.layer4 # 512 channels
        
        # --- Decoder ---
        # Bottleneck: |F1_enc4 - F2_enc4| -> 512 channels
        self.bottleneck_conv = ConvBlock(512, 1024) 

        # Decoders (Same channel sizes for ResNet18 as ResNet34)
        # Up1: In=1024 -> Up=512. Skip(e3)=256. Cat=768. Out=512
        self.up1 = UpSampleBlock(1024, 512, skip_channels=256) 
        
        # Up2: In=512 -> Up=256. Skip(e2)=128. Cat=384. Out=256
        self.up2 = UpSampleBlock(512, 256, skip_channels=128)
        
        # Up3: In=256 -> Up=128. Skip(e1)=64. Cat=192. Out=128
        self.up3 = UpSampleBlock(256, 128, skip_channels=64)
        
        # Up4: In=128 -> Up=64. Skip(e1/inc)=64. Cat=128. Out=64
        self.up4 = UpSampleBlock(128, 64, skip_channels=64)
        
        # Final conv
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward_one(self, x):
        """Feature extraction for one branch"""
        x1 = self.inc(x)      # 1/4 res, 64 ch
        e1 = self.encoder1(x1) # 1/4 res, 64 ch
        e2 = self.encoder2(e1) # 1/8 res, 128 ch
        e3 = self.encoder3(e2) # 1/16 res, 256 ch
        e4 = self.encoder4(e3) # 1/32 res, 512 ch
        return e1, e2, e3, e4

    def forward(self, t1, t2):
        # 1. Feature Extraction (Siamese)
        t1_e1, t1_e2, t1_e3, t1_e4 = self.forward_one(t1)
        t2_e1, t2_e2, t2_e3, t2_e4 = self.forward_one(t2)
        
        # 2. Distance Metric (Absolute Difference) at each level
        diff_e4 = torch.abs(t1_e4 - t2_e4)
        diff_e3 = torch.abs(t1_e3 - t2_e3)
        diff_e2 = torch.abs(t1_e2 - t2_e2)
        diff_e1 = torch.abs(t1_e1 - t2_e1)
        
        # 3. Decoding
        x = self.bottleneck_conv(diff_e4) # 512 -> 1024 (channel increase logic in block)
        
        # Decoder uses the difference maps as skip connections
        x = self.up1(x, diff_e3) # 1024 + 256 -> 512
        x = self.up2(x, diff_e2) # 512 + 128 -> 256
        x = self.up3(x, diff_e1) # 256 + 64 -> 128
        
        # Final upsample to match original resolution (approx)
        # Note: ResNet first conv is stride 2, maxpool stride 2 -> 1/4 resolution initially.
        # We need to upsample more to get back to H, W.
        # Current UpSampleBlock upsamples by 2.
        # Flow: 1/32 -> 1/16 -> 1/8 -> 1/4
        # We need 2 more upsamples or a different head to get to 1/1.
        
        x = self.up4(x, diff_e1) # Logic tweak: diff_e1 is 1/4 res. We need full res.
        
        # Let's adjust up4 to take the 1/4 res result and upsample.
        # But we don't have a 1/1 res feature map from encoder (it starts with stride 2).
        # We can perform bilinear interpolation at the end.
        
        logits = self.outc(x)
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=True)
        
        return self.sigmoid(logits)

class HybridLoss(nn.Module):
    """
    Hybrid Loss = Dice Loss + Focal Loss
    Handles class imbalance.
    """
    def __init__(self, alpha=0.8, gamma=2.0, dice_weight=0.5):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Dice Loss
        smooth = 1.
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        dice_loss = 1 - dice
        
        # Focal Loss
        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce
        focal_loss = focal_loss.mean()
        
        return self.dice_weight * dice_loss + (1 - self.dice_weight) * focal_loss

def sliding_window_inference(model, t1_path, t2_path, window_size=256, stride=128):
    """
    Performs sliding window inference on large GeoTIFFs.
    """
    import rasterio
    from rasterio.windows import Window
    
    with rasterio.open(t1_path) as src_t1, rasterio.open(t2_path) as src_t2:
        meta = src_t1.meta.copy()
        height, width = src_t1.height, src_t1.width
        
        # Output placeholder
        output_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Pad image if necessary (simpler approach: handle edges in loop)
        
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        with torch.no_grad():
            for row in range(0, height, stride):
                for col in range(0, width, stride):
                    # Define window
                    w = min(window_size, width - col)
                    h = min(window_size, height - row)
                    window = Window(col, row, w, h)
                    
                    # Read data
                    t1_chip = src_t1.read(window=window)
                    t2_chip = src_t2.read(window=window)
                    
                    # Pad if chip is smaller than window_size
                    if t1_chip.shape[1] < window_size or t1_chip.shape[2] < window_size:
                        pad_h = window_size - t1_chip.shape[1]
                        pad_w = window_size - t1_chip.shape[2]
                        t1_chip = np.pad(t1_chip, ((0,0), (0, pad_h), (0, pad_w)), mode='constant')
                        t2_chip = np.pad(t2_chip, ((0,0), (0, pad_h), (0, pad_w)), mode='constant')

                    # Preprocess
                    t1_tensor = torch.from_numpy(t1_chip).float().unsqueeze(0).to(device)
                    t2_tensor = torch.from_numpy(t2_chip).float().unsqueeze(0).to(device)
                    
                    # Normalize (should match training normalization)
                    t1_tensor /= 255.0
                    t2_tensor /= 255.0
                    
                    # Inference
                    output = model(t1_tensor, t2_tensor) # Shape: (1, 1, H, W)
                    mask = (output > 0.5).cpu().numpy().astype(np.uint8)[0, 0]
                    
                    # Crop back if padded
                    if t1_chip.shape[1] < window_size or t1_chip.shape[2] < window_size:
                         mask = mask[:h, :w]
                    
                    output_mask[row:row+h, col:col+w] = mask[:h, :w]
                    
        return output_mask, meta

if __name__ == "__main__":
    # Sanity Check
    model = SiameseUNet(in_channels=4)
    x1 = torch.randn(1, 4, 256, 256)
    x2 = torch.randn(1, 4, 256, 256)
    y = model(x1, x2)
    print(f"Input shape: {x1.shape}")
    print(f"Output shape: {y.shape}")
    
    criterion = HybridLoss()
    target = torch.randint(0, 2, (1, 1, 256, 256)).float()
    loss = criterion(y, target)
    print(f"Loss: {loss.item()}")
