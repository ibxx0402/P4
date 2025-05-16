#GPU SSIM 
import torch
import torch.nn.functional as F
import cv2
import numpy as np


def gpu_ssim(img1, img2):
    device = torch.device("mps") #change cuda to mps for apple silicon
 
    
    # Convert from BGR to RGB and normalize to [0, 1]
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) / 255.0
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) / 255.0
    
    # Convert to tensors and add batch dimension
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to(device) #change to .float() instead of half() for float32
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    # Constants for SSIM calculation
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Calculate means
    mu1 = F.avg_pool2d(img1, kernel_size=11, stride=1, padding=5)
    mu2 = F.avg_pool2d(img2, kernel_size=11, stride=1, padding=5)
    
    # Calculate variances and covariance
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=11, stride=1, padding=5) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=11, stride=1, padding=5) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, kernel_size=11, stride=1, padding=5) - mu1_mu2
    
    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # Return mean SSIM
    return ssim_map.mean().item()



""" def gpu_ssim(img1, img2, win_size=11, gaussian_weights=True):
    if isinstance(img1, np.ndarray):
        # Convert from BGR to RGB and normalize to [0, 1]
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) / 255.0
        # Convert to tensor and add batch dimension
        img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float()
        
    if isinstance(img2, np.ndarray):
        # Convert from BGR to RGB and normalize to [0, 1]
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) / 255.0
        # Convert to tensor and add batch dimension
        img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float()
    
    # Check if MPS is available (Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        img1 = img1.to(device)
        img2 = img2.to(device)
    
    # Fallback to CUDA if available (NVIDIA GPU)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        img1 = img1.to(device)
        img2 = img2.to(device)

    ssim_loss = pytorch_ssim.SSIM(window_size=win_size)
    return ssim_loss(img1, img2).item() """