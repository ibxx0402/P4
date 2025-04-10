#based on https://stackoverflow.com/questions/2440504/noise-estimation-noise-measurement-in-image

import cv2
import numpy as np
import math
import sys
import os
from scipy.signal import convolve2d

def estimate_noise(I):
    """
    Estimate the noise level in an image using a Laplacian filter.
    
    Parameters:
    I (numpy.ndarray): Grayscale input image
    
    Returns:
    float: Estimated noise standard deviation
    
    The matrix M is a Laplacian kernel (a discrete approximation of the Laplacian operator):
    - The Laplacian detects areas of rapid intensity change in the image
    - In a noisy image, random variations cause many small intensity changes
    - The sum of absolute values after convolution correlates with noise level
    - The kernel has a specific structure (sum of elements = 0) that makes it sensitive to noise
    - The central value (4) is positive while its neighbors are arranged to detect local variations
    - The normalization factor adjusts the sum to estimate the standard deviation of Gaussian noise
    """
    
    H, W = I.shape

    # Convert to numpy array for better performance
    M = np.array([[1, -2, 1],
                  [-2, 4, -2],
                  [1, -2, 1]])

    sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

    return sigma


if __name__ == "__main__":
    # Get image filename from command line arguments or use default
    image_path = "image.jpg"
        
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
    noise_level = estimate_noise(img)
    print(f"Estimated noise level in '{image_path}': {noise_level:.6f}")
