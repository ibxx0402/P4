#based on https://stackoverflow.com/questions/2440504/noise-estimation-noise-measurement-in-image
#Logic https://www.sciencedirect.com/science/article/pii/S1077314296900600?via%3Dihub


import cv2
import numpy as np
import math
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

    M = [[1, -2, 1],
         [-2, 4, -2],
         [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

    return sigma

# Load the image
img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)  # Loading as grayscale since the function expects 2D array
print(estimate_noise(img))
