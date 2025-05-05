import cv2
import numpy as np

def noisy(image, noise_strength):
    # based on https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    row,col,ch= image.shape
    mean = 0
    var = noise_strength
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)

    # Clip values to valid range and convert back to uint8
    noisy = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy, sigma  # Return both noisy image and the actual sigma used