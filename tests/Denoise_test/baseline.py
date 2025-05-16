import os 
import cv2
import numpy as np
from add_noise import noisy
from ssim_gpu import gpu_ssim


def initialize(noise_strength_list):
    length = len(noise_strength_list)
    denoise_score_array = np.zeros((length, 2))
    return denoise_score_array

def process_images(img_path, noise_strength, i, denoise_score_array):
    lst = os.listdir(img_path)
    image_count = sum(1 for name in lst if ".png" in name)  # Count valid images

    processed_images = 0
    for image in lst:
        if ".png" not in image:
            continue
        print(f"images processed: {processed_images} of {image_count}, current image: {image}", end="\r")

        original_image = cv2.imread(f"{img_path}/{image}")
        noisy_image, _ = noisy(original_image, noise_strength)

        # Calculate SSIM and PSNR
        ssim_value = gpu_ssim(original_image, noisy_image)
        psnr_value = cv2.PSNR(original_image, noisy_image)

        # Store the results in the correct row
        denoise_score_array[i, 0] += ssim_value
        denoise_score_array[i, 1] += psnr_value

        processed_images += 1


    denoise_score_array[i, 0] /= processed_images
    denoise_score_array[i, 1] /= processed_images

    return denoise_score_array[i]
    

def main():
    noise_strength_list = [5, 10, 15, 20]
    baseline_score_array = initialize(noise_strength_list)
    
    for i, noise_strength in enumerate(noise_strength_list):
        img_path = "tests/img"
        baseline_score_array[i] = process_images(img_path, noise_strength, i, baseline_score_array)

    print(baseline_score_array, "\n")

    np.save("tests/baseline_score.npy", baseline_score_array)


if __name__ == "__main__":
    main()
