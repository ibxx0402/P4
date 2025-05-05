import os 
import cv2
import numpy as np
from Denoise_test.add_noise import noisy
from Denoise_test.ssim_gpu import gpu_ssim
from Denoise_test.estimate_noise import estimate_noise

def gaussian_blur_parameter_combinations(denoise_score_array, denoise_ksize_list, denoise_sigma_list, denoise_sigma_total):
    i = 0
    for ksize in range(denoise_ksize_list[0], denoise_ksize_list[1], denoise_ksize_list[2]):
        for sigma in np.arange(denoise_sigma_list[0], denoise_sigma_list[1], denoise_sigma_list[2])[:denoise_sigma_total]: 
            denoise_score_array[i, 2] = ksize
            denoise_score_array[i, 3] = sigma
            i += 1
    return denoise_score_array

def initialize(denoise_ksize_list, denoise_sigma_list):
    denoise_ksize_total = round((denoise_ksize_list[1] - denoise_ksize_list[0]) / denoise_ksize_list[2])
    denoise_sigma_total = round((denoise_sigma_list[1] - denoise_sigma_list[0]) / denoise_sigma_list[2])
    print(denoise_ksize_total*denoise_sigma_total)

    denoise_score_array = np.zeros((denoise_ksize_total*denoise_sigma_total, 4))
    denoise_score_array = gaussian_blur_parameter_combinations(denoise_score_array, denoise_ksize_list, denoise_sigma_list, denoise_sigma_total)
    return denoise_score_array, denoise_sigma_total

def process_images(img_path, denoise_ksize_list, denoise_sigma_list, noise_strength, denoise_score_array, denoise_sigma_total):
    lst = os.listdir(img_path)
    image_count = sum(1 for name in lst if ".png" in name)  # Count valid images
    image_estimate_noise_array = np.zeros((image_count, 2))


    processed_images = 0
    for image in lst:
        if ".png" not in image:
            continue
        print(f"images processed: {processed_images} of {image_count}, current image: {image}", end="\r")

        original_image = cv2.imread(f"{img_path}/{image}")

        noisy_image, added_noise = noisy(original_image, noise_strength)

        # Estimate noise level
        greyscale_noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)
        noise_level = estimate_noise(greyscale_noisy_image)

        image_estimate_noise_array[processed_images, 0] = added_noise
        image_estimate_noise_array[processed_images, 1] = noise_level


        i = 0
        for ksize in range(denoise_ksize_list[0], denoise_ksize_list[1], denoise_ksize_list[2]):
            for sigma in np.arange(denoise_sigma_list[0], denoise_sigma_list[1], denoise_sigma_list[2])[:denoise_sigma_total]: 
                # Denoise the image using Gaussian blur

                denoised_image = cv2.GaussianBlur(noisy_image, (ksize, ksize), sigma)
                # Calculate SSIM and PSNR
                ssim_value = gpu_ssim(original_image, denoised_image)
                psnr_value = cv2.PSNR(original_image, denoised_image)

                # Store the results
                denoise_score_array[i, 0] += ssim_value
                denoise_score_array[i, 1] += psnr_value
                i += 1

        processed_images += 1
    return denoise_score_array, image_count, image_estimate_noise_array,
    

def main():
    denoise_ksize_list = [3, 9, 2] #must be odd
    denoise_sigma_list = [0.1, 2, 0.1] 
    noise_strength_list = [5, 10, 15, 20]

    for noise_strength in noise_strength_list:
        img_path = "tests/img"
        denoise_score_array, denoise_sigma_total = initialize(denoise_ksize_list, denoise_sigma_list)
        denoise_score_array, image_count, image_estimate_noise_array = process_images(img_path, denoise_ksize_list, denoise_sigma_list, noise_strength, denoise_score_array, denoise_sigma_total)

        avg_denoise_score = denoise_score_array.copy()  
        avg_denoise_score[:, 0:2] /= image_count 

        # Save the results
        np.save(f"gaussian_test_files/gaussian_blur_avg_score_{noise_strength}.npy", avg_denoise_score)

        np.save(f"gaussian_test_files/gaussian_blur_estimate_noise_{noise_strength}.npy", image_estimate_noise_array)

if __name__ == "__main__":
    main()
