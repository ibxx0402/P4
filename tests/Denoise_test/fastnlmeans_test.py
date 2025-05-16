import os 
import cv2
import numpy as np
from Denoise_test.add_noise import noisy
from Denoise_test.ssim_gpu import gpu_ssim
from Denoise_test.estimate_noise import estimate_noise

def fastnlmeans_parameter_combinations(denoise_score_array, h_list, h_total, h_color_list, h_color_total, template_size_list, search_size_list):
    i = 0
    for h_value in np.arange(h_list[0], h_list[1], h_list[2])[:h_total]: #cannot use range since it is float
        for h_color_value in np.arange(h_color_list[0], h_color_list[1], h_color_list[2])[:h_color_total]: #cannot use range since it is float
            for template_size in range(template_size_list[0], template_size_list[1], template_size_list[2]):
                for search_size in range(search_size_list[0], search_size_list[1], search_size_list[2]):
                    denoise_score_array[i, 2] = h_value
                    denoise_score_array[i, 3] = h_color_value
                    denoise_score_array[i, 4] = template_size
                    denoise_score_array[i, 5] = search_size
                    i += 1
    
    return denoise_score_array

def initialize(h_list, h_color_list, template_size_list, search_size_list):
    denoise_h_total = round((h_list[1] - h_list[0]) / h_list[2])

    denoise_h_color_total = round((h_color_list[1] - h_color_list[0]) / h_color_list[2])

    denoise_template_size_total = round((template_size_list[1] - template_size_list[0]) / template_size_list[2])

    denoise_search_size_total = round((search_size_list[1] - search_size_list[0]) / search_size_list[2])

    print(f"total size: {denoise_h_total*denoise_h_color_total*denoise_template_size_total*denoise_search_size_total}")

    denoise_score_array = np.zeros((denoise_h_total*denoise_h_color_total*denoise_template_size_total*denoise_search_size_total, 6))
    denoise_score_array = fastnlmeans_parameter_combinations(denoise_score_array, h_list, denoise_h_total, h_color_list, denoise_h_color_total, template_size_list, search_size_list)
    return denoise_score_array, denoise_h_total, denoise_h_color_total

def process_images(img_path, h_list, h_total, h_color_list, h_color_total, template_size_list, search_size_list, noise_strength, denoise_score_array):
    lst = os.listdir(img_path)
    image_count = sum(1 for name in lst if ".png" in name)  # Count valid images

    image_estimate_noise_array = np.zeros((image_count, 2))

    processed_images = 0
    for image in range(1, image_count + 1):
        image = str(image) + ".png"
        print(f"images processed: {processed_images} of {image_count}, current image: {image}", end="\r")

        original_image = cv2.imread(f"{img_path}/{image}")

        noisy_image, added_noise = noisy(original_image, noise_strength)

        # Estimate noise level
        greyscale_noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)
        noise_level = estimate_noise(greyscale_noisy_image)

        image_estimate_noise_array[processed_images, 0] = added_noise
        image_estimate_noise_array[processed_images, 1] = noise_level

        i = 0

        for h_value in np.arange(h_list[0], h_list[1], h_list[2])[:h_total]:
             for h_color_value in np.arange(h_color_list[0], h_color_list[1], h_color_list[2])[:h_color_total]: #cannot use range since it is float
                for template_size in range(template_size_list[0], template_size_list[1], template_size_list[2]):
                    for search_size in range(search_size_list[0], search_size_list[1], search_size_list[2]):
                        denoised_image = cv2.fastNlMeansDenoisingColored(
                            noisy_image,
                            None,
                            float(h_value),
                            float(h_color_value),
                            int(template_size),
                            int(search_size)
                        )
                        # Calculate SSIM and PSNR
                        ssim_value = gpu_ssim(original_image, denoised_image)
                        psnr_value = cv2.PSNR(original_image, denoised_image)

                        # Store the results
                        denoise_score_array[i, 0] += ssim_value
                        denoise_score_array[i, 1] += psnr_value
                        i += 1

        processed_images += 1
    return denoise_score_array, image_count, image_estimate_noise_array
    

def main():
    h_list = [1, 7, 1]
    h_color_list = [1, 9, 1]

    template_size_list = [1, 9, 2] 
    search_size_list = [1, 11, 2] 

    noise_strength_list = [5, 10, 15, 20]

    for noise_strength in noise_strength_list:
        img_path = "img"
        denoise_score_array, h_total, h_color_total = initialize(h_list, h_color_list, template_size_list, search_size_list)
        denoise_score_array, image_count, image_estimate_noise_array = process_images(img_path, h_list, h_total, h_color_list, h_color_total, template_size_list, search_size_list, noise_strength, denoise_score_array)

        avg_denoise_score = denoise_score_array.copy()  
        avg_denoise_score[:, 0:2] /= image_count 

        # Save the results
        np.save(f"fastnlmeans_test_files/fastnlmeans_avg_score_{noise_strength}.npy", avg_denoise_score)

        np.save(f"fastnlmeans_test_files/fastnlmeans_estimate_noise_{noise_strength}.npy", image_estimate_noise_array)

if __name__ == "__main__":
    main()
