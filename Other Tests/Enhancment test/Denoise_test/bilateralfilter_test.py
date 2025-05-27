import os 
import numpy as np
import cv2
from add_noise import noisy
from ssim_gpu import gpu_ssim

def billateral_parameter_combinations(denoise_score_array, denoise_diameter_list, denoise_sigma_colour_list, denoise_sigma_space_list):
    i = 0
    for diameter in range(denoise_diameter_list[0], denoise_diameter_list[1], denoise_diameter_list[2]):
        for sigma_colour in range(denoise_sigma_colour_list[0], denoise_sigma_colour_list[1], denoise_sigma_colour_list[2]):
            for sigma_space in range(denoise_sigma_space_list[0], denoise_sigma_space_list[1], denoise_sigma_space_list[2]):
                denoise_score_array[i, 2] = diameter
                denoise_score_array[i, 3] = sigma_colour
                denoise_score_array[i, 4] = sigma_space
                i += 1
    return denoise_score_array

def main():
    img_path = "tests/img"
    denoise_diameter_list = [5, 10, 1]
    denoise_sigma_colour_list = [25, 70, 5] 
    denoise_sigma_space_list = [1, 5, 1] 
    noise_strength_list = [400]
    
    denoise_diameter_total = round((denoise_diameter_list[1] - denoise_diameter_list[0]) / denoise_diameter_list[2])
    print(denoise_diameter_total)
    
    denoise_sigma_colour_total = round((denoise_sigma_colour_list[1] - denoise_sigma_colour_list[0]) / denoise_sigma_colour_list[2])
    print(denoise_sigma_colour_total)

    denoise_sigma_space_total = round((denoise_sigma_space_list[1] - denoise_sigma_space_list[0]) / denoise_sigma_space_list[2])
    print(denoise_sigma_space_total)
    
    lst = os.listdir(img_path)
    image_count = sum(1 for name in lst if ".png" in name)  # Count valid images
    
    for noise_strength in noise_strength_list:
        processed_images = 0
        
        # Reset the score array for each noise level
        denoise_score_array = np.zeros((denoise_diameter_total * denoise_sigma_colour_total * denoise_sigma_space_total, 5))
        denoise_score_array = billateral_parameter_combinations(denoise_score_array, denoise_diameter_list, denoise_sigma_colour_list, denoise_sigma_space_list)

        for name in lst:
            if ".png" not in name:
                continue
            i = 0 

            print("             images left: ", image_count - processed_images, end="\r")

            original_image = cv2.imread(f"{img_path}/{name}")

            noisy_image, _ = noisy(original_image, noise_strength)

            for diameter in range(denoise_diameter_list[0], denoise_diameter_list[1], denoise_diameter_list[2]):
                print("diameter: ", diameter, end="\r")
                for sigma_colour in range(denoise_sigma_colour_list[0], denoise_sigma_colour_list[1], denoise_sigma_colour_list[2]):
                    for sigma_space in range(denoise_sigma_space_list[0], denoise_sigma_space_list[1], denoise_sigma_space_list[2]):
                        # Denoise the image using bilateral filter
                        denoised_image = cv2.bilateralFilter(noisy_image, diameter, sigma_colour, sigma_space)
                        
                        # Calculate SSIM for denoised image
                        ssim_score = gpu_ssim(original_image, denoised_image)
                        psnr_score = cv2.PSNR(original_image, denoised_image)

                        # Store SSIM value
                        denoise_score_array[i][0] += ssim_score 
                        denoise_score_array[i][1] += psnr_score
                        
                        i += 1
                    
            processed_images += 1

        # Calculate average SSIM values

        avg_denoise_score = denoise_score_array.copy()  
        avg_denoise_score[:, 0:2] /= image_count 


        #save the results
        with open(f"billateral_avg_score_{noise_strength}.npy", "wb") as f:
            np.save(f, avg_denoise_score)

main()