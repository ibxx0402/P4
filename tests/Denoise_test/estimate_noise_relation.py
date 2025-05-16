import os 
import cv2
import numpy as np
from add_noise import noisy
from estimate_noise import estimate_noise

def initialize(nr_of_best):
    """
    Initialize the denoise score array with zeros.
    The array has dimensions (nr_of_best, 3) where:
    - nr_of_best is the number of best parameters to test.
    - The first column will store the noise level added to the image.
    - The second column will store the noise level of the noisy image.
    - The third column will store the noise level of the denoised image.
    """
    denoise_score_array = np.zeros((nr_of_best, 3))

    return denoise_score_array

def process_images(img_path, noise_strength, denoise_score_array, param_list):
    """
    Process images in the specified directory, adding noise and denoising them.
    For each image, the function applies a noise level and denoises it using the specified parameters.
    The function calculates the noise level of the original, noisy, and denoised images and stores them in the denoise score array.
    """

    lst = os.listdir(img_path)
    image_count = sum(1 for name in lst if ".png" in name)  # Count valid images

    processed_images = 0
    for image in range(1, image_count + 1):
        image = str(image) + ".png"
        print(f"images processed: {processed_images} of {image_count}, current image: {image}", end="\r")

        original_image = cv2.imread(f"{img_path}/{image}")

        noisy_image, added_noise = noisy(original_image, noise_strength)

        # Estimate noise level
        greyscale_noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)
        noise_level = estimate_noise(greyscale_noisy_image)

        i = 0
        for param in param_list:
            denoised_image = cv2.bilateralFilter(noisy_image, param[0], param[1], param[2])

            # converyt to grayscale
            greyscale_denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)

            denoise_score_array[i, 0] = added_noise
            denoise_score_array[i, 1] += noise_level
            denoise_score_array[i, 2] += estimate_noise(greyscale_denoised_image)
            i += 1

        processed_images += 1
    return denoise_score_array, image_count
    

def main():
    noise_strength_list = [5, 10, 15, 20]
    nr_of_best = 4
    
    # Input the best parameters for chosen algorithm, 
    # horizontal lists == nr of best parameters, thus 4 lists if 4 best parameters
    # vertical lists == noise strength, thus 4 lists if 4 noise strengths

    param_list = [[[6,10,2], [7,10,2], [8,10,2], [9,10,2]], 
                  [[8,15,2], [9,15,2], [6,15,2], [7,15,2]], 
                  [[8,20,2], [9,20,2], [8,15,2], [9,15,2]], 
                  [[8,20,2], [9,20,2], [8,20,2], [9,20,2]]]
    i=0
    for noise_strength in noise_strength_list:
        img_path = "tests/img"
        denoise_score_array = initialize(nr_of_best)
        denoise_score_array, image_count = process_images(img_path, noise_strength, denoise_score_array, param_list[i])
        
        #average the scores
        avg_denoise_score = denoise_score_array.copy()  
        avg_denoise_score[:, 1:2] /= image_count
        avg_denoise_score[:, 2:3] /= image_count

        print(avg_denoise_score, "\n")

        # Save the results for the specific algorithm
        np.save(f"tests/bilateral_test_files/estimate_noise_{noise_strength}.npy", avg_denoise_score)
        i+=1


if __name__ == "__main__":
    main()
