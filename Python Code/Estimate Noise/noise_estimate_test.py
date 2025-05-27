import os 
import cv2 
from skimage.restoration import estimate_sigma
import estimate_noise
import numpy as np

def noisy(image, noise_strength=0.5):
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

# Initialize lists to store comparison results
noise_levels = np.random.rand(5)*500
custom_errors = []
skimage_errors = []

#load image 
print(f"{'Image':<15} {'Noise Level':<12} {'Actual σ':<10} {'Custom Est.':<12} {'Custom Err%':<12} {'Skimage Est.':<12} {'Skimage Err%':<12} {'Better Method'}")
print("-" * 100)

for name in os.listdir("/Users/ibleminen/Library/CloudStorage/OneDrive-AalborgUniversitet/VISUAL STUDIO/img_server/img/better"):
    if ".png" not in name:
        continue
    
    image = cv2.imread(f"/Users/ibleminen/Library/CloudStorage/OneDrive-AalborgUniversitet/VISUAL STUDIO/img_server/img/better/{name}")
    
    i = 0
    for noise_level in noise_levels:
        # Generate noisy image with known noise level
        noisy_image, actual_sigma = noisy(image, noise_level)

        # Save noisy image only if you want to manually look at them later
        output_path = f"/Users/ibleminen/Library/CloudStorage/OneDrive-AalborgUniversitet/VISUAL STUDIO/img_server/img/better/noise/{i}{name}"
        cv2.imwrite(output_path, noisy_image)
        
        # Convert to grayscale for noise estimation
        gray_noisy = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)
        
        # Estimate noise using custom method
        #custom_est = estimate_noise.estimate_noise(gray_noisy)*1.47
        custom_est = estimate_noise.estimate_noise(gray_noisy)
        
        # Estimate noise using skimage method
        skimage_est = estimate_sigma(noisy_image, channel_axis=-1, average_sigmas=True)
        
        # Calculate error percentages
        custom_error = abs(custom_est - actual_sigma) / actual_sigma * 100
        skimage_error = abs(skimage_est - actual_sigma) / actual_sigma * 100
        
        # Store errors for summary
        custom_errors.append(custom_error)
        skimage_errors.append(skimage_error)
        
        # Determine which method is better
        better_method = "Custom" if custom_error < skimage_error else "Skimage"
        if abs(custom_error - skimage_error) < 1:
            better_method = "Similar"
        

        with open(f"/Users/ibleminen/Library/CloudStorage/OneDrive-AalborgUniversitet/VISUAL STUDIO/img_server/sigma_values.csv", "a") as f:
            f.write(f"{name},{noise_level},{actual_sigma},{custom_est},{custom_error},{skimage_est},{skimage_error},{better_method}\n")
        print(f"{name[:15]:<15} {noise_level:<12} {actual_sigma:.2f}{'':<6} {custom_est:.2f}{'':<7} {custom_error:.2f}%{'':<5} {skimage_est:.2f}{'':<7} {skimage_error:.2f}%{'':<5} {better_method}")
        i += 1

# Print summary statistics
avg_custom_error = np.mean(custom_errors)
avg_skimage_error = np.mean(skimage_errors)
print("\nOverall Comparison:")
print(f"Average Error - Custom Method: {avg_custom_error:.2f}%")
print(f"Average Error - Skimage Method: {avg_skimage_error:.2f}%")
print(f"Better method overall: {'Custom' if avg_custom_error < avg_skimage_error else 'Skimage'}")

