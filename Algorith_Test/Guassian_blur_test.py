import os 
import cv2 
import numpy as np
import matplotlib.pyplot as plt
# Add PyTorch imports
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as cpu_ssim

# GPU-accelerated SSIM function using PyTorch and MPS (Metal Performance Shaders)
def gpu_ssim(img1, img2):
    # Check if MPS (Metal Performance Shaders) is available
    if not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        return cpu_ssim(img1, img2, channel_axis=-1)
    
    # Convert images to PyTorch tensors and move to MPS device
    device = torch.device("mps")
    
    # Convert from BGR to RGB and normalize to [0, 1]
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) / 255.0
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) / 255.0
    
    # Convert to tensors and add batch dimension
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to(device)
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

def retrace_parameters(ssim_array, kernel_list, std_list):
    highest_ssim_index = np.argmax(ssim_array)
    highest_ssim_value = ssim_array[highest_ssim_index]
    print(f"Highest avg ssim value: {highest_ssim_value}")
    print(f"Index of highest avg ssim value: {highest_ssim_index}")
    
    nr_of_std = round((std_list[1]-std_list[0])/std_list[2])
    kernel_size_index = highest_ssim_index // nr_of_std
    kernel_size = kernel_list[kernel_size_index]
    print("kernel size: ", kernel_size)

    # to find the std, we need to find the index of the std value
    std_index = highest_ssim_index % nr_of_std   

    # Apply rounding to avoid floating-point precision issues
    std_value = round(std_list[0] + (std_index * std_list[2]), 5)

    print("std value: ", std_value)
    return kernel_size, std_value, highest_ssim_value, highest_ssim_index

def denoise_image(kernel_list, std_list, noise_strength, strength_list):
    processed_images = 0
    nr_of_std = round((std_list[1]-std_list[0])/std_list[2])
    nr_of_parameters = int(len(kernel_list) * nr_of_std)

    # Initialize SSIM array with the correct size
    ssim_array = np.zeros((nr_of_parameters, 1))

    ssim_array_2 = np.zeros((round((strength_list[1]-strength_list[0])/strength_list[2])*nr_of_parameters, 1))

    path = "/Users/ibleminen/Library/CloudStorage/OneDrive-AalborgUniversitet/VISUAL STUDIO/img_server/img/better"
    lst = os.listdir(path)
    nr_of_images = sum(1 for name in lst if ".png" in name)  # Count valid images
    
    for name in lst:
        if ".png" not in name:
            continue
        i = 0 

        print("images left: ", nr_of_images - processed_images, end="\r")

        image = cv2.imread(f"{path}/{name}")

        # Add 100 noise to ground truth image
        noisy_image, _ = noisy(image, noise_strength)

        for kernel_size in kernel_list:
            for std in np.arange(std_list[0], std_list[1], std_list[2]):  # Use numpy.arange for floating-point steps
                # Denoise the image using Gaussian blur
                denoise_1 = cv2.GaussianBlur(noisy_image, (kernel_size, kernel_size), std)
                
                #Apply unsharp mask 
                i2 = 0
                for strength in np.arange(strength_list[0], strength_list[1], strength_list[2]):
                    denoise_2 = unsharp_mask(denoise_1, std, strength)
                    ssim_score = cpu_ssim(image, denoise_2, channel_axis=-1)
                    ssim_array_2[i2] += ssim_score
                    i2 += 1



                # Use GPU-accelerated SSIM
                ssim_score = gpu_ssim(image, denoise_1)
                ssim_array[i] += ssim_score 

                i += 1 
        processed_images += 1

    print(np.argmax(ssim_array_2/nr_of_images))
    return ssim_array, noisy_image, image, nr_of_images

def unsharp_mask(image, sigma, strength):
    #from https://www.opencvhelp.org/tutorials/image-processing/how-to-sharpen-image/
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    # Subtract the blurred image from the original
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened

def laplacian_filter(image, sigma, strength, kernel_size):
    #from https://www.opencvhelp.org/tutorials/image-processing/how-to-sharpen-image/
    # Apply Gaussian blur with specified kernel size
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    # Subtract the blurred image from the original
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened

def high_pass_filter(image, sigma):
    #from https://www.opencvhelp.org/tutorials/image-processing/how-to-sharpen-image/
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    # Subtract the blurred image from the original
    high_pass = cv2.subtract(image, blurred)
    # Add the high-pass image back to the original
    sharpened = cv2.addWeighted(image, 1.0, high_pass, 1.0, 0)
    return sharpened


def plot_ssim(avg_ssim, highest_ssim_index, std_list, kernel_list, noise_strength):
    nr_of_std = round((std_list[1]-std_list[0])/std_list[2])
    nr_of_kernels = len(kernel_list)
    
    # Set up the figure with a specific size before creating subplots
    plt.figure(figsize=(nr_of_kernels*3.5, 3.5))
    
    # Calculate global min and max for y-axis
    global_min = np.min(avg_ssim)*0.975  # Add a margin of 2.5% to the min value
    global_max = avg_ssim[highest_ssim_index]*1.025  # Add a margin of 2.5% to the max value
    
    for kernels in range(nr_of_kernels):
        ax = plt.subplot(1, nr_of_kernels, kernels+1)
        plt.axhline(y=avg_ssim[highest_ssim_index], color='k', linestyle='--')

        if highest_ssim_index >= nr_of_std*kernels and highest_ssim_index < nr_of_std*(kernels+1):
            plt.axvline(x=highest_ssim_index-(nr_of_std*kernels), color='r', linestyle='--', label='Best Parameter Index')
        
        plt.plot(avg_ssim[(nr_of_std*kernels):nr_of_std*(kernels+1)])
        plt.xticks(np.arange(0, nr_of_std, step=1), labels=[str((i+1)/(1/std_list[2])) for i in range(0, nr_of_std)], rotation=85)
        plt.ylim(global_min, global_max)  # Set the same y-axis range for all subplots
        
        # Hide y-axis labels for all but the first subplot
        if kernels > 0:
            ax.set_yticklabels([])
        
        # Add kernel size as title for each subplot
        plt.title(f'Kernel {kernel_list[kernels]}')
    
    plt.suptitle(f'Avg ssim Values for Different Kernel Sizes, Noise = {noise_strength}')
    
    # Adjust layout with finer control
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.2)
    
    # Adjust spacing between the suptitle and subplots
    plt.subplots_adjust(top=0.85)
    
    plt.savefig(f'ssim_plot{noise_strength}.png', bbox_inches='tight')
    plt.show()

      
def save_denoised_images(noisy_image, image, kernel_size, std_value, noise_strength):
    # save a few denoised images with optimal parameters
    denoise_1 = cv2.GaussianBlur(noisy_image, (kernel_size, kernel_size), std_value)
    cv2.imwrite(f"denoise_{noise_strength}_{kernel_size}_{std_value}.png", denoise_1)
    cv2.imwrite(f"noisy_image_{noise_strength}.png", noisy_image)
    cv2.imwrite(f"original_image.png", image)  

def main():
    # define kernel sizes to test
    kernel_list = [3, 5, 7] # must be odd numbers
    std_list = [0.1, 2, 0.1] # std values to test
    noise_list = [50, 100, 200, 300, 400] # noise levels to test
    strength_list = [0.1, 3, 0.1] # strength values to test

    for noise_strength in noise_list:
        print(f"Testing noise strength: {noise_strength}")
        if not os.path.exists(f"avg_ssim_{noise_strength}.txt"):
            # denoise the image using different parameters
            ssim_array, noisy_image, image, nr_of_images = denoise_image(kernel_list, std_list, noise_strength, strength_list)
            
            # calculate the average ssim values
            avg_ssim = ssim_array / nr_of_images
            # print avg ssim values
            print("Avg ssim values for different parameters:")
            print(avg_ssim)

            kernel_size, std_value, highest_ssim_value, highest_ssim_index = retrace_parameters(avg_ssim, kernel_list, std_list)

            save_denoised_images(noisy_image, image, kernel_size, std_value, noise_strength)

            # save avg ssim values to a file (flatten to avoid brackets)
            with open(f"avg_ssim_{noise_strength}.txt", "w") as f:
                for value in avg_ssim.flatten():
                    f.write(f"{value}\n")
        
        else:
            # load the avg ssim values from the file
            with open(f"avg_ssim_{noise_strength}.txt", "r") as f:
                avg_ssim = [float(line.strip()) for line in f.readlines()]
            # convert to numpy array
            avg_ssim = np.array(avg_ssim).reshape(-1, 1)  # Reshape to match original shape
            print(avg_ssim)

            kernel_size, std_value, highest_ssim_value, highest_ssim_index = retrace_parameters(avg_ssim, kernel_list, std_list)
            plot_ssim(avg_ssim, highest_ssim_index, std_list, kernel_list, noise_strength)
        

main()

