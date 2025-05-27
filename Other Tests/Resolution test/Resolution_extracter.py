import numpy as np
import cv2 
import os

#load folder with images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

# Extract resolution from images
def extract_resolutions(images):
    resolutions = []
    for img in images:
        height, width, _ = img.shape
        resolutions.append((width, height))
    return resolutions

def main():
    folder = 'CPP_network/latency_test/frames_15_1'  # Specify the folder containing images
    images = load_images_from_folder(folder)
    
    if not images:
        print("No images found in the specified folder.")
        return
    
    resolutions = extract_resolutions(images)
    
    for i, res in enumerate(resolutions):
        print(f"Image {i+1}: Width = {res[0]}, Height = {res[1]}")

if __name__ == "__main__":
    main()