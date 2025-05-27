import re
from PIL import Image, ImageEnhance
import pytesseract
import glob
import os

# Configuration
IMAGE_FOLDER = "."
OUTPUT_FILE = "extracted_data.txt"

def extract_data_from_image(image_path, frame_number):
    """
    Extracts the long number from the blue number region and the white number region.
    """
    try:
        image = Image.open(image_path)

        # Crop region for the blue number (top region)
        blue_number_region = image.crop((0, 520, 640, 720))  # Adjust as needed
        blue_number_text = pytesseract.image_to_string(blue_number_region)
        print(f"OCR output (blue number) for {image_path}:\n{blue_number_text}")

        # Regex to extract the long number (e.g., 1747820951048)
        pattern = r"(\d{10,})"  # Matches any number with 10 or more digits
        blue_number_matches = re.findall(pattern, blue_number_text)
        print(f"Regex matches (blue number) for {image_path}: {blue_number_matches}")

        # Get the first match for the blue number
        blue_long_number = blue_number_matches[0] if blue_number_matches else None

        # Crop region for the white number (bottom-left region)
        white_number_region = image.crop((0, 680, 600, 720))  # Adjusted coordinates for bottom-left
        white_number_text = pytesseract.image_to_string(white_number_region, config="--psm 7")
        print(f"OCR output (white number) for {image_path}:\n{white_number_text}")

        # Regex to extract the frame number and long number from the white number region
        frame_pattern = r"Frame\s+(\d+)"  # Matches "Frame <number>"
        white_number_matches = re.findall(pattern, white_number_text)
        white_frame_matches = re.findall(frame_pattern, white_number_text)
        print(f"Regex matches (white number) for {image_path}: {white_number_matches}")
        print(f"Regex matches (white frame) for {image_path}: {white_frame_matches}")

        # Get the first match for the white number and frame
        white_long_number = white_number_matches[0] if white_number_matches else None
        white_frame_number = white_frame_matches[0] if white_frame_matches else None

        # Ensure both numbers are extracted
        if blue_long_number and white_long_number and white_frame_number:
            return frame_number, blue_long_number, int(white_frame_number), white_long_number
        else:
            print(f"Failed to extract data from {image_path}")
            return None
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_images():
    """
    Processes all images in the folder and extracts data.
    """
    image_paths = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "frame_*.jpg")))
    results = []

    frame_number = 0  # Start frame number from 0
    for image_path in image_paths:
        data = extract_data_from_image(image_path, frame_number)
        if data:
            frame_blue, blue_long_number, frame_white, white_long_number = data
            results.append(f"{frame_blue}, {blue_long_number}, {frame_white}, {white_long_number}")
            print(f"Extracted from {image_path}: {results[-1]}")
        else:
            print(f"Failed to extract data from {image_path}")
        frame_number += 1  # Increment the frame number for each image

    # Write results to the output file
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(results))
    print(f"Data written to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_images()