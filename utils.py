import os
import shutil
import gdown
import numpy as np
import cv2
import matplotlib.pyplot as plt


def crop_white_background(image):

    print("Cropping the image... with initial shape: ", image.shape)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to get a binary image (First check background color)
    if gray[0, 0] == 0:
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(
            gray, 200, 255, cv2.THRESH_BINARY_INV
        )  # It has to be inverted bebcause it has issues finding obbjects in a white background

    # Find the contours of the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No object detected")
        return image

    print("Number of contours: ", len(contours))

    # ***************** This method works ok but can lead to errors
    # If the clothes are light so the person will end up being cropped too in different contours in some cases

    # for i, contour in enumerate(contours):
    #     print(f"Contour {i} area: ", cv2.contourArea(contour))

    # # Get the bounding box of the largest contour by area
    # largest_contour = max(contours, key=cv2.contourArea)

    # # Get the bounding box of the largest contour
    # x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image using the bounding box
    # cropped_img = image[y : y + h, x : x + w]

    # ************ If multiple contours are detected, we will crop the image using the bounding box that covers all contours
    # This way we can avoid cropping the person in the image

    # Get a bounding box that contains all contours
    x_min, y_min, w_max, h_max = float("inf"), float("inf"), 0, 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min, y_min = min(x_min, x), min(y_min, y)
        w_max, h_max = max(w_max, x + w), max(h_max, y + h)

    # Crop the image using the bounding box that covers all contours
    cropped_img = image[y_min:h_max, x_min:w_max]

    print("Cropped image shape: ", cropped_img.shape)

    return cropped_img


def detect_yoga_pose(image):
    # Separate RGB channels
    R_Im = image[:, :, 2]  # Red channel
    G_Im = image[:, :, 1]  # Green channel
    B_Im = image[:, :, 0]  # Blue channel

    # Line profile (vertical profile at a fixed column, e.g., 180)
    line_profile = R_Im[:, 180] if R_Im.shape[1] > 180 else R_Im[:, R_Im.shape[1] // 2]

    # Define a smoothing kernel
    k = np.array([1, 1, 1, 1, 1])

    # Apply convolution to smooth the profile
    conv_line_profile = np.convolve(line_profile, k, mode="same")

    # Plot the line profile for visualization
    plt.figure()
    plt.plot(conv_line_profile)
    plt.title("Convolved Line Profile (Red Channel)")
    plt.xlabel("Vertical Position")
    plt.ylabel("Intensity")
    plt.show()

    # Binary thresholding to isolate the person in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Check background color to adjust threshold type
    if gray[0, 0] == 0:
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Calculate profiles based on binary image
    vertical_profile = np.sum(binary, axis=1)  # Sum across width for height profile
    horizontal_profile = np.sum(binary, axis=0)  # Sum across height for width profile

    # Calculate aspect ratio
    height, width = binary.shape
    aspect_ratio = width / height

    # Pose detection logic using profiles and aspect ratios
    if aspect_ratio < 1 and max(vertical_profile) > height * 0.7:
        if max(horizontal_profile) > width * 0.5:
            return "Downward Dog"

    elif aspect_ratio > 1.5 and max(vertical_profile) < height * 0.6:
        if np.all(np.diff(horizontal_profile) < 10):  # Uniform horizontal profile
            return "Plank"

    elif aspect_ratio < 1 and max(vertical_profile) < height * 0.5:
        return "Tree Pose"

    elif aspect_ratio < 1 and max(horizontal_profile) < width * 0.8:
        if np.count_nonzero(vertical_profile > height * 0.4) > height * 0.5:
            return "Goddess Pose"

    elif (
        aspect_ratio < 1.5
        and np.count_nonzero(horizontal_profile > width * 0.5) < width * 0.7
    ):
        return "Warrior Pose"

    return "Unknown Pose"


# Google Drive folder URL
url = "https://drive.google.com/drive/folders/1sx5VT6tMf_BJ_Tt279cOlaITOA1ugDeA"

download_data = False

# Descargar las imágenes desde Google Drive, sólo la primera vez
if download_data:

    def download_drive_folder(folder_id, destination="downloads"):
        """
        Downloads all files from a public Google Drive folder.

        Parameters:
        - folder_id (str): The ID of the Google Drive folder.
        - destination (str): The directory where files will be saved.
        """
        # Construct the URL for the folder
        url = f"https://drive.google.com/drive/folders/{folder_id}"

        # Download the folder
        gdown.download_folder(
            url=url, output=destination, quiet=False, use_cookies=False
        )

        print(f'All files have been downloaded to the "{destination}" folder.')

    def filter_images(source, destination):
        """
        Moves only image files from the source directory to the destination directory.

        Parameters:
        - source (str): The source directory containing all downloaded files.
        - destination (str): The directory where only images will be saved.
        """
        # Define allowed image extensions
        extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")

        if not os.path.exists(destination):
            os.makedirs(destination)

        image_found = False
        for root, dirs, files in os.walk(source):
            for file in files:
                if file.lower().endswith(extensions):
                    full_path = os.path.join(root, file)
                    shutil.move(full_path, os.path.join(destination, file))
                    image_found = True
                    break
            if image_found:
                break

        print(f'All images have been moved to the "{destination}" folder.')

    # Replace with your Google Drive folder ID
    folder_id = "1sx5VT6tMf_BJ_Tt279cOlaITOA1ugDeA"

    # Define destination folders
    download_destination = "downloaded_folder"
    images_destination = "downloaded_images"

    # Step 1: Download the entire folder
    download_drive_folder(folder_id, download_destination)
