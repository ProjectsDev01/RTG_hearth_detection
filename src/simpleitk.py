import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the X-ray image
image_path = './images/chest-ray3.jpg'
xray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Normalize the grayscale intensities
normalized_image = cv2.normalize(xray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# Function to preprocess the image
def preprocess_image(image):
    # Apply Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Enhance contrast using histogram equalization
    equalized_image = cv2.equalizeHist(blurred_image)

    return equalized_image

# Function to detect the largest contour in the central area
def detect_heart_region(image):
    height, width = image.shape

    # Define the central region of interest
    central_top = int(height * 0.3)
    central_bottom = int(height * 0.7)
    central_left = int(width * 0.4)
    central_right = int(width * 0.8)
    central_region = image[central_top:central_bottom, central_left:central_right]

    # Apply binary thresholding
    _, binary_image = cv2.threshold(central_region, 100, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Get bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Adjust coordinates back to the original image
        x_min = central_left + x
        y_min = central_top + y
        x_max = x_min + w
        y_max = y_min + h

        return x_min, y_min, x_max, y_max
    
    return None

# Preprocess the image
preprocessed_image = preprocess_image(normalized_image)

# Detect the heart region
heart_region = detect_heart_region(preprocessed_image)

# Create the base region around the detected heart region if found
if heart_region:
    x_min, y_min, x_max, y_max = heart_region

    # Define the base region
    base_top = max(0, y_min - 20)
    base_bottom = min(preprocessed_image.shape[0], y_max + 20)
    base_left = max(0, x_min - 20)
    base_right = min(preprocessed_image.shape[1], x_max + 20)

    # Extract the base region
    base_region = preprocessed_image[base_top:base_bottom, base_left:base_right]

    # Detect heart region within the extracted base region
    base_contours, _ = cv2.findContours(base_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if base_contours:
        largest_base_contour = max(base_contours, key=cv2.contourArea)
        cv2.drawContours(base_region, [largest_base_contour], -1, (255, 255, 255), 2)
else:
    base_region = None

# Display results
fig, ax = plt.subplots(1, 4, figsize=(24, 6))

# Original image
ax[0].imshow(normalized_image, cmap='gray')
ax[0].set_title("Original Image")
ax[0].axis("off")

# Preprocessed image
ax[1].imshow(preprocessed_image, cmap='gray')
ax[1].set_title("Preprocessed Image")
ax[1].axis("off")

# Heart region and base
heart_overlay = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)
if heart_region:
    cv2.rectangle(heart_overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Heart bounding box
    cv2.rectangle(heart_overlay, (base_left, base_top), (base_right, base_bottom), (255, 0, 0), 2)  # Base region
ax[2].imshow(cv2.cvtColor(heart_overlay, cv2.COLOR_BGR2RGB))
ax[2].set_title("Heart Region and Base")
ax[2].axis("off")

# Extracted heart region with contour
if base_region is not None:
    ax[3].imshow(base_region, cmap='gray')
    ax[3].set_title("Extracted Heart Region")
    ax[3].axis("off")

plt.tight_layout()
plt.show()
