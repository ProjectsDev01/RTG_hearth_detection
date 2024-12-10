import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the X-ray image
image_path = './images/chest-ray.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Step 1: Apply CLAHE to enhance contrast
clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(8, 8))
enhanced_image = clahe.apply(image)

# Step 2: Apply Gaussian Blur to reduce noise
blurred_image = cv2.GaussianBlur(enhanced_image, (5, 5), 1)

# Step 3: Create an elliptical mask to focus on the central region (zwiększony rozmiar elipsy)
mask = np.zeros_like(blurred_image)
rows, cols = blurred_image.shape
center = (cols // 2 + cols // 8, rows // 2)
axes_length = (cols // 4, rows // 3)  # Zwiększamy szerokość i wysokość elipsy
cv2.ellipse(mask, center, axes_length, 0, 0, 360, 255, -1)

# Apply the mask to the blurred image
masked_image = cv2.bitwise_and(blurred_image, blurred_image, mask=mask)

# Step 4: Adaptive Thresholding (zmiana parametrów)
adaptive_thresh = cv2.adaptiveThreshold(
    masked_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY_INV, 15, 8  # Zwiększamy blok do 15x15 i C do 5
)

# Alternative Otsu Thresholding
_, otsu_thresh = cv2.threshold(masked_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Step 5: Morphological Operations to clean up the image
kernel = np.ones((3,3), np.uint8)  # Powiększony kernel
morph_open = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
morph_close = cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, kernel)

# Step 6: Edge detection using Canny (detekcja przed morfologią)
edges = cv2.Canny(adaptive_thresh, 30, 100)  # Obniżone progi do 30 i 100

# Step 7: Find contours in the masked region
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a mask for the contours
heart_mask = np.zeros_like(image)
cv2.drawContours(heart_mask, contours, -1, (255), thickness=cv2.FILLED)

# Extract the region of interest (ROI)
roi = cv2.bitwise_and(image, image, mask=heart_mask)

# Plot the results
plt.figure(figsize=(12, 12))
plt.subplot(2, 3, 1), plt.imshow(image, cmap='gray'), plt.title("Original Image")
plt.subplot(2, 3, 2), plt.imshow(masked_image, cmap='gray'), plt.title("Masked Image")
plt.subplot(2, 3, 3), plt.imshow(adaptive_thresh, cmap='gray'), plt.title("Adaptive Thresholding")
plt.subplot(2, 3, 4), plt.imshow(otsu_thresh, cmap='gray'), plt.title("Otsu Thresholding")
plt.subplot(2, 3, 5), plt.imshow(morph_close, cmap='gray'), plt.title("Morphological Cleaning")
plt.subplot(2, 3, 6), plt.imshow(roi, cmap='gray'), plt.title("Detected Heart Region")
plt.savefig('./images/detected_heart_region_masked.png', bbox_inches='tight', pad_inches=0)
plt.show()
