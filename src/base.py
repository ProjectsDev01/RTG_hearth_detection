import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# Load the X-ray image
image_path = './images/chest-ray.jpg'
xray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Normalize the grayscale intensities
normalized_image = cv2.normalize(xray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# Function to apply preprocessing with dynamic thresholding
def preprocess(threshold1, threshold2):
    # Apply Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(normalized_image, (5, 5), 0)

    # Enhance contrast using histogram equalization
    equalized_image = cv2.equalizeHist(blurred_image)

    # Apply Canny edge detection
    edges = cv2.Canny(equalized_image, threshold1, threshold2)
    
    return equalized_image, edges

# Initial threshold values
initial_thresh1 = 25
initial_thresh2 = 100

# Apply initial preprocessing
equalized_image, edges = preprocess(initial_thresh1, initial_thresh2)

# Create the figure and axes for interactive threshold adjustment
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(bottom=0.25)

# Display the preprocessed image
ax[0].imshow(equalized_image, cmap='gray')
ax[0].set_title("Preprocessed X-ray Image")
ax[0].axis("off")

# Display the initial edge detection result
ax[1].imshow(edges, cmap='gray')
ax[1].set_title("Edges Detected")
ax[1].axis("off")

# Placeholder for region of interest (ROI) visualization
ax_roi = ax[2]
ax_roi.imshow(edges, cmap='gray')
ax_roi.set_title("ROI Analysis")
ax_roi.axis("off")

# Create sliders for threshold adjustment
ax_slider1 = plt.axes([0.2, 0.1, 0.65, 0.03])
ax_slider2 = plt.axes([0.2, 0.05, 0.65, 0.03])

slider_thresh1 = Slider(ax_slider1, 'Threshold1', 0, 255, valinit=initial_thresh1)
slider_thresh2 = Slider(ax_slider2, 'Threshold2', 0, 255, valinit=initial_thresh2)

# Update function for sliders
def update(val):
    thresh1 = int(slider_thresh1.val)
    thresh2 = int(slider_thresh2.val)
    
    # Reapply preprocessing with new thresholds
    _, edges = preprocess(thresh1, thresh2)
    
    # Update the edge detection plot
    ax[1].imshow(edges, cmap='gray')
    ax[1].set_title("Edges Detected")
    
    # Recalculate the ROI and display
    height, width = edges.shape
    top = int(height * 0.35)
    bottom = int(height * 0.8)
    left = int(width * 0.3)
    right = int(width * 0.75)
    cropped_base = edges[top:bottom, left:right]

    # Recalculate contours for the cropped region
    contours, _ = cv2.findContours(cropped_base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on a copy of the cropped region
    contour_image = cv2.cvtColor(cropped_base, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    ax_roi.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    ax_roi.set_title("Contours on Base Region")
    
    fig.canvas.draw_idle()

# Connect the sliders to the update function
slider_thresh1.on_changed(update)
slider_thresh2.on_changed(update)

plt.show()

# Proceed to calculate the "base" region for the heart.
# Step 2: Define the "base" region for the heart
# Define the dimensions of the image
height, width = equalized_image.shape

# Estimate the region of interest (central area)
# Heart is typically in the lower central part of the chest X-ray
top = int(height * 0.35)  # Start slightly below the center of the image
bottom = int(height * 0.8)  # Cover until about 80% of the image height
left = int(width * 0.3)  # Start from 30% of the width
right = int(width * 0.75)  # Cover until 70% of the width

# Draw the base region on the original image
base_image = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for colored rectangle
cv2.rectangle(base_image, (left, top), (right, bottom), (0, 255, 0), 2)  # Green rectangle

# Display the image with the base region highlighted
# plt.figure(figsize=(6, 6))
# plt.imshow(cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB))
# plt.title("Base Region Highlighted on X-ray")
# plt.axis("off")
# plt.show()

# Adjust the base region to shift slightly to the right
shift = int(width * 0.05)  # Shift by 5% of the image width to the right
left += shift
right += shift

# Redraw the adjusted base region
adjusted_base_image = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for colored rectangle
cv2.rectangle(adjusted_base_image, (left, top), (right, bottom), (0, 255, 0), 2)  # Green rectangle

# Display the adjusted base region
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(adjusted_base_image, cv2.COLOR_BGR2RGB))
plt.title("Adjusted Base Region Highlighted on X-ray")
plt.axis("off")
plt.show()

# Proceed to more detailed analysis within this adjusted base region.
# Step 1: Crop the image to the "base" region
cropped_base = equalized_image[top:bottom, left:right]

# Step 2: Sobel Mask
sobel_x = cv2.Sobel(cropped_base, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(cropped_base, cv2.CV_64F, 0, 1, ksize=3)

# Połączenie wyników Sobela w jedno
sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)

# Step 3: Apply edge detection using the Canny algorithm
edges = cv2.Canny(cropped_base, threshold1=25, threshold2=100)

# Step 4: Find contours from the edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the detected contours on the cropped image
contour_image = cv2.cvtColor(cropped_base, cv2.COLOR_GRAY2BGR)  # Convert to BGR for drawing
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # Draw all contours in green

# Step 5: Display the results
plt.figure(figsize=(12, 6))

# Original cropped "base" region
plt.subplot(1, 4, 1)
plt.imshow(cropped_base, cmap='gray')
plt.title("Cropped Base Region")
plt.axis("off")

# Edges detected in the "base" region
plt.subplot(1, 4, 2)
plt.imshow(edges, cmap='gray')
plt.title("Edges Detected")
plt.axis("off")

# Contours drawn on the "base" region
plt.subplot(1, 4, 3)
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.title("Contours Detected")
plt.axis("off")

# sobel mask
plt.subplot(1, 4, 4)
plt.imshow(sobel_magnitude, cmap='gray')
plt.title("Połączona Maska Sobela")
plt.axis('off')

plt.tight_layout()
plt.show()
