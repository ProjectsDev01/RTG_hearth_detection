import cv2
import numpy as np
from matplotlib import pyplot as plt

# Wczytaj obraz w odcieniach szarości
image_path = './images/chest-ray4.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Nie można wczytać pliku: {image_path}")

# 1. Preprocessing: Normalizacja jasności obrazu (wyrównanie histogramu)
# Wyrównanie histogramu, aby rozkład jasności był bardziej równomierny
equalized_image = cv2.equalizeHist(image)

# Alternatywnie, zastosowanie CLAHE, aby uniknąć nadmiernego kontrastowania
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# equalized_image = clahe.apply(image)

# 2. Zastosowanie CLAHE dla poprawy kontrastu (opcjonalnie, jeśli chcemy dodatkową poprawę)
clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(8, 8))
enhanced_image = clahe.apply(equalized_image)

# 3. Gaussian Blur do wygładzenia
blurred_image = cv2.GaussianBlur(enhanced_image, (5, 5), 1)

# 4. Tworzenie maski eliptycznej dla centralnego obszaru (zwiększenie rozmiaru elipsy)
mask = np.zeros_like(blurred_image)
rows, cols = blurred_image.shape
center = (cols // 2 + cols // 8, rows // 2)
axes_length = (cols // 4, rows // 3)  # mask size
cv2.ellipse(mask, center, axes_length, 0, 0, 360, 255, -1)

# Zastosowanie maski do rozmytego obrazu
masked_image = cv2.bitwise_and(blurred_image, blurred_image, mask=mask)

# 5. Adaptive Thresholding
adaptive_thresh = cv2.adaptiveThreshold(
    masked_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY_INV, 15, 8  # Zwiększenie bloku do 15x15 i C do 5
)

# Alternatywne Otsu Thresholding
_, otsu_thresh = cv2.threshold(masked_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 6. Operacje morfologiczne dla oczyszczania obrazu
kernel = np.ones((3,3), np.uint8)  # Powiększony kernel
morph_open = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
morph_close = cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, kernel)

# 7. Detekcja krawędzi za pomocą Canny (przed morfologią)
edges = cv2.Canny(adaptive_thresh, 30, 100)  # Obniżone progi do 30 i 100

# 8. Znalezienie konturów w obszarze wykrytym przez Canny
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 9. Rysowanie konturów na oryginalnym obrazie
contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Konwertowanie obrazu na kolorowy do narysowania konturów
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # Rysowanie konturów na kolorowym obrazie (zielone)

# 10. Tworzenie maski na podstawie wykrytych konturów
heart_mask = np.zeros_like(image)
cv2.drawContours(heart_mask, contours, -1, (255), thickness=cv2.FILLED)

# 11. Wyodrębnienie regionu zainteresowania (ROI)
roi = cv2.bitwise_and(image, image, mask=heart_mask)

# 12. Wyświetlenie wyników
plt.figure(figsize=(15, 10))  # Zwiększenie rozmiaru całego wykresu

# Dostosowanie do dwóch wierszy i trzech kolumn
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Oryginał")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(masked_image, cmap='gray')
plt.title("Masked Image")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(adaptive_thresh, cmap='gray')
plt.title("Adaptive Thresholding")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(otsu_thresh, cmap='gray')
plt.title("Otsu Thresholding")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(morph_close, cmap='gray')
plt.title("Morphological Cleaning")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(contour_image)
plt.title("Oryginał z konturami")
plt.axis('off')

plt.tight_layout()
plt.show()
