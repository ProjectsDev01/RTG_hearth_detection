import cv2
import numpy as np
from matplotlib import pyplot as plt

# Wczytaj obraz w odcieniach szarości
image_path = './images/chest-ray.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Nie można wczytać pliku: {image_path}")

# 1. Preprocessing: Normalizacja jasności obrazu (wyrównanie histogramu)
equalized_image = cv2.equalizeHist(image)

# 2. Zastosowanie CLAHE dla poprawy kontrastu
clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(8, 8))
enhanced_image = clahe.apply(equalized_image)

# 3. Gaussian Blur do wygładzenia
blurred_image = cv2.GaussianBlur(enhanced_image, (5, 5), 1)

# 4. Zastosowanie maski Sobela
sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
sobel_magnitude = np.uint8(cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX))

# 5. Progowanie
threshold_value = 50
_, thresholded_image = cv2.threshold(sobel_magnitude, threshold_value, 255, cv2.THRESH_BINARY)

# 6. Usuwanie krawędzi przy brzegach
height, width = thresholded_image.shape
margin = 20  # Margines w pikselach
thresholded_image[:margin, :] = 0  # Górna krawędź
thresholded_image[-margin:, :] = 0  # Dolna krawędź
thresholded_image[:, :margin] = 0  # Lewa krawędź
thresholded_image[:, -margin:] = 0  # Prawa krawędź

# 7. Znalezienie konturów
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 8. Filtruj kontury na podstawie lokalizacji i obszaru
center_x, center_y = width // 2, height // 2
valid_contours = []

for contour in contours:
    # Sprawdź, czy kontur znajduje się w centralnym regionie obrazu
    x, y, w, h = cv2.boundingRect(contour)
    if center_x - width // 4 < x + w // 2 < center_x + width // 4 and \
       center_y - height // 4 < y + h // 2 < center_y + height // 4:
        valid_contours.append(contour)

if not valid_contours:
    raise ValueError("Nie znaleziono konturów w centralnym obszarze.")

# Wybierz największy kontur
heart_contour = max(valid_contours, key=cv2.contourArea)

# Utwórz maskę tylko z konturem serca
heart_mask = np.zeros_like(thresholded_image)
cv2.drawContours(heart_mask, [heart_contour], -1, 255, thickness=cv2.FILLED)

# Debugowanie: Wyświetlenie pośrednich obrazów
# cv2.imshow("Thresholded Image", thresholded_image)
# cv2.imshow("Heart Mask", heart_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 9. Wyświetlenie wyników
plt.figure(figsize=(12, 6))

# Oryginalny obraz
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Oryginał")
plt.axis('off')

# Po progowaniu Sobela
plt.subplot(1, 3, 2)
plt.imshow(thresholded_image, cmap='gray')
plt.title("Po Progowaniu Sobela")
plt.axis('off')

# Kontur serca
plt.subplot(1, 3, 3)
plt.imshow(heart_mask, cmap='gray')
plt.title("Pożądane Rozwiązanie")
plt.axis('off')

plt.tight_layout()
plt.show()
