import cv2
import numpy as np
from matplotlib import pyplot as plt

# Wczytaj obraz w odcieniach szarości
image_path = './images/chest-ray.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Nie można wczytać pliku: {image_path}")

# 1. Zastosowanie CLAHE dla poprawy kontrastu
clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(8, 8))
enhanced_image = clahe.apply(image)

# 2. Gaussian Blur do wygładzenia
blurred_image = cv2.GaussianBlur(enhanced_image, (5, 5), 1)

# Wymiary obrazu
height, width = image.shape

# Ustal środkowy obszar do analizy (np. środkowe 50% szerokości i wysokości)
center_x, center_y = width // 2, height // 2
half_width, half_height = width // 3, height // 3
roi = blurred_image[center_y - half_height:center_y + half_height, center_x - half_width:center_x + half_width]

# 3. Zastosowanie maski Sobela do wykrywania krawędzi w środkowym obszarze
sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)

# Połączenie wyników Sobela w jedno
sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)

# Konwertowanie do 8-bitowego obrazu
sobel_magnitude = np.uint8(np.absolute(sobel_magnitude))

# Parametr progowania - zmiana tej wartości pomoże dostosować, które krawędzie są uważane za istotne
threshold_value = 110  # <-- Możesz zmieniać ten próg, aby testować granicę istotnych krawędzi

# 4. Progowanie - Wyodrębnianie istotnych krawędzi
_, thresholded_image = cv2.threshold(sobel_magnitude, threshold_value, 255, cv2.THRESH_BINARY)

# 5. Wyświetlenie wyników
plt.figure(figsize=(12, 12))

# Oryginalny obraz
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Oryginał")
plt.axis('off')

# CLAHE obraz
plt.subplot(2, 3, 2)
plt.imshow(enhanced_image, cmap='gray')
plt.title("CLAHE")
plt.axis('off')

# Gaussian Blur obraz
plt.subplot(2, 3, 3)
plt.imshow(blurred_image, cmap='gray')
plt.title("Gaussian Blur")
plt.axis('off')

# Wybrany środkowy obszar
plt.subplot(2, 3, 4)
plt.imshow(roi, cmap='gray')
plt.title("Środkowy Obszar")
plt.axis('off')

# Połączona maska Sobela w środkowym obszarze
plt.subplot(2, 3, 5)
plt.imshow(sobel_magnitude, cmap='gray')
plt.title("Połączona Maska Sobela (Środek)")
plt.axis('off')

# Maska po progowaniu (istotne krawędzie w środkowym obszarze)
plt.subplot(2, 3, 6)
plt.imshow(thresholded_image, cmap='gray')
plt.title("Filtracja Krawędzi (Progowanie Środek)")
plt.axis('off')

plt.tight_layout()
plt.show()
