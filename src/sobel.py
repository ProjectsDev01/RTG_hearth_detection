import cv2
import numpy as np
from matplotlib import pyplot as plt

# Funkcja do obsługi kliknięcia myszą
def mouse_callback(event, x, y, flags, param):
    global selected_edges, original_edge_image, contours
    if event == cv2.EVENT_LBUTTONDOWN:
        # Znajdź kontur najbliższy klikniętemu punktowi
        min_distance = float('inf')
        selected_contour = None
        for contour in contours:
            for point in contour:
                dist = np.linalg.norm(np.array([x, y]) - point[0])
                if dist < min_distance:
                    min_distance = dist
                    selected_contour = contour

        # Jeśli znaleziono kontur, dodaj go do wynikowego obrazu
        if selected_contour is not None:
            mask = np.zeros_like(original_edge_image, dtype=np.uint8)
            cv2.drawContours(mask, [selected_contour], -1, 255, thickness=cv2.FILLED)
            selected_edges = cv2.bitwise_or(selected_edges, mask)

        # Pokaż zaznaczone krawędzie na obrazie
        display_image = cv2.addWeighted(original_edge_image, 0.5, selected_edges, 0.5, 0)
        cv2.imshow("Kliknij krawędzie konturu serca", display_image)

# 1. Wczytaj obraz w odcieniach szarości
image_path = './images/chest-ray.jpg'  # Ustaw ścieżkę do pliku
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Nie można wczytać pliku: {image_path}")

# 2. Normalizacja jasności obrazu
normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

# 3. CLAHE (dla wzmocnienia lokalnego kontrastu)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_image = clahe.apply(normalized_image)

# 4. Gaussian Blur do wygładzenia szumów
blurred_image = cv2.GaussianBlur(enhanced_image, (5, 5), 1)

# 5. Maska Sobela
sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
sobel_magnitude = np.uint8(cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX))

# 6. Progowanie
threshold_value = 50
_, edge_image = cv2.threshold(sobel_magnitude, threshold_value, 255, cv2.THRESH_BINARY)

# 7. Usunięcie krawędzi przy brzegach
margin = 20
edge_image[:margin, :] = 0
edge_image[-margin:, :] = 0
edge_image[:, :margin] = 0
edge_image[:, -margin:] = 0

# Zachowaj oryginalny obraz progowania
original_edge_image = edge_image.copy()

# 8. Znalezienie konturów
contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 9. Kopia obrazu do wyświetlania i zbierania kliknięć
selected_edges = np.zeros_like(edge_image)

cv2.imshow("Kliknij krawędzie konturu serca", original_edge_image)
cv2.setMouseCallback("Kliknij krawędzie konturu serca", mouse_callback)

# Czekaj na kliknięcia użytkownika
print("Kliknij w kilka krawędzi konturu serca, a następnie naciśnij klawisz ESC.")
while True:
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()

# 10. Połącz zaznaczone krawędzie
final_edges = selected_edges

# 11. Wyświetlenie wyników
plt.figure(figsize=(12, 6))

# Oryginał
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Oryginał")
plt.axis('off')

# Obraz progowany
plt.subplot(1, 3, 2)
plt.imshow(original_edge_image, cmap='gray')
plt.title("Progowanie Sobela")
plt.axis('off')

# Zaznaczone krawędzie
plt.subplot(1, 3, 3)
plt.imshow(final_edges, cmap='gray')
plt.title("Wybrane Krawędzie")
plt.axis('off')

plt.tight_layout()
plt.show()
