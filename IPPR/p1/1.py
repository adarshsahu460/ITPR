import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('input.png')
if image is None:
    print("Image not found.")
    exit()

# Convert to RGB
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Split RGB
R, G, B = cv2.split(rgb)

# Construct RGB channel images
R_img = np.stack([R, np.zeros_like(R), np.zeros_like(R)], axis=2)
G_img = np.stack([np.zeros_like(G), G, np.zeros_like(G)], axis=2)
B_img = np.stack([np.zeros_like(B), np.zeros_like(B), B], axis=2)

# Convert to CMY
C = 255 - R
M = 255 - G
Y = 255 - B

# Construct CMY channel images with true CMY tints
C_img = np.stack([np.zeros_like(C), C, C], axis=2)  # Cyan = G+B
M_img = np.stack([M, np.zeros_like(M), M], axis=2)  # Magenta = R+B
Y_img = np.stack([Y, Y, np.zeros_like(Y)], axis=2)  # Yellow = R+G

# Full CMY image
cmy_img = 255 - rgb

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Plot
titles = [
    "Original RGB", "Red Channel", "Green Channel", "Blue Channel",
    "CMY Image", "Cyan Channel", "Magenta Channel", "Yellow Channel",
    "Grayscale"
]
images = [
    rgb, R_img.astype(np.uint8), G_img.astype(np.uint8), B_img.astype(np.uint8),
    cmy_img, C_img.astype(np.uint8), M_img.astype(np.uint8), Y_img.astype(np.uint8),
    gray
]

plt.figure(figsize=(15, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    if i == 8:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
