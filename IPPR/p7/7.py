# 7: Apply edge detection operations
# Objective: Implement edge detector operators like sobel, Prewitt, Canny etc.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
img = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error: Image not found.")
    exit()

# ------------------ Sobel Edge Detection ------------------
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)
sobel = cv2.convertScaleAbs(sobel)

# ------------------ Prewitt Edge Detection ------------------
# Define Prewitt kernels
kernelx = np.array([[1, 0, -1],
                    [1, 0, -1],
                    [1, 0, -1]], dtype=np.float32)
kernely = np.array([[1,  1,  1],
                    [0,  0,  0],
                    [-1, -1, -1]], dtype=np.float32)

prewitt_x = cv2.filter2D(img, -1, kernelx)
prewitt_y = cv2.filter2D(img, -1, kernely)
prewitt = cv2.magnitude(np.float32(prewitt_x), np.float32(prewitt_y))
prewitt = cv2.convertScaleAbs(prewitt)

# ------------------ Canny Edge Detection ------------------
canny = cv2.Canny(img, threshold1=100, threshold2=200)

# ------------------ Display All ------------------
titles = ['Original Image', 'Sobel Edges', 'Prewitt Edges', 'Canny Edges']
images = [img, sobel, prewitt, canny]

plt.figure(figsize=(10, 6))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
