# Apply Morphological Operations
# Objective: Calculate image matrices with mathematical and logical operations

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image and threshold to binary
img = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error: Image not found.")
    exit()

# Binary image (you can adjust threshold as needed)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Define a 3x3 square structuring element
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Apply morphological operations
erosion = cv2.erode(binary, kernel, iterations=1)
dilation = cv2.dilate(binary, kernel, iterations=1)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)

# Plot the results
titles = ['Original Binary', 'Erosion', 'Dilation', 'Opening', 'Closing', 'Gradient', 'Top Hat', 'Black Hat']
images = [binary, erosion, dilation, opening, closing, gradient, tophat, blackhat]

plt.figure(figsize=(14, 8))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
