# Obtain image histogram and process on it

import cv2
import matplotlib.pyplot as plt

# Read image in grayscale
image = cv2.imread('image.jpeg', cv2.IMREAD_GRAYSCALE)

# Apply Histogram Equalization
equalized = cv2.equalizeHist(image)

# Plotting images and histograms
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

# Equalized Image
plt.subplot(2, 2, 2)
plt.title("Equalized Image")
plt.imshow(equalized, cmap='gray')
plt.axis('off')

# Histogram of Original Image
plt.subplot(2, 2, 3)
plt.title("Histogram - Original")
plt.hist(image.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

# Histogram of Equalized Image
plt.subplot(2, 2, 4)
plt.title("Histogram - Equalized")
plt.hist(equalized.ravel(), bins=256, range=(0, 256), color='green', alpha=0.7)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()