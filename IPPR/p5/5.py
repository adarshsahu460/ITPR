# 5: Convert image into transform domain
# Objective: Learn DFT, DCT and DWT representations and their support for compression

import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

# Load grayscale image
img = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Image not found.")
    exit()

# ------------------ DFT ------------------
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_dft = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)

# ------------------ DCT ------------------
img_float = np.float32(img) / 255.0
dct = cv2.dct(img_float)
magnitude_dct = np.log(np.abs(dct) + 1)
magnitude_dct = cv2.normalize(magnitude_dct, None, 0, 255, cv2.NORM_MINMAX)

# ------------------ DWT ------------------
coeffs2 = pywt.dwt2(img, 'haar')  # Try 'db1', 'sym2' for other wavelets
cA, (cH, cV, cD) = coeffs2

# ------------------ Display ------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(magnitude_dft, cmap='gray')
plt.title('DFT Magnitude')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(magnitude_dct, cmap='gray')
plt.title('DCT Log-Magnitude')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(cA, cmap='gray')
plt.title('DWT Approximation (cA)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(cH, cmap='gray')
plt.title('DWT Horizontal Detail (cH)')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(cV, cmap='gray')
plt.title('DWT Vertical Detail (cV)')
plt.axis('off')

# Optional: Save output to file
# plt.savefig("transforms_output.png", bbox_inches='tight')

plt.tight_layout()
plt.show()
