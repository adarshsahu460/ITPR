# Apply spatial and frequency domain filters
# Objective: Apply image de-noising, smoothing, sharpening etc.


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
img = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error: Image not found.")
    exit()

# ------------------ Spatial Filters ------------------

# 1. Gaussian Blur (smoothing)
gaussian_blur = cv2.GaussianBlur(img, (5, 5), 0)

# 2. Median Filter (denoising)
median_blur = cv2.medianBlur(img, 5)

# 3. Laplacian Filter (sharpening)
laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian_abs = cv2.convertScaleAbs(laplacian)
sharpened = cv2.addWeighted(img, 1.0, laplacian_abs, -1.0, 0)

# ------------------ Frequency Domain Filters ------------------

# Helper: Create Ideal Low-Pass and High-Pass Masks
def create_ideal_filter(shape, radius, highpass=False):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 1, -1)
    if highpass:
        return 1 - mask
    else:
        return mask

# DFT of image
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Low-pass filtering
lowpass_mask = create_ideal_filter(img.shape, radius=30, highpass=False)
lowpass_mask_3d = np.repeat(lowpass_mask[:, :, np.newaxis], 2, axis=2)
dft_low = dft_shift * lowpass_mask_3d
low_filtered = cv2.idft(np.fft.ifftshift(dft_low))
low_filtered_img = cv2.magnitude(low_filtered[:, :, 0], low_filtered[:, :, 1])
low_filtered_img = cv2.normalize(low_filtered_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# High-pass filtering
highpass_mask = create_ideal_filter(img.shape, radius=30, highpass=True)
highpass_mask_3d = np.repeat(highpass_mask[:, :, np.newaxis], 2, axis=2)
dft_high = dft_shift * highpass_mask_3d
high_filtered = cv2.idft(np.fft.ifftshift(dft_high))
high_filtered_img = cv2.magnitude(high_filtered[:, :, 0], high_filtered[:, :, 1])
high_filtered_img = cv2.normalize(high_filtered_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# ------------------ Display Results ------------------
titles = ['Original', 'Gaussian Blur', 'Median Filter', 'Sharpened (Laplacian)',
          'Ideal Low-Pass (Freq)', 'Ideal High-Pass (Freq)']
images = [img, gaussian_blur, median_blur, sharpened, low_filtered_img, high_filtered_img]

plt.figure(figsize=(12, 8))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
