# Convert color 24 bit image into 24 bit negative, 8 bit negative
import cv2
import numpy as np

# Load the 24-bit color image (default loaded as BGR in OpenCV)
image = cv2.imread('input.png')  # Replace with your image path

if image is None:
    print("Error: Image not found or path is incorrect.")
    exit()

# 1. Create 24-bit negative image
negative_24bit = 255 - image

# 2. Convert to 8-bit grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Create 8-bit grayscale negative
negative_gray = 255 - gray_image

# Convert grayscale images to BGR for uniform display (3 channels)
gray_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
negative_gray_bgr = cv2.cvtColor(negative_gray, cv2.COLOR_GRAY2BGR)

# Function to add a label above the image
def add_label(img, label, height=50):
    labeled_img = img.copy()
    labeled_img = cv2.copyMakeBorder(labeled_img, height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(labeled_img, label, (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return labeled_img

# Add labels
original_labeled = add_label(image, 'Original (24-bit Color)')
negative_24bit_labeled = add_label(negative_24bit, 'Negative (24-bit)')
gray_labeled = add_label(gray_bgr, 'Grayscale (8-bit)')
negative_gray_labeled = add_label(negative_gray_bgr, 'Negative (8-bit Gray)')

# Resize all to the same dimensions
target_size = (image.shape[1], image.shape[0] + 50)  # Account for label space
original_labeled = cv2.resize(original_labeled, target_size)
negative_24bit_labeled = cv2.resize(negative_24bit_labeled, target_size)
gray_labeled = cv2.resize(gray_labeled, target_size)
negative_gray_labeled = cv2.resize(negative_gray_labeled, target_size)

# Combine images into a grid
top_row = cv2.hconcat([original_labeled, negative_24bit_labeled])
bottom_row = cv2.hconcat([gray_labeled, negative_gray_labeled])
final_grid = cv2.vconcat([top_row, bottom_row])

# Save and show the final image
cv2.imwrite('negative_conversions_labeled.png', final_grid)
cv2.imshow('Negative Conversions with Labels', final_grid)
cv2.waitKey(0)
cv2.destroyAllWindows()
