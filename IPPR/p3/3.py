# Apply geometric transformation on images

import cv2
import numpy as np

# Load input image
image = cv2.imread('input.png')  # Replace with your image path

if image is None:
    print("Error: Image not found or path is incorrect.")
    exit()

height, width = image.shape[:2]

# 1. Rotation (rotate by 45 degrees around center)
center = (width // 2, height // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

# 2. Translation (move right 100px and down 50px)
translation_matrix = np.float32([[1, 0, 100], [0, 1, 50]])
translated_image = cv2.warpAffine(image, translation_matrix, (width, height))

# 3. Scaling (scale by 0.5 and then resize back to original size)
scaled_image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
scaled_image_resized = cv2.resize(scaled_image, (width, height))

# 4. Shearing (horizontal shear)
shear_matrix = np.float32([[1, 0.5, 0], [0, 1, 0]])
sheared_image = cv2.warpAffine(image, shear_matrix, (int(width * 1.5), height))
sheared_image = cv2.resize(sheared_image, (width, height))

# 5. Reflection (horizontal flip)
reflected_image = cv2.flip(image, 1)

# Function to add label above image
def add_label(img, label):
    labeled_img = img.copy()
    # Create a blank space above the image for text
    space = 50
    labeled_img = cv2.copyMakeBorder(labeled_img, space, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(labeled_img, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return labeled_img

# Prepare labeled images
labeled_images = [
    add_label(image, 'Original'),
    add_label(rotated_image, 'Rotated'),
    add_label(translated_image, 'Translated'),
    add_label(scaled_image_resized, 'Scaled'),
    add_label(sheared_image, 'Sheared'),
    add_label(reflected_image, 'Reflected')
]

# Resize to ensure uniformity
labeled_images = [cv2.resize(img, (width, height + 50)) for img in labeled_images]

# Create a grid: 2 rows x 3 columns
row1 = cv2.hconcat(labeled_images[:3])
row2 = cv2.hconcat(labeled_images[3:])
grid_image = cv2.vconcat([row1, row2])

# Save and show the combined image
cv2.imwrite('combined_transformations_labeled.png', grid_image)
cv2.imshow('All Transformations with Labels', grid_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
