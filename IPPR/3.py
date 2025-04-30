import cv2
import numpy as np

def apply_scaling(image, scale_x=1.5, scale_y=1.5):
    """Apply scaling transformation to the image."""
    height, width = image.shape[:2]
    new_size = (int(width * scale_x), int(height * scale_y))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)

def apply_rotation(image, angle_degrees=45):
    """Apply rotation transformation to the image."""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos_a = np.abs(matrix[0, 0])
    sin_a = np.abs(matrix[0, 1])
    new_width = int(width * cos_a + height * sin_a +.5)
    new_height = int(width * sin_a + height * cos_a +.5)
    matrix[0, 2] += (new_width - width) / 2
    matrix[1, 2] += (new_height - height) / 2
    return cv2.warpAffine(image, matrix, (new_width, new_height), flags=cv2.INTER_CUBIC)

def apply_reflection(image):
    """Apply horizontal reflection (flip over vertical axis)."""
    return cv2.flip(image, 1)

def main():
    try:
        # Load image (OpenCV uses BGR by default)
        img = cv2.imread('input_image.jpg')
        if img is None:
            raise FileNotFoundError("Input image not found")
        
        # Apply scaling (1.5x)
        scaled_img = apply_scaling(img, scale_x=1.5, scale_y=1.5)
        cv2.imwrite('output_scaled_image.jpg', scaled_img)
        print("Scaled image saved as 'output_scaled_image.jpg'")
        
        # Apply rotation (45 degrees)
        rotated_img = apply_rotation(img, angle_degrees=45)
        cv2.imwrite('output_rotated_image.jpg', rotated_img)
        print("Rotated image saved as 'output_rotated_image.jpg'")
        
        # Apply horizontal reflection
        reflected_img = apply_reflection(img)
        cv2.imwrite('output_reflected_image.jpg', reflected_img)
        print("Reflected image saved as 'output_reflected_image.jpg'")
        
    except FileNotFoundError:
        print("Error: Input image file not found")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()