import cv2
import numpy as np

def binarize_image(image, threshold=128):
    """Convert grayscale image to binary using a threshold."""
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def apply_morphological_operations(image):
    """Apply erosion, dilation, opening, and closing."""
    # Define 3x3 square structuring element
    struct_element = np.ones((3, 3), dtype=np.uint8)
    
    # Morphological operations
    eroded_image = cv2.erode(image, struct_element)
    dilated_image = cv2.dilate(image, struct_element)
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, struct_element)
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, struct_element)
    
    return eroded_image, dilated_image, opened_image, closed_image

def main():
    try:
        # Load and convert to grayscale
        input_image = cv2.imread('input_image.jpg')
        if input_image is None:
            raise FileNotFoundError("Input image file not found.")
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        
        # Binarize the image
        binary_image = binarize_image(gray_image, threshold=128)
        cv2.imwrite('output_binary_image.jpg', binary_image)
        print("Binary image saved as 'output_binary_image.jpg'")
        
        # Apply morphological operations
        eroded_image, dilated_image, opened_image, closed_image = apply_morphological_operations(binary_image)
        
        # Save results
        cv2.imwrite('output_eroded_image.jpg', eroded_image)
        cv2.imwrite('output_dilated_image.jpg', dilated_image)
        cv2.imwrite('output_opened_image.jpg', opened_image)
        cv2.imwrite('output_closed_image.jpg', closed_image)
        
        print("Morphological operations applied:")
        print(" - Eroded image saved as 'output_eroded_image.jpg'")
        print(" - Dilated image saved as 'output_dilated_image.jpg'")
        print(" - Opened image saved as 'output_opened_image.jpg'")
        print(" - Closed image saved as 'output_closed_image.jpg'")
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()