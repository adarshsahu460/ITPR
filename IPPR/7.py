import cv2
import numpy as np

def apply_sobel(image):
    """Apply Sobel edge detector."""
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = cv2.magnitude(grad_x, grad_y)
    return np.clip(grad_magnitude, 0, 255).astype(np.uint8)

def apply_prewitt(image):
    """Apply Prewitt edge detector (approximated via custom kernels)."""
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    grad_x = cv2.filter2D(image, cv2.CV_64F, prewitt_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, prewitt_y)
    grad_magnitude = cv2.magnitude(grad_x, grad_y)
    return np.clip(grad_magnitude, 0, 255).astype(np.uint8)

def apply_canny(image):
    """Apply Canny edge detector."""
    edges = cv2.Canny(image, 25.5, 51.0)  # low_threshold=0.1*255, high_threshold=0.2*255
    return edges

def main():
    try:
        # Load and convert to grayscale
        input_image = cv2.imread('input_image.jpg')
        if input_image is None:
            raise FileNotFoundError("Input image file not found.")
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detectors
        sobel_image = apply_sobel(gray_image)
        prewitt_image = apply_prewitt(gray_image)
        canny_image = apply_canny(gray_image)
        
        # Save results
        cv2.imwrite('output_sobel_edges.jpg', sobel_image)
        cv2.imwrite('output_prewitt_edges.jpg', prewitt_image)
        cv2.imwrite('output_canny_edges.jpg', canny_image)
        
        print("Edge detection applied:")
        print(" - Sobel edge detection saved as 'output_sobel_edges.jpg'")
        print(" - Prewitt edge detection saved as 'output_prewitt_edges.jpg'")
        print(" - Canny edge detection saved as 'output_canny_edges.jpg'")
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()