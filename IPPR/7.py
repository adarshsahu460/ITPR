from PIL import Image
import numpy as np
from scipy import ndimage
try:
    from skimage import feature
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

def apply_sobel(image):
    """Apply Sobel edge detector."""
    img_array = np.array(image).astype(float)
    
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Compute gradients
    grad_x = ndimage.convolve(img_array, sobel_x)
    grad_y = ndimage.convolve(img_array, sobel_y)
    
    # Gradient magnitude
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_magnitude = np.clip(grad_magnitude, 0, 255).astype(np.uint8)
    
    return Image.fromarray(grad_magnitude, mode='L')

def apply_prewitt(image):
    """Apply Prewitt edge detector."""
    img_array = np.array(image).astype(float)
    
    # Prewitt kernels
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    
    # Compute gradients
    grad_x = ndimage.convolve(img_array, prewitt_x)
    grad_y = ndimage.convolve(img_array, prewitt_y)
    
    # Gradient magnitude
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_magnitude = np.clip(grad_magnitude, 0, 255).astype(np.uint8)
    
    return Image.fromarray(grad_magnitude, mode='L')

def apply_canny(image):
    """Apply Canny edge detector."""
    img_array = np.array(image).astype(float)
    
    if SKIMAGE_AVAILABLE:
        # Use scikit-image's Canny implementation
        edges = feature.canny(img_array, sigma=1.0, low_threshold=0.1*255, high_threshold=0.2*255)
        edges = (edges * 255).astype(np.uint8)
    else:
        # Simplified Canny (Sobel + thresholding)
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        grad_x = ndimage.convolve(img_array, sobel_x)
        grad_y = ndimage.convolve(img_array, sobel_y)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Thresholding (simplified hysteresis)
        low_threshold = 0.1 * grad_magnitude.max()
        high_threshold = 0.2 * grad_magnitude.max()
        edges = np.zeros_like(grad_magnitude)
        edges[grad_magnitude > high_threshold] = 255
        edges[(grad_magnitude > low_threshold) & (grad_magnitude <= high_threshold)] = 128
        edges = np.clip(edges, 0, 255).astype(np.uint8)
    
    return Image.fromarray(edges, mode='L')

def main():
    # Load the input image (replace 'input_image.jpg' with your image path)
    try:
        input_image = Image.open('input_image.jpg')
        
        # Verify and convert to grayscale (8-bit)
        if input_image.mode != 'RGB':
            print("Input image is not in RGB mode. Converting to RGB...")
            input_image = input_image.convert('RGB')
        gray_image = input_image.convert('L')
        
        # Apply Sobel edge detector
        sobel_image = apply_sobel(gray_image)
        sobel_image.save('output_sobel_edges.jpg')
        print("Sobel edge detection saved as 'output_sobel_edges.jpg'")
        
        # Apply Prewitt edge detector
        prewitt_image = apply_prewitt(gray_image)
        prewitt_image.save('output_prewitt_edges.jpg')
        print("Prewitt edge detection saved as 'output_prewitt_edges.jpg'")
        
        # Apply Canny edge detector
        canny_image = apply_canny(gray_image)
        canny_image.save('output_canny_edges.jpg')
        print("Canny edge detection saved as 'output_canny_edges.jpg'")
        
    except FileNotFoundError:
        print("Error: Input image file not found. Please provide a valid image path.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()