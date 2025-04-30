import cv2
import numpy as np

def apply_spatial_filters(image):
    """Apply spatial domain filters: Gaussian blur, sharpening, median."""
    gaussian_image = cv2.GaussianBlur(image, (5, 5), 2)
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharp_image = cv2.filter2D(image, -1, sharpening_kernel)
    median_image = cv2.medianBlur(image, 3)
    return gaussian_image, sharp_image, median_image

def apply_frequency_filters(image):
    """Apply frequency domain filters: low-pass and high-pass."""
    # Compute DFT
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Create Gaussian low-pass filter
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    D0 = 20  # Cutoff frequency
    u, v = np.indices((rows, cols))
    D = np.sqrt((u - crow)**2 + (v - ccol)**2)
    low_pass_filter = np.exp(-(D**2) / (2 * (D0**2)))
    low_pass_filter = np.repeat(low_pass_filter[:, :, np.newaxis], 2, axis=2)
    
    # High-pass filter
    high_pass_filter = 1 - low_pass_filter
    
    # Apply filters
    low_pass_dft = dft_shift * low_pass_filter
    high_pass_dft = dft_shift * high_pass_filter
    
    # Inverse DFT
    low_pass_ishift = np.fft.ifftshift(low_pass_dft)
    high_pass_ishift = np.fft.ifftshift(high_pass_dft)
    low_pass_array = cv2.idft(low_pass_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    high_pass_array = cv2.idft(high_pass_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    
    # Clip and convert to uint8
    low_pass_image = np.clip(low_pass_array, 0, 255).astype(np.uint8)
    high_pass_image = np.clip(high_pass_array, 0, 255).astype(np.uint8)
    
    return low_pass_image, high_pass_image

def main():
    try:
        # Load and convert to grayscale
        input_image = cv2.imread('input_image.jpg')
        if input_image is None:
            raise FileNotFoundError("Input image file not found.")
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        
        # Apply spatial filters
        gaussian_image, sharp_image, median_image = apply_spatial_filters(gray_image)
        cv2.imwrite('output_gaussian_blur.jpg', gaussian_image)
        cv2.imwrite('output_sharpened.jpg', sharp_image)
        cv2.imwrite('output_median_filter.jpg', median_image)
        print("Spatial filters applied:")
        print(" - Gaussian blur saved as 'output_gaussian_blur.jpg'")
        print(" - Sharpened image saved as 'output_sharpened.jpg'")
        print(" - Median filter saved as 'output_median_filter.jpg'")
        
        # Apply frequency filters
        low_pass_image, high_pass_image = apply_frequency_filters(gray_image)
        cv2.imwrite('output_low_pass.jpg', low_pass_image)
        cv2.imwrite('output_high_pass.jpg', high_pass_image)
        print("Frequency filters applied:")
        print(" - Low-pass filter saved as 'output_low_pass.jpg'")
        print(" - High-pass filter saved as 'output_high_pass.jpg'")
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()