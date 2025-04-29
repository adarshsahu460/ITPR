from PIL import Image, ImageFilter
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def apply_spatial_filters(image):
    """Apply spatial domain filters: Gaussian blur, sharpening, median."""
    # Gaussian blur (smoothing)
    gaussian_image = image.filter(ImageFilter.GaussianBlur(radius=2))
    
    # Sharpening (using a custom kernel)
    sharpening_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharp_array = ndimage.convolve(np.array(image).astype(float), sharpening_kernel)
    sharp_array = np.clip(sharp_array, 0, 255).astype(np.uint8)
    sharp_image = Image.fromarray(sharp_array, mode='L')
    
    # Median filter (de-noising)
    median_array = ndimage.median_filter(np.array(image), size=3)
    median_image = Image.fromarray(median_array, mode='L')
    
    return gaussian_image, sharp_image, median_image

def apply_frequency_filters(image):
    """Apply frequency domain filters: low-pass and high-pass."""
    img_array = np.array(image).astype(float)
    height, width = img_array.shape
    
    # Compute 2D DFT
    dft = np.fft.fft2(img_array)
    dft_shift = np.fft.fftshift(dft)
    
    # Create frequency grid
    u = np.fft.fftfreq(height, d=1/height)
    v = np.fft.fftfreq(width, d=1/width)
    U, V = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)  # Distance from center (frequency)
    
    # Low-pass filter (Gaussian)
    D0 = 20  # Cutoff frequency
    low_pass_filter = np.exp(-(D**2) / (2 * (D0**2)))
    
    # High-pass filter (1 - Gaussian low-pass)
    high_pass_filter = 1 - low_pass_filter
    
    # Visualize filters
    plt.figure(figsize=(6, 6))
    plt.imshow(low_pass_filter, cmap='gray')
    plt.title('Low-Pass Filter')
    plt.axis('off')
    plt.savefig('output_low_pass_filter.png')
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(high_pass_filter, cmap='gray')
    plt.title('High-Pass Filter')
    plt.axis('off')
    plt.savefig('output_high_pass_filter.png')
    plt.close()
    
    # Apply low-pass filter
    low_pass_dft = dft_shift * low_pass_filter
    low_pass_ishift = np.fft.ifftshift(low_pass_dft)
    low_pass_array = np.fft.ifft2(low_pass_ishift).real
    low_pass_array = np.clip(low_pass_array, 0, 255).astype(np.uint8)
    low_pass_image = Image.fromarray(low_pass_array, mode='L')
    
    # Apply high-pass filter
    high_pass_dft = dft_shift * high_pass_filter
    high_pass_ishift = np.fft.ifftshift(high_pass_dft)
    high_pass_array = np.fft.ifft2(high_pass_ishift).real
    high_pass_array = np.clip(high_pass_array, 0, 255).astype(np.uint8)
    high_pass_image = Image.fromarray(high_pass_array, mode='L')
    
    return low_pass_image, high_pass_image

def main():
    # Load the input image (replace 'input_image.jpg' with your image path)
    try:
        input_image = Image.open('input_image.jpg')
        
        # Verify and convert to grayscale (8-bit)
        if input_image.mode != 'RGB':
            print("Input image is not in RGB mode. Converting to RGB...")
            input_image = input_image.convert('RGB')
        gray_image = input_image.convert('L')
        
        # Apply spatial domain filters
        gaussian_image, sharp_image, median_image = apply_spatial_filters(gray_image)
        gaussian_image.save('output_gaussian_blur.jpg')
        sharp_image.save('output_sharpened.jpg')
        median_image.save('output_median_filter.jpg')
        print("Spatial filters applied:")
        print(" - Gaussian blur saved as 'output_gaussian_blur.jpg'")
        print(" - Sharpened image saved as 'output_sharpened.jpg'")
        print(" - Median filter saved as 'output_median_filter.jpg'")
        
        # Apply frequency domain filters
        low_pass_image, high_pass_image = apply_frequency_filters(gray_image)
        low_pass_image.save('output_low_pass.jpg')
        high_pass_image.save('output_high_pass.jpg')
        print("Frequency filters applied:")
        print(" - Low-pass filter saved as 'output_low_pass.jpg'")
        print(" - High-pass filter saved as 'output_high_pass.jpg'")
        print(" - Low-pass filter mask saved as 'output_low_pass_filter.png'")
        print(" - High-pass filter mask saved as 'output_high_pass_filter.png'")
        
    except FileNotFoundError:
        print("Error: Input image file not found. Please provide a valid image path.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()