from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def compute_rgb_histogram(image):
    """Compute histograms for R, G, B channels of an RGB image."""
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Compute histogram for each channel (0-255)
    hist_r, bins = np.histogram(img_array[:,:,0].ravel(), bins=256, range=(0, 256))
    hist_g, bins = np.histogram(img_array[:,:,1].ravel(), bins=256, range=(0, 256))
    hist_b, bins = np.histogram(img_array[:,:,2].ravel(), bins=256, range=(0, 256))
    
    return hist_r, hist_g, hist_b

def plot_rgb_histogram(hist_r, hist_g, hist_b, filename='rgb_histogram.png'):
    """Plot and save the RGB histograms."""
    plt.figure(figsize=(10, 6))
    plt.plot(hist_r, color='red', label='Red')
    plt.plot(hist_g, color='green', label='Green')
    plt.plot(hist_b, color='blue', label='Blue')
    plt.title('RGB Histogram')
    plt.xlabel('Intensity Value')
    plt.ylabel('Pixel Count')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def histogram_equalization(gray_image):
    """Apply histogram equalization to an 8-bit grayscale image."""
    # Convert grayscale image to numpy array
    img_array = np.array(gray_image).astype(np.uint8)
    
    # Compute histogram and cumulative distribution function (CDF)
    hist, bins = np.histogram(img_array.ravel(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]  # Normalize to 0-255
    
    # Apply equalization: map original intensities to new values
    equalized_array = np.interp(img_array.ravel(), bins[:-1], cdf_normalized)
    equalized_array = equalized_array.reshape(img_array.shape).astype(np.uint8)
    
    return Image.fromarray(equalized_array, mode='L')

def plot_grayscale_histogram(image, filename='grayscale_histogram.png'):
    """Plot and save the histogram of a grayscale image."""
    img_array = np.array(image).astype(np.uint8)
    hist, bins = np.histogram(img_array.ravel(), bins=256, range=(0, 256))
    
    plt.figure(figsize=(10, 6))
    plt.plot(hist, color='gray', label='Grayscale')
    plt.title('Grayscale Histogram')
    plt.xlabel('Intensity Value')
    plt.ylabel('Pixel Count')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def main():
    # Load the input image (replace 'input_image.jpg' with your image path)
    try:
        input_image = Image.open('input_image.jpg')
        
        # Verify the image is in RGB mode (24-bit)
        if input_image.mode != 'RGB':
            print("Input image is not in RGB mode. Converting to RGB...")
            input_image = input_image.convert('RGB')
        
        # Compute and plot RGB histogram
        hist_r, hist_g, hist_b = compute_rgb_histogram(input_image)
        plot_rgb_histogram(hist_r, hist_g, hist_b, 'output_rgb_histogram.png')
        print("RGB histogram saved as 'output_rgb_histogram.png'")
        
        # Convert to grayscale and plot its histogram
        gray_image = input_image.convert('L')
        plot_grayscale_histogram(gray_image, 'output_grayscale_histogram.png')
        print("Grayscale histogram saved as 'output_grayscale_histogram.png'")
        
        # Apply histogram equalization and save the result
        equalized_image = histogram_equalization(gray_image)
        equalized_image.save('output_equalized_image.jpg')
        print("Equalized image saved as 'output_equalized_image.jpg'")
        
        # Plot histogram of equalized image
        plot_grayscale_histogram(equalized_image, 'output_equalized_histogram.png')
        print("Equalized histogram saved as 'output_equalized_histogram.png'")
        
    except FileNotFoundError:
        print("Error: Input image file not found. Please provide a valid image path.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()