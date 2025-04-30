import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_rgb_histogram(image):
    """Compute histograms for R, G, B channels of an RGB image."""
    return [cv2.calcHist([image], [i], None, [256], [0, 256]).ravel() for i in range(3)]

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

def plot_grayscale_histogram(image, filename='grayscale_histogram.png'):
    """Plot and save the histogram of a grayscale image."""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
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
    try:
        # Load image (OpenCV uses BGR by default)
        img = cv2.imread('input_image.jpg')
        if img is None:
            raise FileNotFoundError("Input image not found")
        
        # Convert BGR to RGB for histogram
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Compute and plot RGB histogram
        hist_r, hist_g, hist_b = compute_rgb_histogram(img_rgb)
        plot_rgb_histogram(hist_r, hist_g, hist_b, 'output_rgb_histogram.png')
        print("RGB histogram saved as 'output_rgb_histogram.png'")
        
        # Convert to grayscale and plot its histogram
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plot_grayscale_histogram(gray_img, 'output_grayscale_histogram.png')
        print("Grayscale histogram saved as 'output_grayscale_histogram.png'")
        
        # Apply histogram equalization and save
        equalized_img = cv2.equalizeHist(gray_img)
        cv2.imwrite('output_equalized_image.jpg', equalized_img)
        print("Equalized image saved as 'output_equalized_image.jpg'")
        
        # Plot histogram of equalized image
        plot_grayscale_histogram(equalized_img, 'output_equalized_histogram.png')
        print("Equalized histogram saved as 'output_equalized_histogram.png'")
        
    except FileNotFoundError:
        print("Error: Input image file not found")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()