import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

def apply_dft(image):
    """Apply DFT, threshold, and reconstruct."""
    # Compute 2D DFT
    dft = cv2.dft(image.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft, axes=(0, 1))
    
    # Visualize DFT magnitude (log scale)
    dft_magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    plt.figure(figsize=(6, 6))
    plt.imshow(np.log1p(dft_magnitude), cmap='gray')
    plt.title('DFT Magnitude Spectrum')
    plt.axis('off')
    plt.savefig('output_dft_magnitude.png')
    plt.close()
    
    # Compression: Threshold small coefficients
    threshold = np.percentile(dft_magnitude, 90)  # Keep top 10%
    dft_compressed = dft_shift * (dft_magnitude > threshold)[..., None]
    
    # Reconstruct image
    dft_ishift = np.fft.ifftshift(dft_compressed, axes=(0, 1))
    reconstructed = cv2.idft(dft_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    return np.clip(reconstructed, 0, 255).astype(np.uint8)

def apply_dct(image):
    """Apply DCT, threshold, and reconstruct."""
    # Apply DCT
    dct_array = cv2.dct(image.astype(np.float32))
    
    # Visualize DCT coefficients (log scale)
    plt.figure(figsize=(6, 6))
    plt.imshow(np.log1p(np.abs(dct_array)), cmap='gray')
    plt.title('DCT Coefficients')
    plt.axis('off')
    plt.savefig('output_dct_magnitude.png')
    plt.close()
    
    # Compression: Threshold small coefficients
    threshold = np.percentile(np.abs(dct_array), 90)  # Keep top 10%
    dct_compressed = dct_array * (np.abs(dct_array) > threshold)
    
    # Reconstruct image
    reconstructed = cv2.idct(dct_compressed)
    return np.clip(reconstructed, 0, 255).astype(np.uint8)

def apply_dwt(image, wavelet='db1', level=1):
    """Apply DWT, threshold, and reconstruct."""
    # Compute DWT
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    cA, *details = coeffs
    
    # Visualize approximation coefficients (log scale)
    cA_magnitude = np.log1p(np.abs(cA))
    plt.figure(figsize=(6, 6))
    plt.imshow(cA_magnitude, cmap='gray')
    plt.title('DWT Approximation Coefficients')
    plt.axis('off')
    plt.savefig('output_dwt_magnitude.png')
    plt.close()
    
    # Compression: Threshold detail coefficients
    threshold = np.percentile(np.abs(np.concatenate([d.ravel() for d in details])), 90)
    compressed_details = []
    for detail in details:
        cH, cV, cD = detail
        cH = cH * (np.abs(cH) > threshold)
        cV = cV * (np.abs(cV) > threshold)
        cD = cD * (np.abs(cD) > threshold)
        compressed_details.append((cH, cV, cD))
    
    # Reconstruct image
    compressed_coeffs = [cA] + compressed_details
    reconstructed = pywt.waverec2(compressed_coeffs, wavelet)
    return np.clip(reconstructed, 0, 255).astype(np.uint8)

def main():
    try:
        # Load image and convert to grayscale
        img = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError("Input image not found")
        
        # Apply DFT
        dft_img = apply_dft(img)
        cv2.imwrite('output_dft_reconstructed.jpg', dft_img)
        print("DFT reconstructed image saved as 'output_dft_reconstructed.jpg'")
        
        # Apply DCT
        dct_img = apply_dct(img)
        cv2.imwrite('output_dct_reconstructed.jpg', dct_img)
        print("DCT reconstructed image saved as 'output_dct_reconstructed.jpg'")
        
        # Apply DWT
        dwt_img = apply_dwt(img, wavelet='db1', level=1)
        cv2.imwrite('output_dwt_reconstructed.jpg', dwt_img)
        print("DWT reconstructed image saved as 'output_dwt_reconstructed.jpg'")
        
    except FileNotFoundError:
        print("Error: Input image file not found")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()