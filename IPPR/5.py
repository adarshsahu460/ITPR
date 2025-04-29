from PIL import Image
import numpy as np
import pywt
import matplotlib.pyplot as plt

def apply_dft(image):
    """Apply DFT, threshold, and reconstruct."""
    img_array = np.array(image).astype(float)
    
    # Compute 2D DFT
    dft = np.fft.fft2(img_array)
    dft_shift = np.fft.fftshift(dft)  # Shift zero frequency to center
    
    # Visualize DFT magnitude (log scale)
    dft_magnitude = np.log1p(np.abs(dft_shift))
    plt.figure(figsize=(6, 6))
    plt.imshow(dft_magnitude, cmap='gray')
    plt.title('DFT Magnitude Spectrum')
    plt.axis('off')
    plt.savefig('output_dft_magnitude.png')
    plt.close()
    
    # Compression: Threshold small coefficients
    threshold = np.percentile(np.abs(dft_shift), 90)  # Keep top 10%
    dft_compressed = dft_shift * (np.abs(dft_shift) > threshold)
    
    # Reconstruct image
    dft_ishift = np.fft.ifftshift(dft_compressed)
    reconstructed = np.fft.ifft2(dft_ishift).real
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    return Image.fromarray(reconstructed, mode='L')

def apply_dct(image):
    """Apply DCT, threshold, and reconstruct."""
    img_array = np.array(image).astype(float)
    height, width = img_array.shape
    
    # Apply DCT on 8x8 blocks
    block_size = 8
    dct_array = np.zeros_like(img_array)
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = img_array[i:i+block_size, j:j+block_size]
            if block.shape == (block_size, block_size):
                dct_block = dct2(block)
                dct_array[i:i+block_size, j:j+block_size] = dct_block
    
    # Visualize DCT coefficients (log scale)
    dct_magnitude = np.log1p(np.abs(dct_array))
    plt.figure(figsize=(6, 6))
    plt.imshow(dct_magnitude, cmap='gray')
    plt.title('DCT Coefficients')
    plt.axis('off')
    plt.savefig('output_dct_magnitude.png')
    plt.close()
    
    # Compression: Threshold small coefficients
    threshold = np.percentile(np.abs(dct_array), 90)  # Keep top 10%
    dct_compressed = dct_array * (np.abs(dct_array) > threshold)
    
    # Reconstruct image
    recon_array = np.zeros_like(img_array)
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = dct_compressed[i:i+block_size, j:j+block_size]
            if block.shape == (block_size, block_size):
                recon_block = idct2(block)
                recon_array[i:i+block_size, j:j+block_size] = recon_block
    
    recon_array = np.clip(recon_array, 0, 255).astype(np.uint8)
    return Image.fromarray(recon_array, mode='L')

def dct2(block):
    """2D DCT on a block."""
    return np.fft.fft2(block).real  # Simplified DCT approximation

def idct2(block):
    """2D inverse DCT on a block."""
    return np.fft.ifft2(block).real

def apply_dwt(image, wavelet='db1', level=1):
    """Apply DWT, threshold, and reconstruct."""
    img_array = np.array(image).astype(float)
    
    # Compute DWT
    coeffs = pywt.wavedec2(img_array, wavelet, level=level)
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
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    return Image.fromarray(reconstructed, mode='L')

def main():
    # Load the input image (replace 'input_image.jpg' with your image path)
    try:
        input_image = Image.open('input_image.jpg')
        
        # Verify and convert to grayscale (8-bit)
        if input_image.mode != 'RGB':
            print("Input image is not in RGB mode. Converting to RGB...")
            input_image = input_image.convert('RGB')
        gray_image = input_image.convert('L')
        
        # Apply DFT
        dft_image = apply_dft(gray_image)
        dft_image.save('output_dft_reconstructed.jpg')
        print("DFT reconstructed image saved as 'output_dft_reconstructed.jpg'")
        
        # Apply DCT
        dct_image = apply_dct(gray_image)
        dct_image.save('output_dct_reconstructed.jpg')
        print("DCT reconstructed image saved as 'output_dct_reconstructed.jpg'")
        
        # Apply DWT
        dwt_image = apply_dwt(gray_image, wavelet='db1', level=1)
        dwt_image.save('output_dwt_reconstructed.jpg')
        print("DWT reconstructed image saved as 'output_dwt_reconstructed.jpg'")
        
    except FileNotFoundError:
        print("Error: Input image file not found. Please provide a valid image path.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()