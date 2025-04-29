from PIL import Image
import numpy as np
from scipy import ndimage

def binarize_image(image, threshold=128):
    """Convert grayscale image to binary using a threshold."""
    img_array = np.array(image).astype(np.uint8)
    binary_array = (img_array > threshold).astype(np.uint8) * 255
    return Image.fromarray(binary_array, mode='L')

def apply_morphological_operations(image):
    """Apply erosion, dilation, opening, and closing."""
    img_array = np.array(image).astype(np.uint8) // 255  # Convert to binary (0,1)
    
    # Define 3x3 square structuring element
    struct_element = np.ones((3, 3), dtype=np.uint8)
    
    # Erosion
    eroded_array = ndimage.binary_erosion(img_array, structure=struct_element).astype(np.uint8) * 255
    
    # Dilation
    dilated_array = ndimage.binary_dilation(img_array, structure=struct_element).astype(np.uint8) * 255
    
    # Opening (erosion followed by dilation)
    opened_array = ndimage.binary_opening(img_array, structure=struct_element).astype(np.uint8) * 255
    
    # Closing (dilation followed by erosion)
    closed_array = ndimage.binary_closing(img_array, structure=struct_element).astype(np.uint8) * 255
    
    return (
        Image.fromarray(eroded_array, mode='L'),
        Image.fromarray(dilated_array, mode='L'),
        Image.fromarray(opened_array, mode='L'),
        Image.fromarray(closed_array, mode='L')
    )

def main():
    # Load the input image (replace 'input_image.jpg' with your image path)
    try:
        input_image = Image.open('input_image.jpg')
        
        # Verify and convert to grayscale (8-bit)
        if input_image.mode != 'RGB':
            print("Input image is not in RGB mode. Converting to RGB...")
            input_image = input_image.convert('RGB')
        gray_image = input_image.convert('L')
        
        # Binarize the image
        binary_image = binarize_image(gray_image, threshold=128)
        binary_image.save('output_binary_image.jpg')
        print("Binary image saved as 'output_binary_image.jpg'")
        
        # Apply morphological operations
        eroded_image, dilated_image, opened_image, closed_image = apply_morphological_operations(binary_image)
        
        # Save results
        eroded_image.save('output_eroded_image.jpg')
        print("Eroded image saved as 'output_eroded_image.jpg'")
        
        dilated_image.save('output_dilated_image.jpg')
        print("Dilated image saved as 'output_dilated_image.jpg'")
        
        opened_image.save('output_opened_image.jpg')
        print("Opened image saved as 'output_opened_image.jpg'")
        
        closed_image.save('output_closed_image.jpg')
        print("Closed image saved as 'output_closed_image.jpg'")
        
    except FileNotFoundError:
        print("Error: Input image file not found. Please provide a valid image path.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()