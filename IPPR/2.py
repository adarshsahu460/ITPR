from PIL import Image
import numpy as np

def rgb_to_negative(image):
    """Convert a 24-bit RGB image to its 24-bit negative."""
    # Convert image to numpy array
    img_array = np.array(image).astype(np.uint8)
    
    # Invert each channel: new_value = 255 - original_value
    negative_array = 255 - img_array
    
    # Convert back to PIL image
    return Image.fromarray(negative_array)

def rgb_to_grayscale_negative(image):
    """Convert a 24-bit RGB image to 8-bit grayscale negative."""
    # Convert to grayscale (8-bit, mode 'L')
    gray_image = image.convert('L')
    
    # Convert to numpy array
    gray_array = np.array(gray_image).astype(np.uint8)
    
    # Invert grayscale values
    negative_gray_array = 255 - gray_array
    
    # Convert back to PIL image
    return Image.fromarray(negative_gray_array, mode='L')

def main():
    # Load the input image (replace 'input_image.jpg' with your image path)
    try:
        input_image = Image.open('input_image.jpg')
        
        # Verify the image is in RGB mode (24-bit)
        if input_image.mode != 'RGB':
            print("Input image is not in RGB mode. Converting to RGB...")
            input_image = input_image.convert('RGB')
        
        # Convert to 24-bit negative
        negative_image = rgb_to_negative(input_image)
        negative_image.save('output_negative_rgb.jpg')
        print("24-bit negative image saved as 'output_negative_rgb.jpg'")
        
        # Convert to 8-bit grayscale negative
        gray_negative_image = rgb_to_grayscale_negative(input_image)
        gray_negative_image.save('output_negative_gray.jpg')
        print("8-bit grayscale negative image saved as 'output_negative_gray.jpg'")
        
    except FileNotFoundError:
        print("Error: Input image file not found. Please provide a valid image path.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()