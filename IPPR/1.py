from PIL import Image
import numpy as np

def rgb_to_cmy(image):
    """Convert a 24-bit RGB image to 24-bit CMY model."""
    # Convert image to numpy array for processing
    img_array = np.array(image).astype(float)
    
    # Normalize RGB values to [0, 1]
    img_array /= 255.0
    
    # CMY = 1 - RGB
    cmy_array = 1.0 - img_array
    
    # Convert back to 0-255 range and uint8 type
    cmy_array = (cmy_array * 255).astype(np.uint8)
    
    # Convert back to PIL image
    return Image.fromarray(cmy_array)

def rgb_to_grayscale(image):
    """Convert a 24-bit RGB image to 8-bit grayscale."""
    # Use PIL's convert method to get grayscale (luminance: 0.299R + 0.587G + 0.114B)
    gray_image = image.convert('L')  # 'L' mode is 8-bit grayscale
    return gray_image

def main():
    # Load the input image (replace 'input_image.jpg' with your image path)
    try:
        input_image = Image.open('input_image.jpg')
        
        # Verify the image is in RGB mode (24-bit)
        if input_image.mode != 'RGB':
            print("Input image is not in RGB mode. Converting to RGB...")
            input_image = input_image.convert('RGB')
        
        # Convert to CMY
        cmy_image = rgb_to_cmy(input_image)
        cmy_image.save('output_cmy_image.jpg')
        print("CMY image saved as 'output_cmy_image.jpg'")
        
        # Convert to 8-bit grayscale
        gray_image = rgb_to_grayscale(input_image)
        gray_image.save('output_gray_image.jpg')
        print("Grayscale image saved as 'output_gray_image.jpg'")
        
    except FileNotFoundError:
        print("Error: Input image file not found. Please provide a valid image path.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()