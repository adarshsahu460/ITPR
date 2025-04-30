import cv2
import numpy as np

def rgb_to_cmy(image):
    """Convert RGB image to CMY model."""
    return (1.0 - image / 255.0) * 255.0

def main():
    try:
        # Load image in RGB (OpenCV uses BGR by default)
        img = cv2.imread('input_image.jpg')
        if img is None:
            raise FileNotFoundError("Input image not found")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to CMY and save
        cmy_img = rgb_to_cmy(img_rgb).astype(np.uint8)
        cv2.imwrite('output_cmy_image.jpg', cv2.cvtColor(cmy_img, cv2.COLOR_RGB2BGR))
        print("CMY image saved as 'output_cmy_image.jpg'")
        
        # Convert to grayscale and save
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('output_gray_image.jpg', gray_img)
        print("Grayscale image saved as 'output_gray_image.jpg'")
        
    except FileNotFoundError:
        print("Error: Input image file not found")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()