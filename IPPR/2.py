import cv2
import numpy as np

def rgb_to_negative(image):
    """Convert RGB image to its negative."""
    return 255 - image

def main():
    try:
        # Load image (OpenCV uses BGR by default)
        img = cv2.imread('input_image.jpg')
        if img is None:
            raise FileNotFoundError("Input image not found")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to 24-bit negative and save
        negative_img = rgb_to_negative(img_rgb)
        cv2.imwrite('output_negative_rgb.jpg', cv2.cvtColor(negative_img, cv2.COLOR_RGB2BGR))
        print("24-bit negative image saved as 'output_negative_rgb.jpg'")
        
        # Convert to 8-bit grayscale negative and save
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_negative_img = 255 - gray_img
        cv2.imwrite('output_negative_gray.jpg', gray_negative_img)
        print("8-bit grayscale negative image saved as 'output_negative_gray.jpg'")
        
    except FileNotFoundError:
        print("Error: Input image file not found")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()