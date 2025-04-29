from PIL import Image
import math

def apply_scaling(image, scale_x=1.5, scale_y=1.5):
    """Apply scaling transformation to the image."""
    width, height = image.size
    # Define affine transformation matrix for scaling
    # [scale_x, 0, 0, 0, scale_y, 0] centers the scaling at origin
    matrix = (scale_x, 0, 0, 0, scale_y, 0)
    # Apply transformation, adjusting output size
    new_width, new_height = int(width * scale_x), int(height * scale_y)
    return image.transform((new_width, new_height), Image.AFFINE, matrix, resample=Image.BICUBIC)

def apply_rotation(image, angle_degrees=45):
    """Apply rotation transformation to the image."""
    width, height = image.size
    # Convert angle to radians
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    # Define affine transformation matrix for rotation around center
    # Translate to origin, rotate, translate back
    center_x, center_y = width / 2, height / 2
    matrix = (
        cos_a, sin_a, -center_x * cos_a - center_y * sin_a + center_x,
        -sin_a, cos_a, center_x * sin_a - center_y * cos_a + center_y
    )
    # Calculate new bounding box to avoid cropping
    new_width = int(abs(width * cos_a) + abs(height * sin_a))
    new_height = int(abs(width * sin_a) + abs(height * cos_a))
    return image.transform((new_width, new_height), Image.AFFINE, matrix, resample=Image.BICUBIC)

def apply_reflection(image):
    """Apply horizontal reflection (flip over vertical axis)."""
    width, height = image.size
    # Define affine transformation matrix for horizontal reflection
    # [ -1, 0, width, 0, 1, 0 ] flips x-coordinate
    matrix = (-1, 0, width, 0, 1, 0)
    return image.transform((width, height), Image.AFFINE, matrix, resample=Image.BICUBIC)

def main():
    # Load the input image (replace 'input_image.jpg' with your image path)
    try:
        input_image = Image.open('input_image.jpg')
        
        # Verify the image is in RGB mode (24-bit)
        if input_image.mode != 'RGB':
            print("Input image is not in RGB mode. Converting to RGB...")
            input_image = input_image.convert('RGB')
        
        # Apply scaling (1.5x)
        scaled_image = apply_scaling(input_image, scale_x=1.5, scale_y=1.5)
        scaled_image.save('output_scaled_image.jpg')
        print("Scaled image saved as 'output_scaled_image.jpg'")
        
        # Apply rotation (45 degrees)
        rotated_image = apply_rotation(input_image, angle_degrees=45)
        rotated_image.save('output_rotated_image.jpg')
        print("Rotated image saved as 'output_rotated_image.jpg'")
        
        # Apply horizontal reflection
        reflected_image = apply_reflection(input_image)
        reflected_image.save('output_reflected_image.jpg')
        print("Reflected image saved as 'output_reflected_image.jpg'")
        
    except FileNotFoundError:
        print("Error: Input image file not found. Please provide a valid image path.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()