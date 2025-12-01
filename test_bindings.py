import sys
import os
import numpy as np

# Add build directory to path to find the .so file
sys.path.append(os.path.join(os.getcwd(), "build"))

try:
    import bill_cuda
    print("Successfully imported bill_cuda!")
except ImportError as e:
    print(f"Failed to import bill_cuda: {e}")
    sys.exit(1)

def test_grayscale():
    width, height = 100, 100
    channels = 3
    
    # Create random RGB image
    img = np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)
    
    # Initialize CUDA object
    cuda_img = bill_cuda.ImageCUDA(width, height, channels)
    
    # Load data
    cuda_img.load_data(img)
    print("Data loaded to GPU.")
    
    # Process
    cuda_img.to_grayscale()
    print("Converted to grayscale on GPU.")
    
    # Get result
    result = cuda_img.get_data()
    print(f"Result shape: {result.shape}")
    
    if result.ndim == 2 and result.shape == (height, width):
        print("PASS: Output shape is correct.")
    else:
        print("FAIL: Output shape is incorrect.")

def test_adaptive_threshold():
    width, height = 100, 100
    channels = 1
    
    # Create a gradient image (0 to 255)
    x = np.linspace(0, 255, width)
    img = np.tile(x, (height, 1)).astype(np.uint8)
    
    # Add some "text" (dark pixels) on top of the gradient
    # Left side (dark background): Text is 0, Background is ~0-50
    img[20:30, 20:30] = 0
    # Right side (light background): Text is 100, Background is ~200-255
    img[20:30, 80:90] = 100
    
    cuda_img = bill_cuda.ImageCUDA(width, height, channels)
    cuda_img.load_data(img)
    
    # Apply adaptive threshold (Window 15, C=10)
    cuda_img.apply_adaptive_threshold(15, 10.0)
    
    result = cuda_img.get_data()
    
    # Check if "text" regions are black (0) and background is white (255)
    # Left text
    left_text_mean = np.mean(result[20:30, 20:30])
    # Right text
    right_text_mean = np.mean(result[20:30, 80:90])
    
    print(f"Left Text Mean (Should be near 0): {left_text_mean}")
    print(f"Right Text Mean (Should be near 0): {right_text_mean}")
    
    if left_text_mean < 10 and right_text_mean < 10:
        print("PASS: Adaptive Thresholding worked.")
    else:
        print("FAIL: Adaptive Thresholding failed.")

def test_morphology():
    width, height = 100, 100
    channels = 1
    
    # Create a black image with a white square in the center
    img = np.zeros((height, width), dtype=np.uint8)
    img[40:60, 40:60] = 255
    
    cuda_img = bill_cuda.ImageCUDA(width, height, channels)
    cuda_img.load_data(img)
    
    # Test Dilation (Should expand white region)
    cuda_img.apply_morphology("dilation", 5)
    result_dilated = cuda_img.get_data()
    
    # Check a pixel that was black but should now be white (e.g., 38, 38)
    # Original square: 40-60. Window 5 means +/- 2 expansion.
    # So 38-62 should be white.
    if result_dilated[38, 38] == 255:
        print("PASS: Dilation expanded the region.")
    else:
        print(f"FAIL: Dilation did not expand region. Pixel value: {result_dilated[38, 38]}")

    # Reset image
    cuda_img.load_data(img)
    
    # Test Erosion (Should shrink white region)
    cuda_img.apply_morphology("erosion", 5)
    result_eroded = cuda_img.get_data()
    
    # Check a pixel that was white but should now be black (e.g., 40, 40)
    # Original square: 40-60. Window 5 means +/- 2 shrinkage.
    # So 40, 41 should be eroded?
    # Window 5: center needs +/- 2 neighbors to be white.
    # At 40,40: neighbors at 38,38 are black. So min is 0.
    if result_eroded[40, 40] == 0:
        print("PASS: Erosion shrank the region.")
    else:
        print(f"FAIL: Erosion did not shrink region. Pixel value: {result_eroded[40, 40]}")

if __name__ == "__main__":
    test_grayscale()
    print("-" * 20)
    test_adaptive_threshold()
    print("-" * 20)
    test_morphology()
