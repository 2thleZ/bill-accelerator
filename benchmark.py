import time
import cv2
import numpy as np
import sys
import os

# Add build to path
sys.path.append(os.path.join(os.getcwd(), "build"))
import bill_cuda

def benchmark():
    print("🚀 Benchmarking: CUDA vs OpenCV (CPU)")
    
    # Create a large 4K image to stress the system
    # 3840 x 2160 (4K resolution)
    width, height = 3840, 2160
    print(f"Image Size: {width}x{height} (4K)")
    
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # --- OpenCV (CPU) ---
    # Warmup
    for _ in range(3):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10)

    start_cpu = time.perf_counter()
    for _ in range(10):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh_cpu = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY, 15, 10)
        kernel = np.ones((5,5), np.uint8)
        morph_cpu = cv2.dilate(thresh_cpu, kernel, iterations=1)
    end_cpu = time.perf_counter()
    
    avg_cpu = ((end_cpu - start_cpu) / 10) * 1000
    print(f"CPU Average Latency: {avg_cpu:.2f} ms")
    
    # --- CUDA (GPU) ---
    cuda_img = bill_cuda.ImageCUDA(width, height, 3)
    
    # Warmup
    cuda_img.load_data(img)
    cuda_img.to_grayscale()
    
    # Measure Total Latency (Transfer + Compute)
    # We re-create the object to reset state (channels) properly
    start_gpu_total = time.perf_counter()
    for _ in range(10):
        # Include creation/destruction in total latency for a fair "end-to-end" test
        temp_cuda = bill_cuda.ImageCUDA(width, height, 3)
        temp_cuda.load_data(img)
        temp_cuda.to_grayscale()
        temp_cuda.apply_adaptive_threshold(15, 10.0)
        temp_cuda.apply_morphology("dilation", 5)
        result = temp_cuda.get_data()
    end_gpu_total = time.perf_counter()
    
    avg_gpu_total = ((end_gpu_total - start_gpu_total) / 10) * 1000
    print(f"GPU Average Latency (Total): {avg_gpu_total:.2f} ms")
    
    # Measure Kernel Compute Only (Approximate)
    # Re-create object to ensure clean state (3 channels)
    cuda_img = bill_cuda.ImageCUDA(width, height, 3)
    cuda_img.load_data(img) 
    cuda_img.to_grayscale()
    
    start_gpu_compute = time.perf_counter()
    for _ in range(10):
        # We just benchmark the heavy math kernels
        # Since they swap buffers, we can run them repeatedly
        cuda_img.apply_adaptive_threshold(15, 10.0)
        cuda_img.apply_morphology("dilation", 5)
        # Force sync
        cuda_img.synchronize()
    end_gpu_compute = time.perf_counter()
    
    avg_gpu_compute = ((end_gpu_compute - start_gpu_compute) / 10) * 1000
    print(f"GPU Average Compute (Kernels): {avg_gpu_compute:.2f} ms")
    
    # --- Results ---
    print("-" * 30)
    print(f"Speedup (Total Latency): {avg_cpu / avg_gpu_total:.2f}x")
    print(f"Speedup (Compute Only):  {avg_cpu / avg_gpu_compute:.2f}x")
    
    print("\n📝 Analysis:")
    if avg_cpu / avg_gpu_total > 1.0:
        print("GPU is faster overall.")
    else:
        print("GPU is slower overall due to PCIe transfer overhead.")
        print("Compute-only speedup demonstrates kernel efficiency.")
        print("Future Optimization: Pipeline OCR engine to run directly on GPU memory.")

if __name__ == "__main__":
    benchmark()
