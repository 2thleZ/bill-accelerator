import os
import sys
import numpy as np
import cv2
import easyocr
import re

# Add build directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))

try:
    import bill_cuda
except ImportError:
    print("Warning: Could not import bill_cuda. Ensure it is built.")
    bill_cuda = None

class BillProcessor:
    def __init__(self, use_gpu=True):
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)
        self.use_cuda = (bill_cuda is not None) and use_gpu
        print(f"BillProcessor initialized. CUDA Preprocessing: {self.use_cuda}")

    def preprocess(self, image):
        """
        Preprocess image using CUDA kernels if available.
        """
        if not self.use_cuda:
            # Fallback to OpenCV
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Simple adaptive threshold
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
            return processed

        # CUDA Path
        h, w = image.shape[:2]
        channels = 3 if image.ndim == 3 else 1
        
        cuda_img = bill_cuda.ImageCUDA(w, h, channels)
        cuda_img.load_data(image)
        
        # Pipeline: Grayscale -> Adaptive Threshold -> Dilation (to connect chars)
        cuda_img.to_grayscale()
        cuda_img.apply_adaptive_threshold(15, 10.0)
        # Optional: mild dilation to thicken text for OCR
        # cuda_img.apply_morphology("dilation", 3) 
        
        return cuda_img.get_data()

    def extract_text(self, image_path):
        """
        Full pipeline: Load -> Preprocess -> OCR -> Parse
        """
        # Load Image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Preprocess
        processed_img = self.preprocess(img)
        
        # OCR
        # detail=0 returns just the list of strings
        results = self.reader.readtext(processed_img, detail=0)
        
        return results

    def parse_bill(self, text_lines):
        """
        Simple regex parser to find items and prices.
        """
        items = []
        total = 0.0
        
        # Regex for price: looks for numbers with decimal like 12.99
        price_pattern = re.compile(r'(\d+\.\d{2})')
        
        for line in text_lines:
            # Find price
            price_match = price_pattern.search(line)
            if price_match:
                price = float(price_match.group(1))
                # Assume everything before price is item name
                item_name = line[:price_match.start()].strip()
                # Filter out noise
                if len(item_name) > 2:
                    items.append({"item": item_name, "price": price})
        
        return items

if __name__ == "__main__":
    # Test
    processor = BillProcessor()
    # Create a dummy image for testing if no file exists
    dummy = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.putText(dummy, "Burger 10.50", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    processed = processor.preprocess(dummy)
    print(f"Processed shape: {processed.shape}")
    
    # We can't easily test OCR on dummy noise, but we can verify the pipeline runs
