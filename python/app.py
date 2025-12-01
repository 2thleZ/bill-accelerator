import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
from ocr_engine import BillProcessor

st.set_page_config(page_title="Bill Accelerator", layout="wide")

st.title("⚡ Bill Accelerator: HPC-Powered OCR")
st.markdown("""
This application uses **custom CUDA kernels** for high-performance image preprocessing 
before feeding data into a Deep Learning OCR model.
""")

# Sidebar
st.sidebar.header("Configuration")
use_gpu = st.sidebar.checkbox("Use GPU Acceleration", value=True)

@st.cache_resource
def get_processor():
    return BillProcessor(use_gpu=True) # Always init with GPU if available

try:
    processor = get_processor()
    st.sidebar.success("Engine Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"Failed to load engine: {e}")
    st.stop()

# Main Interface
uploaded_file = st.file_uploader("Upload a Receipt/Bill", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Convert to CV2 format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, channels="BGR")
        
    with col2:
        st.subheader("GPU Preprocessed (Adaptive Threshold)")
        # Run preprocessing
        with st.spinner("Running CUDA Kernels..."):
            processed = processor.preprocess(image)
            st.image(processed, caption="Binarized Output")

    # OCR Section
    if st.button("Extract & Parse"):
        with st.spinner("Running OCR..."):
            # Save temp file for EasyOCR (it likes paths or bytes, but we have array)
            # Actually EasyOCR reader.readtext accepts numpy array too
            results = processor.reader.readtext(processed, detail=0)
            
            st.subheader("Extracted Text")
            st.write(results)
            
            st.subheader("Parsed Items")
            items = processor.parse_bill(results)
            if items:
                st.table(items)
                total = sum(item['price'] for item in items)
                st.metric("Total Bill", f"${total:.2f}")
            else:
                st.warning("No items parsed. Try a clearer image.")

    # Chat Interface
    st.divider()
    st.subheader("Chat with your Bill")
    user_input = st.text_input("Ask something about the bill (e.g., 'Split this 3 ways')")
    if user_input:
        st.write(f"**AI**: I see you want to {user_input}. (LLM integration placeholder)")

