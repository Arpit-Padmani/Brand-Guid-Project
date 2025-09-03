import streamlit as st
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Page configuration
st.set_page_config(page_title="Brand Guide AI", page_icon="🤖")

# Optional CSS styling
css = """
<style>
h1 {
    color: #4CAF50;
}
</style>
"""
st.write(css, unsafe_allow_html=True)

# Header
st.header("Brand Guide AI 🤖")

# Sidebar for PDF upload
st.sidebar.header("Upload your PDF")
uploaded_pdf = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"], key="pdf_uploader")

if uploaded_pdf:
    st.success(f"Uploaded PDF: {uploaded_pdf.name}")
    # Extract text from first page (optional)
    try:
        reader = PdfReader(uploaded_pdf)
        if len(reader.pages) > 0:
            first_page = reader.pages[0]
            text = first_page.extract_text()
            st.write("**First Page Text:**")
            st.write(text)
        else:
            st.warning("PDF has no pages.")
    except Exception as e:
        st.error(f"Error reading PDF: {e}")

# Sidebar for Image upload
st.sidebar.header("Upload an Image")
uploaded_img = st.sidebar.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], key="img_uploader")

if uploaded_img:
    st.success(f"Uploaded Image: {uploaded_img.name}")
    st.image(uploaded_img, caption="Uploaded Image", use_column_width=True)
    
    img = Image.open(uploaded_img)
    text = pytesseract.image_to_string(img)
    
    st.write(text)


