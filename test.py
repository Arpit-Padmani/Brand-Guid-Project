import streamlit as st
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

st.set_page_config(page_title="Font Color Detector", page_icon="🎨")

st.header("🎨 Font Color Detector")

# Upload image
uploaded_img = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_img:
    st.success(f"Uploaded Image: {uploaded_img.name}")
    img = Image.open(uploaded_img).convert("RGB")
    img_cv = np.array(img)  # Keep it in RGB
    st.image(img_cv, caption="Uploaded Image", use_column_width=True)

    # Reshape pixels (all image pixels)
    pixels = img_cv.reshape(-1, 3)

    # KMeans with 3 clusters (since we know there are 3 colors)
    k =4 
    kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)
    colors = [tuple(map(int, center)) for center in kmeans.cluster_centers_]

    st.subheader("Detected Font Colors:")
    cols = st.columns(len(colors))

    for i, color in enumerate(colors):
        rgb = tuple(color)  # Already RGB
        hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)

        with cols[i]:
            st.markdown(
                f"""
                <div style="width:120px;height:80px;background-color:{hex_color};
                border-radius:8px;border:1px solid #000;"></div>
                <p style="text-align:center;">{hex_color}<br>RGB: {rgb}</p>
                """,
                unsafe_allow_html=True
            )
