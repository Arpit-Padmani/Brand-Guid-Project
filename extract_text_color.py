import cv2
import pytesseract
import numpy as np
import webcolors
from sklearn.cluster import KMeans

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def closest_color(requested_color):
    min_distance = float("inf")
    closest_name = None
    for name in webcolors.names("css3"):
        r_c, g_c, b_c = webcolors.name_to_rgb(name)
        distance = (r_c - requested_color[0]) ** 2 + \
                   (g_c - requested_color[1]) ** 2 + \
                   (b_c - requested_color[2]) ** 2
        if distance < min_distance:
            min_distance = distance
            closest_name = name
    return closest_name

def get_dominant_background_color(image_path, n_clusters=3):
    img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # OCR detect text regions
    data = pytesseract.image_to_data(rgb_img, output_type=pytesseract.Output.DICT)

    mask = np.ones(rgb_img.shape[:2], dtype=np.uint8) * 255  # start with full background
    for i in range(len(data['text'])):
        if data['text'][i].strip() != "":
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            mask[y:y+h, x:x+w] = 0  # mask text region as non-background

    # Extract only background pixels
    background_pixels = rgb_img[mask > 0]

    if len(background_pixels) == 0:
        return None

    # Cluster background colors
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(background_pixels)
    counts = np.bincount(kmeans.labels_)
    dominant_color = kmeans.cluster_centers_[np.argmax(counts)].astype(int)

    hex_color = '#%02x%02x%02x' % tuple(dominant_color)
    color_name = closest_color(dominant_color)

    return hex_color, tuple(dominant_color), color_name

# Example usage
image_path = "test4.png"
dominant_bg = get_dominant_background_color(image_path)

if dominant_bg:
    hex_color, rgb, name = dominant_bg
    print(f"Dominant background color: HEX={hex_color} | RGB={rgb} | Name={name}")
else:
    print("No background detected")
