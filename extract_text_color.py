import cv2
import pytesseract
import numpy as np
from sklearn.cluster import DBSCAN
import webcolors

# Path to Tesseract OCR (update if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def closest_color(requested_color):
    """Find the closest CSS3 color name"""
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

def extract_text_colors(image_path):
    # Load image
    img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # OCR detect text regions
    data = pytesseract.image_to_data(rgb_img, output_type=pytesseract.Output.DICT)

    collected_colors = []

    for i in range(len(data['text'])):
        if data['text'][i].strip() != "":
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            word_region = rgb_img[y:y+h, x:x+w]

            if word_region.size > 0:
                # Convert to grayscale for thresholding
                gray = cv2.cvtColor(word_region, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

                # Keep only text pixels
                text_pixels = word_region[mask > 0]

                if len(text_pixels) > 0:
                    avg_color = np.mean(text_pixels, axis=0)
                    collected_colors.append(avg_color)

    if len(collected_colors) == 0:
        return []

    collected_colors = np.array(collected_colors)

    # Cluster dynamically
    clustering = DBSCAN(eps=25, min_samples=2).fit(collected_colors)
    unique_labels = set(clustering.labels_)
    results = []

    for label in unique_labels:
        if label == -1:
            continue  # skip noise
        cluster_points = collected_colors[clustering.labels_ == label]
        avg_cluster_color = np.mean(cluster_points, axis=0).astype(int)

        hex_color = '#%02x%02x%02x' % tuple(avg_cluster_color)
        rgb = tuple(avg_cluster_color)
        color_name = closest_color(rgb)
        results.append((hex_color, rgb, color_name))

    return results


# Example usage
image_path = "test4.png"  # your image path
colors = extract_text_colors(image_path)

print("Final distinct text colors with names:")
for hex_color, rgb, name in colors:
    print(f"HEX: {hex_color} | RGB: {rgb} | Name: {name}")
