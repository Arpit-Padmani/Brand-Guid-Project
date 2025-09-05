from PIL import Image
from collections import Counter

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def extract_all_colors(image_path, top_n=None):
    # Open image and convert to RGB
    img = Image.open(image_path).convert("RGB")
    pixels = list(img.getdata())
    total_pixels = len(pixels)

    # Count each color
    counter = Counter(pixels)

    # Sort by frequency
    most_common = counter.most_common(top_n) if top_n else counter.most_common()

    result = []
    for color, count in most_common:
        rgb = tuple(map(int, color))
        percent = (count / total_pixels) * 100
        result.append({
            "rgb": rgb,
            "hex": rgb_to_hex(rgb),
            "percent": round(percent, 2)
        })

    return result


# Example usage
colors = extract_all_colors("check.png", top_n=20)  # change/remove top_n to get all
for c in colors:
    print(f"RGB: {c['rgb']} | HEX: {c['hex']} | Percentage: {c['percent']}%")
