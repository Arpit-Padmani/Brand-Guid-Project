import cv2
import pytesseract
from pytesseract import Output
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

MODEL_NAME = "storia/font-classify-onnx"
font_model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
font_feat = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
font_model.to(device)

def classify_font(crop_img):
    pil_img = Image.fromarray(crop_img).convert("RGB")
    inputs = font_feat(pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = font_model(**inputs).logits
    idx = logits.argmax(dim=-1).item()
    return font_model.config.id2label[idx]

def get_fonts_in_image(image_path):
    img = cv2.imread(image_path)
    data = pytesseract.image_to_data(img, output_type=Output.DICT, config="--psm 6")
    found_fonts = set()

    for i, word in enumerate(data['text']):
        if not word.strip():
            continue
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        crop = img[y:y+h, x:x+w]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_CUBIC)
        try:
            font_name = classify_font(crop)
            found_fonts.add(font_name)
        except:
            continue

    return list(found_fonts)

if __name__ == "__main__":
    fonts_used = get_fonts_in_image("sample_text.png")
    print("Fonts found:", fonts_used)
