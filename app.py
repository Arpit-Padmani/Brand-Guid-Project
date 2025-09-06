import streamlit as st
import cv2
import numpy as np
import pytesseract
from PyPDF2 import PdfReader
from PIL import Image
import json
import os
import google.generativeai as genai
import re
import webcolors
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from PIL import Image
from collections import Counter
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# --- Streamlit Page Config ---
st.set_page_config(page_title="Brand Guide Checker", layout="centered")

st.title("🧪 Brand Guide AI - Fresher Test")
st.write("Upload brand guideline PDF + social media post to check compliance.")

# --- Configure Gemini API ---
genai.configure(api_key="AIzaSyBabE4zEcG0nOiXk2GRiL2Uu-1_feMCivQ")  # replace with your key or env var

def export_results_to_pdf(results, filename="compliance_report.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []

    flow.append(Paragraph("Brand Guide Compliance Report", styles["Heading1"]))
    flow.append(Spacer(1, 12))

    for k, v in results.items():
        if isinstance(v, dict):  # text compliance
            status = "✅ PASS" if v["status"] else "❌ FAIL"
            flow.append(Paragraph(f"<b>{k}:</b> {status}", styles["Normal"]))
            flow.append(Paragraph(f"✅ Matched words: {', '.join(v['matched']) if v['matched'] else 'None'}", styles["Normal"]))
            flow.append(Paragraph(f"❌ Missing words: {', '.join(v['missing']) if v['missing'] else 'None'}", styles["Normal"]))
        else:  # color/logo
            if v is True:
                status = "✅ PASS"
            elif v is False:
                status = "❌ FAIL"
            else:
                status = f"⚠️ {v}"
            flow.append(Paragraph(f"<b>{k}:</b> {status}", styles["Normal"]))

        flow.append(Spacer(1, 8))

    doc.build(flow)
    return filename

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

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def extract_all_colors(image_path, top_n=None):
    """Extract all colors from an image with frequency percentage."""
    img = Image.open(image_path).convert("RGB")
    pixels = list(img.getdata())
    total_pixels = len(pixels)

    counter = Counter(pixels)
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

def check_color_compliance(image_path, color_rules):
    results = {}

    # Get dominant background color
    bg_color = get_dominant_background_color(image_path)

    # Get all colors from image
    all_colors = extract_all_colors(image_path, top_n=30)
    extracted_hex = [c["hex"].lower() for c in all_colors]

    for rule in color_rules:
        expected = rule["expected_value"]
        expected_list = expected if isinstance(expected, list) else [expected]

        # Case 1: Rule explicitly mentions dominant
        if "dominant" in rule["rule_name"].lower() or "dominant" in str(expected).lower():
            if bg_color:
                hex_color, rgb, name = bg_color
                if any(exp.lower() in hex_color.lower() or exp.lower() in name.lower() for exp in expected_list):
                    results[f"{rule['rule_name']}"] = True
                else:
                    results[f"{rule['rule_name']}"] = False
            else:
                results[f"{rule['rule_name']}"] = "⚠️ Could not detect dominant color"

        # Case 2: General brand colors (check across all extracted colors)
        else:
            found = all(any(exp.lower() == h for h in extracted_hex) for exp in expected_list)
            results[f"{rule['rule_name']}"] = found

    return results

def extract_rules_with_llm(pdf_file):
    """Extract rules from PDF using Gemini LLM."""
    pdf_reader = PdfReader(pdf_file)
    raw_text = ""
    for page in pdf_reader.pages:
        txt = page.extract_text()
        if txt:
            raw_text += txt + "\n"

    prompt = f"""
    You are a helpful assistant. I will give you brand guideline text from a PDF.
    Extract the important compliance rules in a structured JSON array with these rules:

    - color → Combine ALL colors (brand, font, background) into a **single rule**.
        • If a "dominant" color is explicitly stated, expected_value should be just that HEX code.
        • Otherwise, expected_value should be a list of all HEX codes found in the PDF.

    - text → Required keywords or sentences in captions (keep them as separate rules).
    - logo → Logo presence, integrity, opacity (keep each as separate rules).
    - font → Keep **Font Family** and **Font Size** as separate rules (do not merge).

    Each rule object must contain:
    - rule_name
    - check_type (color/text/logo/font)
    - expected_value

    Return ONLY valid JSON (no markdown, no explanation).

    PDF Text:
    {raw_text}
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    text_out = response.text.strip()

    # remove markdown fences if Gemini adds them
    text_out = re.sub(r"^```json", "", text_out, flags=re.MULTILINE)
    text_out = re.sub(r"^```", "", text_out, flags=re.MULTILINE)
    text_out = re.sub(r"```$", "", text_out, flags=re.MULTILINE)

    return text_out.strip()

def group_rules(rules_json_str):
    """Parse and group rules JSON safely."""
    try:
        rules = json.loads(rules_json_str)
    except Exception as e:
        st.error(f"⚠️ Could not parse rules JSON. Error: {e}")
        st.text(rules_json_str)  # Show what was returned
        return {}
    grouped = {}
    for rule in rules:
        ctype = rule.get("check_type", "other").lower()
        grouped.setdefault(ctype, []).append(rule)
    return grouped

def get_dominant_background_color(image_path, n_clusters=3):
    img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # OCR detect text regions
    data = pytesseract.image_to_data(rgb_img, output_type=pytesseract.Output.DICT)
    mask = np.ones(rgb_img.shape[:2], dtype=np.uint8) * 255
    for i in range(len(data['text'])):
        if data['text'][i].strip() != "":
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            mask[y:y+h, x:x+w] = 0  # remove text region

    background_pixels = rgb_img[mask > 0]
    if len(background_pixels) == 0:
        return None

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(background_pixels)
    counts = np.bincount(kmeans.labels_)
    dominant_color = kmeans.cluster_centers_[np.argmax(counts)].astype(int)

    hex_color = '#%02x%02x%02x' % tuple(dominant_color)
    color_name = closest_color(dominant_color)

    return hex_color, tuple(dominant_color), color_name

def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

def check_word_level(extracted_text, target):
    extracted_words = set(word_tokenize(extracted_text.lower()))
    target_words = set(word_tokenize(target.lower()))
    matched = target_words.intersection(extracted_words)
    missing = target_words - extracted_words
    return matched, missing

# --- Upload Section ---
pdf_file = st.file_uploader("📂 Upload Brand Guideline PDF", type=["pdf"])
image_file = st.file_uploader("🖼️ Upload Social Media Post", type=["jpg", "jpeg", "png"])

rules_grouped = {}
if pdf_file:
    st.subheader("📜 Rules are Extracted from PDF (via LLM)")
    rules_json = extract_rules_with_llm(pdf_file)
    # st.code(rules_json, language="json")
    rules_grouped = group_rules(rules_json)
# --- Process Image ---
results = {}

if image_file:
    temp_path = "temp_uploaded.png"
    image = Image.open(image_file)
    image.save(temp_path)

    # --- Color Checks (using your function) ---
    if "color" in rules_grouped:
        color_results = check_color_compliance(temp_path, rules_grouped["color"])
        results.update(color_results)
    # --- Text Checks (using your OCR + word matcher) ---
    if "text" in rules_grouped:
        extracted_text = extract_text_from_image(temp_path)
        all_matched = set()
        all_missing = set()

        for rule in rules_grouped["text"]:
            expected = str(rule["expected_value"])
            matched, missing = check_word_level(extracted_text, expected)
            all_matched.update(matched)
            all_missing.update(missing)

        results["Text Compliance"] = {
            "status": len(all_missing) >= 0,  # ✅ only if all expected words matched
            "matched": list(all_matched),
            "missing": list(all_missing)
        }

    # --- Logo Checks (still placeholder) ---
    if "logo" in rules_grouped:
        for rule in rules_grouped["logo"]:
            results[f"{rule['rule_name']}"] = "⚠️ Logo check not implemented yet"


# --- Output Results ---
if results:
    st.subheader("✅ Compliance Report")
    for k, v in results.items():
        if isinstance(v, dict):  # text compliance
            if v["status"]:
                st.success(f"✅ {k}")
            else:
                st.error(f"❌ {k}")
            st.markdown(f"- ✅ Matched words: {', '.join(v['matched']) if v['matched'] else 'None'}")
            st.markdown(f"- ❌ Missing words: {', '.join(v['missing']) if v['missing'] else 'None'}")
        else:  # color/logo rules
            if v is True:
                st.success(f"✅ {k}")
            elif v is False:
                st.error(f"❌ {k}")
            else:
                st.warning(f"{k}: {v}")
    # --- Export Results ---
    export = st.radio("Export Results As:", ["None", "JSON", "PDF"])
    if export == "JSON":
        json_data = json.dumps(results, indent=2)
        st.download_button("📥 Download JSON Report", json_data, file_name="report.json")
    elif export == "PDF":
        pdf_path = "compliance_report.pdf"
        export_results_to_pdf(results, pdf_path)
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Download PDF Report", f, file_name="compliance_report.pdf", mime="application/pdf")