from sentence_transformers import SentenceTransformer, util
from PIL import Image
import pytesseract
import nltk
# nltk.download("punkt")
from nltk.tokenize import word_tokenize

# Load semantic model
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_image(image_path):
    # Set the path to the tesseract executable if it's not in your system PATH
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    # Open the image file
    img = Image.open(image_path)

    # Perform OCR on the image
    text = pytesseract.image_to_string(img)

    return text

def check_word_level(extracted_text, target):
    # Tokenize both texts into words
    extracted_words = set(word_tokenize(extracted_text.lower()))
    target_words = set(word_tokenize(target.lower()))

    matched = target_words.intersection(extracted_words)
    missing = target_words - extracted_words

    return matched, missing

# Example usage
image_file = 'check.png'  # Replace with the actual path to your image
extracted_text = extract_text_from_image(image_file)
target = "Happy Janmashtami Unique hello"

# Check with word-level matching
matched_words, missing_words = check_word_level(extracted_text, target)

print("\nWord-level check:")
if matched_words:
    print("✅ Found words:", matched_words)
    if missing_words:
        print("❌ Missing words:", missing_words)
    else:
        print("🎉 All target words found")
else:
    print("⚠️ No target words found")
