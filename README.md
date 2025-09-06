
# 🎨 Brand Guide AI 

A **Streamlit-based AI tool** to check compliance of social media posts against brand guidelines. Upload a **brand guideline PDF** and an **image**, and the app will verify **color, text** using AI and image processing.

---

## Features

- ✅ **PDF Rule Extraction**: Extracts brand rules from uploaded PDF using Google Gemini AI.
- ✅ **Color Compliance**: Checks colors in the uploaded image against the brand guidelines.
- ✅ **Text Compliance**: Performs OCR on images and verifies required text/keywords.
- ✅ **Export Reports**: Download compliance results as **JSON** or **PDF** (`Brand-Check-Report.pdf`).

---

## Demo Video

Watch the demo to see how the app works:  

https://github.com/user-attachments/assets/2c5c1bc6-f471-49ba-b827-bdd91418dd83

---

## Project Structure

```

brand_guide_checker/
│
├─ app.py                     # Streamlit main app
├─ requirements.txt           # Python dependencies
├─ sample_test_data/          # Sample PDFs and images for testing
│   ├─ sample_rule.pdf
│   ├─ post1.png
│   └─ post2.jpg
├─ scripts/                   # Individual standalone scripts
│   ├─ dominenet_color_from_image.py
│   ├─ Extarct_all_color_from_image.py
│   ├─ extract_images_from_pdf.py
│   ├─ extract_text_from_image.py
│   └─ get_font_style_image.py
├─ .env                       # Store your API key here (not committed)
└─ Brand-Check-Report.pdf     # Example exported report

````

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Arpit-Padmani/Brand-Guid-Project.git
cd Brand-Guid-Project
````

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Install **Tesseract** OCR separately:

* **Windows:**
Download and install from [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).
Add Tesseract to your system PATH.
**Example:** `C:\Program Files\Tesseract-OCR\tesseract.exe`

* **Linux (Ubuntu/Debian):**

```bash
sudo apt update
sudo apt install tesseract-ocr
```

* **macOS:**

```bash
brew install tesseract
```

4. Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

---

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

1. Upload a **Brand Guideline PDF**.
2. Upload a **social media post image** (jpg, jpeg, png).
3. View the **compliance report** on the page.
4. Export the report as **JSON** or **PDF** (`Brand-Check-Report.pdf`).

> ⚠️ **Note:** You must upload the PDF first. If you upload an image before the PDF, the app will show a warning.

---

## Dependencies

* `streamlit` – Web app framework
* `pytesseract` – OCR for text extraction (requires Tesseract OCR installed)
* `opencv-python` – Image processing
* `Pillow` – Image handling
* `PyPDF2` – PDF parsing
* `google-generativeai` – LLM for rule extraction
* `webcolors` – Color name conversion
* `scikit-learn` – KMeans clustering for dominant color
* `sentence-transformers`, `nltk` – Text processing
* `reportlab` – PDF report generation
* `python-dotenv` – Load environment variables

---

## Future Enhancements

* Implement **logo compliance checks**.
* Add **font compliance verification**.
* Support **batch image uploads**.
* Improve **color matching with tolerance**.

---

## License

MIT License © 2025
