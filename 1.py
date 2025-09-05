from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

target = "Happy Janmashtami trly Unique year"
extracted_text = """
We wish you a very Happy janmansthami 
celebration with joy. This year is truly unique for all of us.
"""

result = classifier(extracted_text, candidate_labels=[target])

print(result)

if result["labels"][0] == target:
    print("✅ Target detected")
else:
    print("❌ Not detected")
