import fitz

doc = fitz.open("Essential-GraphRAG.pdf")
text = ""
for page in doc:
    text += page.get_text()

with open("pdf_text.txt", "w", encoding="utf-8") as f:
    f.write(text)
print("PDF text extracted and saved to pdf_text.txt")
