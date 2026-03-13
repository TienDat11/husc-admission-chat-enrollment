import urllib.request
import os

pdf_url = "https://go.neo4j.com/rs/710-RRC-335/images/Essential-GraphRAG.pdf"
urllib.request.urlretrieve(pdf_url, "Essential-GraphRAG.pdf")
print("Downloaded Essential-GraphRAG.pdf, size:", os.path.getsize("Essential-GraphRAG.pdf"))
