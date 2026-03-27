# PaddedRAG Thesis Context  
  
## 1. Dataset  
- Thuc te: Du lieu tuyen sinh HUSC/Dai hoc Hue 2025.  
- Dinh dang: JSONL (raw, validated, chunked).  
  
## 2. Embedding Model  
- Model: BAAI/bge-m3  
- Dimension: 1024-dim  
  
## 3. Vector Store  
- Qdrant Cloud: hue_admissions_2025  
- Local Fallback: NumpyVectorStore  
  
## 4. Chunking Strategy  
- Adaptive Chunking: FAQ (320), Policy (450)  
  
## 5. LLM Backend  
- GLM-4.5 (Z.AI) and Groq Llama 3.3  
  
## 6. Evaluation  
- Tests: 22/27 passed. 
