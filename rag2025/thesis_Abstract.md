# Tóm tắt Luận văn (Abstract)

Sự bùng nổ của các Mô hình Ngôn ngữ Lớn (LLMs) đã thúc đẩy mạnh mẽ các phương pháp truy xuất ngữ nghĩa, tiêu biểu là kỹ thuật Sinh văn bản Tăng cường Truy xuất (RAG - Retrieval-Augmented Generation). Tuy nhiên, các kiến trúc Naive RAG hiện hữu (thường dựa trên Dense Retrieval với Vector Database) bắt đầu bộc lộ những hạn chế rõ rệt khi đối mặt với các câu hỏi đòi hỏi tính suy luận đa bước (multi-hop reasoning) hoặc tính kết nối toàn cục giữa các thực thể.

Luận văn **"Từ Naive RAG đến GraphRAG: Cơ sở lý thuyết và Tích hợp thực tiễn vào Hệ thống RAG 2025"** trình bày một lộ trình nâng cấp hệ thống máy học từ kiến trúc RAG thuần túy (FastAPI, Qdrant, BGE-M3, HyDE) lên cơ chế Lai (Hybrid) giữa Vector Database và Knowledge Graph (sử dụng Neo4j). 

Nghiên cứu tập trung vào ba phần cốt lõi:
1. **Phân tích cơ sở lý thuyết**: Đào sâu triết lý Biểu diễn Tri thức Đồ thị (Property Graph Model) và sự khác biệt về mặt kiến trúc so với không gian Vector đa chiều. Tham chiếu trực tiếp từ ấn bản *"Neo4j Essential GraphRAG"* và các công bố trên arXiv mới nhất.
2. **Kỹ thuật kết nối lai (Hybrid Context Merging)**: Thiết kế giải pháp trích xuất thực thể (Entity Extraction) để xây dựng đồ thị từ tập văn bản phi cấu trúc, đồng thời xây dựng bộ định tuyến truy vấn (Query Router) cho phép gọi đồng thời Vector Search (Qdrant) và Graph Traversal (Cypher/Neo4j).
3. **Thực nghiệm trên Hệ thống RAG 2025**: Mô hình hóa và cài đặt trực tiếp lên mã nguồn `rag2025/` đang vận hành, nhằm đánh giá Trade-off giữa chi phí tính toán (Latency/Tokens) và mức độ sâu sắc (Depth of Knowledge) mà hệ thống thu về.

Kết quả của nghiên cứu đóng góp một khung kiến trúc (Architectural Framework) mang tính ứng dụng cao, giúp cải thiện hiện tượng "Ảo giác" (Hallucination) của LLM trong các truy vấn nghiệp vụ phức tạp. Đồng thời, nghiên cứu đề xuất hướng đi tương lai về Text2Cypher và Cấu trúc bản thể linh hoạt (Dynamic Ontologies).

**Từ khóa (Keywords):** Large Language Models (LLM), Retrieval-Augmented Generation (RAG), GraphRAG, Neo4j, Qdrant, Dense Retrieval, Knowledge Graph, Multi-hop Reasoning.