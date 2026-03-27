# Chương 2: Cơ sở Lý thuyết và Tổng quan Nghiên cứu

Trong bối cảnh hệ thống truy xuất thông tin (Information Retrieval) ngày càng phát triển, sự chuyển dịch từ các mô hình RAG truyền thống (Naive RAG) sang mô hình lấy đồ thị làm trung tâm (GraphRAG) đã phản ánh sự thay đổi sâu sắc trong cách máy móc xử lý và biểu diễn tri thức. Chương này tập trung làm rõ cơ sở lý thuyết và khía cạnh học thuật của luận án thông qua việc đánh giá các cấu trúc RAG, làm nổi bật nền tảng biểu diễn tri thức bằng đồ thị, phân tích khung lý thuyết Essential GraphRAG của Neo4j và tổng hợp các phương pháp luận trong việc kết hợp các kỹ thuật truy xuất theo hướng lai (Hybrid Retrieval).

## 2.1. Phân tích Phê phán Kiến trúc Naive RAG (Dense Retrieval với BGE-M3/Qdrant, HyDE)

Naive Retrieval-Augmented Generation (Naive RAG) dựa chủ yếu vào kiến trúc dense retrieval, nơi dữ liệu dạng văn bản được mã hóa thành các vector nhúng (embeddings) đa chiều. Theo các nghiên cứu gần đây về RAG [1], mô hình này hoạt động thông qua việc so sánh độ tương đồng cosine (cosine similarity) giữa truy vấn của người dùng và các mảnh tài liệu (chunks) được lưu trữ trong cơ sở dữ liệu vector như Qdrant. 

Dù thể hiện sức mạnh trong việc giải quyết các truy vấn mang tính ngữ nghĩa cục bộ (local semantic), Naive RAG tồn tại nhiều hạn chế cốt lõi. Hạn chế đầu tiên là sự "phân mảnh thông tin". Khi tài liệu được chia nhỏ để phù hợp với giới hạn cửa sổ ngữ cảnh (context window) của các mô hình ngôn ngữ lớn (LLMs), cấu trúc liên kết và sự liền mạch của tài liệu bị đứt gãy. Khi một truy vấn yêu cầu khả năng tổng hợp (global query) liên quan đến nhiều mảnh tài liệu hoặc nhiều nguồn, độ chính xác của Naive RAG giảm sút nghiêm trọng [2]. 

Để cải thiện hiệu năng, các kỹ thuật như Hypothetical Document Embeddings (HyDE) đã được tích hợp nhằm sinh ra các câu trả lời giả định làm cầu nối ngữ nghĩa giữa truy vấn ngắn và tài liệu phức tạp. Mặc dù vậy, theo phân tích từ [3], những cách tiếp cận này chỉ là giải pháp tạm thời (workarounds) nhằm bù đắp cho sự thiếu hụt khả năng mô hình hóa mối quan hệ trực tiếp giữa các thực thể, vốn không thể thực hiện được trong không gian vector phẳng.

## 2.2. Triết lý Biểu diễn Tri thức Đồ thị (Neo4j Property Graph Model)

Để vượt qua giới hạn không gian phẳng của dense vectors, GraphRAG giới thiệu một hệ triết lý mới dựa trên đồ thị để biểu diễn tri thức [4]. Triết lý này tập trung vào sự kết nối tự nhiên của thế giới thực: các thực thể (entities) không tồn tại độc lập mà luôn gắn kết với nhau thông qua các mối quan hệ (relationships). 

Property Graph Model của Neo4j là cấu trúc nền tảng cho kiến trúc này. Trong mô hình Property Graph, dữ liệu được cấu trúc dưới dạng:
- **Nút (Nodes):** Đại diện cho các thực thể (như Người, Tổ chức, Khái niệm). Nút có thể chứa các nhãn (labels) để định dạng vai trò trong không gian tri thức.
- **Cạnh (Edges / Relationships):** Biểu diễn mối liên hệ mang tính ngữ nghĩa giữa các nút. Cạnh luôn có tính định hướng (directed) và thuộc tính (type) cụ thể, ví dụ `(Person)-[WORKS_AT]->(Organization)`.
- **Thuộc tính (Properties):** Các cặp khóa-giá trị (key-value pairs) được gán vào cả nút và cạnh, lưu trữ thông tin meta và các mô tả định lượng/định tính.

Việc lưu trữ thông tin dưới dạng đồ thị cho phép các hệ thống AI mô hình hóa cấu trúc phân cấp (hierarchical structures) và biểu diễn tính ngữ cảnh toàn cục (global context) - điều mà Naive RAG không làm được [5]. Khi các LLM tương tác với tri thức đồ thị, thay vì nhận các vector rời rạc, chúng nhận được một mạng lưới ngữ nghĩa (semantic network) phong phú giúp sinh ra những phản hồi logic và gắn kết sâu sắc.

## 2.3. Khung Lý thuyết Neo4j Essential GraphRAG

Tài liệu tham khảo nền tảng "Neo4j Essential GraphRAG" [6] cung cấp khung thiết kế kiến trúc chuẩn mực để tích hợp GraphRAG vào các ứng dụng suy luận AI. Quá trình hoạt động của khung lý thuyết này bao gồm ba giai đoạn trọng yếu:

### 2.3.1. Khởi tạo và Xây dựng Đồ thị (Graph Initialization)
Giai đoạn này chuyển đổi dữ liệu phi cấu trúc (văn bản) thành một đồ thị tri thức có cấu trúc. Thông qua quá trình phân tích ngôn ngữ tự nhiên (NLP) hoặc thông qua sự hỗ trợ của LLM (LLM-based extraction), các thực thể cốt lõi và mối quan hệ giữa chúng được trích xuất. Giai đoạn này cũng bao gồm việc thiết lập không gian nhúng (embeddings) cho các đặc tính nội tại của nút và cạnh, cho phép kết hợp toán học đồ thị và toán học vector.

### 2.3.2. Độ phân giải Thực thể (Entity Resolution)
Một thách thức lớn trong xây dựng GraphRAG là sự dư thừa và không nhất quán khi trích xuất thực thể. Độ phân giải thực thể (Entity Resolution) là thuật toán nhằm nhận diện, hợp nhất và liên kết các biểu diễn khác nhau của cùng một thực thể (ví dụ: "AI", "Artificial Intelligence", "Trí tuệ nhân tạo") vào một node duy nhất [6]. Quá trình này giúp đồ thị gọn nhẹ, giảm nhiễu, đảm bảo độ chính xác của các đường đi (paths) trong biểu đồ và nâng cao độ tin cậy của thuật toán truy xuất.

### 2.3.3. Duyệt đồ thị (Graph Traversal)
Đây là cơ chế trích xuất thông tin chủ đạo trong GraphRAG. Tại thời điểm truy vấn (query time), từ các "nút mỏ neo" (anchor nodes) xác định bằng đối sánh ngữ nghĩa, hệ thống sẽ lan truyền thông tin theo chiều sâu hoặc chiều rộng qua các cạnh (Graph Traversal) để trích xuất biểu đồ phụ (sub-graph). Biểu đồ phụ này đóng vai trò là ngữ cảnh sâu sắc chứa chuỗi lập luận (reasoning chains) cung cấp cho LLMs để xử lý ngôn ngữ và đưa ra câu trả lời cuối cùng. 

## 2.4. Phương pháp Hybrid Retrieval (Cosine vs Graph Traversal)

Thực tiễn nghiên cứu khẳng định không một chiến lược đơn lẻ nào đủ sức giải quyết mọi dạng truy vấn. Do đó, Hybrid Retrieval (Truy xuất Lai) đang trở thành tiêu chuẩn vàng cho các hệ thống RAG 2025 [1], [6]. Phương pháp này hòa trộn hai mô hình truy xuất với nguyên lý hoạt động trái ngược nhau:

1. **Cosine Similarity (Dense Retrieval):** Đóng vai trò là công cụ "khám phá bề rộng". Dựa trên embedding vectors, mô hình cosine similarity cực kỳ hiệu quả trong việc nắm bắt ngôn ngữ đa dạng, từ đồng nghĩa và phát hiện các ý định tiềm ẩn (implicit intents) của người dùng để tìm ra điểm khởi đầu trong khối dữ liệu khổng lồ. Tuy nhiên, nó thiếu khả năng lập luận đa bước (multi-hop reasoning).

2. **Graph Traversal (Structured Retrieval):** Cung cấp "độ sâu phân tích". Sau khi vector search chỉ định các cụm (nodes) khởi đầu tương quan cao nhất, Graph Traversal mở rộng truy vấn theo các cạnh được định hướng tường minh. Quá trình này khai phá các mối quan hệ ẩn sâu, cung cấp bằng chứng suy luận (factual evidence) rõ ràng với tính minh bạch cao.

Mô hình Hybrid Retrieval kết hợp hai phương pháp trên nhằm khắc phục các "điểm mù" của nhau. Thuật toán kết hợp (fusion algorithm) như Reciprocal Rank Fusion (RRF) thường được sử dụng để hợp nhất kết quả từ luồng Dense Search và luồng Graph Search. Điều này giúp hệ thống vừa đạt độ phủ rộng (recall) của cosine similarity, vừa duy trì độ chuẩn xác (precision) cao nhờ lập luận liên kết của đồ thị. 

---

**Tài liệu tham khảo**

[1] H. Han et al., "Retrieval-Augmented Generation with Graphs (GraphRAG)," arXiv preprint arXiv:2501.00309, 2025.
[2] D. Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization," arXiv preprint arXiv:2404.16130, 2024.
[3] Z. Zhu et al., "Graph-based Approaches and Functionalities in Retrieval-Augmented Generation: A Comprehensive Survey," arXiv preprint arXiv:2504.10499, 2025.
[4] Z. Xiang et al., "When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation," arXiv preprint arXiv:2506.05690, 2025.
[5] Y. Li et al., "RGL: A Graph-Centric, Modular Framework for Efficient Retrieval-Augmented Generation on Graphs," arXiv preprint arXiv:2503.19314, 2025.
[6] Neo4j, "Neo4j Essential GraphRAG," Neo4j, 2025. [Online]. Available: https://go.neo4j.com/rs/710-RRC-335/images/Essential-GraphRAG.pdf.
