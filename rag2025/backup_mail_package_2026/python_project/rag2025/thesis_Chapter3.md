# Chương 3: Thực nghiệm Đánh giá và Tích hợp Thực tiễn

## 3.1. Tái cấu trúc Hệ thống RAG 2025

Quá trình chuyển đổi từ một hệ thống RAG cơ bản (Naive RAG) sang một kiến trúc tiên tiến hơn đòi hỏi việc đánh giá và tái cấu trúc toàn diện codebase hiện hữu. Hệ thống RAG 2025 ban đầu được xây dựng dựa trên FastAPI để cung cấp các API giao tiếp, sử dụng Qdrant Vector Store để lưu trữ và truy vấn vector, mô hình BGE-M3 cho việc biểu diễn nhúng (embedding), và kỹ thuật HyDE (Hypothetical Document Embeddings) để cải thiện độ chính xác truy vấn.

Tuy nhiên, việc đánh giá chi tiết codebase hiện tại chỉ ra một số hạn chế cốt lõi. Cụ thể, kiến trúc Naive RAG thường gặp khó khăn trong việc xử lý các truy vấn phức tạp đòi hỏi suy luận nhiều bước (multi-hop reasoning) do thiếu sự liên kết ngữ nghĩa rõ ràng giữa các thực thể (entities). Các văn bản được chunking một cách độc lập, làm mất đi ngữ cảnh tổng thể và các mối quan hệ tiềm ẩn. Do đó, việc tái cấu trúc hệ thống tập trung vào việc thiết kế các interface linh hoạt hơn, cho phép dễ dàng cắm (plug-in) các module đồ thị tri thức (Knowledge Graph) mà không làm gián đoạn luồng xử lý vector hiện tại. 

## 3.2. Đề xuất Kiến trúc Tích hợp Neo4j GraphRAG

Để khắc phục các nhược điểm của Naive RAG, chúng tôi đề xuất tích hợp cơ sở dữ liệu đồ thị Neo4j để xây dựng một kiến trúc GraphRAG hybrid. Sự kết hợp này tận dụng khả năng tìm kiếm tương đồng mạnh mẽ của Vector Search và sức mạnh biểu diễn tri thức cấu trúc của Graph Search.

### 3.2.1. Graph Indexing

Bước đầu tiên trong việc xây dựng GraphRAG là bổ sung khả năng trích xuất thực thể vào quá trình tiền xử lý văn bản. Module `src/chunker.py` được mở rộng để không chỉ phân chia văn bản thành các chunks mà còn nhận diện các thực thể có nghĩa (Named Entity Recognition) và các mối quan hệ (Relations) giữa chúng.

Quá trình này chuyển đổi văn bản phi cấu trúc thành các bộ ba (triplets) dạng `(Subject, Relation, Object)`. Sau khi được trích xuất, các nút (nodes) đại diện cho thực thể và các cạnh (edges) đại diện cho mối quan hệ sẽ được lưu trữ vào cơ sở dữ liệu Neo4j thông qua module `src/services/graph_store.py`. Việc chỉ mục hóa (indexing) này tạo ra một đồ thị tri thức phong phú, phản ánh sâu sắc cấu trúc ngữ nghĩa của tập dữ liệu.

### 3.2.2. Hybrid Query Router

Khớp nối cốt lõi của kiến trúc đề xuất nằm tại `src/main.py` với sự ra đời của Hybrid Query Router. Sau khi áp dụng HyDE để tạo ra một giả thuyết tài liệu từ truy vấn của người dùng, hệ thống sẽ thực thi đồng thời hai luồng truy vấn. Luồng thứ nhất sử dụng Qdrant để tìm kiếm các vector tương đồng. Luồng thứ hai tiến hành dịch truy vấn sang ngôn ngữ Cypher để lấy Graph Context từ Neo4j. Cuối cùng, hai nguồn ngữ cảnh (Vector Context và Graph Context) sẽ được hợp nhất (merge) theo một chiến lược đánh giá trọng số để cung cấp một bức tranh toàn cảnh cho mô hình ngôn ngữ lớn (LLM) sinh ra câu trả lời cuối cùng.

**Pseudo-code Kiến trúc `HybridRetriever`:**

```python
class HybridRetriever:
    def __init__(self, vector_store: QdrantStore, graph_store: Neo4jStore, llm_client: LLMClient):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.llm_client = llm_client

    async def retrieve(self, query: str) -> str:
        # Bước 1: Áp dụng HyDE để tạo giả thuyết tài liệu
        hypothesis = await self.generate_hyde_hypothesis(query)
        
        # Bước 2: Truy vấn Vector Store (Qdrant)
        vector_results = await self.vector_store.similarity_search(hypothesis, top_k=5)
        vector_context = self._format_vector_results(vector_results)
        
        # Bước 3: Trích xuất thực thể từ truy vấn và truy vấn Graph Store (Neo4j)
        entities = self.extract_entities_from_query(query)
        cypher_query = self._build_cypher_query(entities)
        graph_results = await self.graph_store.execute_query(cypher_query)
        graph_context = self._format_graph_results(graph_results)
        
        # Bước 4: Hợp nhất Vector Context và Graph Context
        hybrid_context = self.merge_contexts(vector_context, graph_context)
        
        return hybrid_context

    def merge_contexts(self, vector_context: str, graph_context: str) -> str:
        # Chiến lược kết hợp: Ưu tiên Graph Context cho các mối quan hệ, Vector Context cho thông tin chi tiết
        merged = f"=== Thông tin Từ Đồ Thị Tri Thức ===\n{graph_context}\n\n"
        merged += f"=== Thông tin Từ Vector Store ===\n{vector_context}"
        return merged
```

## 3.3. Thiết lập Thực nghiệm

Để đánh giá một cách toàn diện hiệu năng của hệ thống Hybrid GraphRAG, chúng tôi thiết lập một môi trường thực nghiệm với các cấu hình cụ thể. Schema của Neo4j được thiết kế để nắm bắt các khái niệm phức tạp trong lĩnh vực tuyển sinh (ví dụ: `(Student)-[APPLIED_TO]->(Major)`, `(Major)-[HAS_REQUIREMENT]->(Subject)`). 

Tập dữ liệu thử nghiệm (test dataset) được xây dựng cẩn thận, bao gồm các truy vấn đòi hỏi suy luận nhiều bước (multi-hop reasoning). Ví dụ: "Sinh viên có chứng chỉ IELTS 6.5 và điểm Toán 8.0 có thể đăng ký vào ngành nào yêu cầu ngoại ngữ và có mức học phí dưới 20 triệu?". Tập dữ liệu này đóng vai trò quan trọng trong việc làm bộc lộ những điểm yếu của kiến trúc Naive RAG truyền thống.

## 3.4. Phân tích và Đánh giá Kết quả

Quá trình đánh giá được thực hiện dựa trên sự so sánh trực tiếp giữa hệ thống Naive RAG nguyên bản và kiến trúc Hybrid GraphRAG đề xuất. Hai nhóm tiêu chí chính được phân tích là độ trễ (Latency) và độ chính xác/độ phủ (Precision/Recall).

Kết quả thực nghiệm cho thấy, trong các truy vấn đơn giản (single-hop), cả hai hệ thống đều đạt độ chính xác tương đương, tuy nhiên kiến trúc Hybrid có độ trễ cao hơn một chút do chi phí truy vấn đồ thị. Ngược lại, đối với các truy vấn phức tạp (multi-hop), hệ thống Hybrid GraphRAG thể hiện sự vượt trội đáng kể về cả Precision và Recall. Khả năng theo dấu các cạnh trên đồ thị giúp hệ thống tránh được hiện tượng ảo giác (hallucination) thường gặp ở các mô hình sinh văn bản khi thiếu ngữ cảnh có cấu trúc chặt chẽ. Việc hợp nhất hai luồng ngữ cảnh đã cung cấp cho LLM một nền tảng thông tin đầy đủ và chính xác hơn.

## 3.5. Kết luận và Định hướng

Thực nghiệm đã chứng minh tính hiệu quả của việc tích hợp Neo4j GraphRAG vào kiến trúc RAG 2025. Bằng cách bổ sung thêm một lớp tri thức có cấu trúc (Knowledge Graph), hệ thống không chỉ cải thiện độ chính xác trong các tác vụ suy luận phức tạp mà còn mở ra những khả năng mới cho việc biểu diễn và quản lý thông tin.

Về định hướng tương lai, việc nghiên cứu các hệ thống ontology động (Dynamic Ontologies), cho phép đồ thị tri thức tự động cập nhật và tiến hóa theo thời gian là một bài toán đầy triển vọng. Thêm vào đó, việc tích hợp sâu các mô hình Text-to-Cypher (Text2Cypher) sẽ giảm thiểu đáng kể rào cản kỹ thuật trong việc truy xuất dữ liệu đồ thị, tối ưu hóa quá trình tạo luồng truy vấn hybrid một cách tự nhiên và linh hoạt hơn.