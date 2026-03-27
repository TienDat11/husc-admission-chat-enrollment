# Phần 3: Phân tích Kỹ thuật Thuật toán cốt lõi của GraphRAG

Phần này đi sâu vào ba trụ cột thuật toán nền tảng tạo nên sức mạnh của hệ thống GraphRAG theo kiến trúc Neo4j, tập trung vào phân rã đồ thị, sinh truy vấn Cypher và đối sánh ngữ nghĩa.

## 3.1. Thuật toán phân rã đồ thị (Leiden/Louvain) cho Entity Resolution và Gom cụm đỉnh V

Trong kỹ thuật đồ thị tri thức (Knowledge Graph), sau khi các thực thể (entities) và mối quan hệ (relationships) được trích xuất từ văn bản thô, đồ thị thu được thường rất lớn, nhiễu và rời rạc. Để LLM có thể xử lý hiệu quả, đồ thị cần được phân rã thành các cấu trúc cộng đồng (Community Structure) phân cấp.

### 3.1.1. Entity Resolution (Đồng nhất thực thể)

**Entity Resolution** là quá trình xác định và gộp các biến thể khác nhau của cùng một thực thể thế giới thực (ví dụ: "IBM", "International Business Machines", "IBM Corp") thành một node (đỉnh) duy nhất trên đồ thị. 

Thuật toán gom cụm (Clustering) đóng vai trò cốt lõi trong bước này:
1. **Tính toán nhúng (Embedding)**: Mỗi biến thể thực thể được biểu diễn bằng vector nhúng ngữ nghĩa $v_i \in \mathbb{R}^d$.
2. **Khởi tạo đồ thị tương tự (Similarity Graph)**: Xây dựng đồ thị K-NN (K-Nearest Neighbors) dựa trên độ tương tự Cosine giữa các vector thực thể: $sim(v_i, v_j) > \tau$ (với $\tau$ là ngưỡng tương tự).
3. **Phân rã (Resolution)**: Sử dụng các thuật toán phát hiện cộng đồng (Community Detection) để nhóm các thực thể tương tự nhau, sau đó LLM đánh giá lại các cụm này để quyết định việc hợp nhất (merge) các node, giảm triệt để sự rời rạc của đồ thị tri thức.

### 3.1.2. Gom cụm đỉnh V với Thuật toán Louvain và Leiden

Để tạo ra các **Community Summaries** (Tóm tắt cộng đồng) cho phép truy vấn toàn cục (Global Search) trong cấu trúc Map-Reduce RAG, đồ thị $G(V, E)$ cần được phân chia thành các cộng đồng $C = \{C_1, C_2, \dots, C_k\}$.

**Thuật toán Louvain**:
Dựa trên nguyên lý tối đa hóa hàm **Modularity (Tính mô-đun)** $Q$ để tìm ra các nhóm đỉnh liên kết dày đặc với nhau:
$Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$
Trong đó: $A_{ij}$ là trọng số cạnh giữa đỉnh $i, j$; $k_i$ là bậc của đỉnh $i$; $m$ là tổng trọng số cạnh; $\delta(c_i, c_j)$ = 1 nếu đỉnh $i, j$ cùng thuộc một cộng đồng, ngược lại = 0.

*Hạn chế của Louvain*: Có xu hướng tạo ra các cộng đồng bị ngắt kết nối nội bộ (internally disconnected communities) do thuật toán gộp node quá tham lam, làm sai lệch cấu trúc tóm tắt của GraphRAG.

**Thuật toán Leiden (Bản nâng cấp tối ưu)**:
Microsoft GraphRAG mặc định sử dụng **Leiden**, bản nâng cấp của Louvain, để giải quyết triệt để lỗi cộng đồng mất kết nối:
1. **Local Moving (Di chuyển cục bộ)**: Giống Louvain, các đỉnh (nodes) di chuyển sang cộng đồng làm tăng tối đa Modularity.
2. **Refinement (Tinh chỉnh)**: Đây là bước đột phá của Leiden. Thuật toán chia các cộng đồng vừa tạo thành các cộng đồng con (sub-communities) nhỏ hơn nhưng liên kết chặt chẽ hơn.
3. **Aggregation (Gộp)**: Co mỗi cộng đồng tinh chỉnh thành một siêu đỉnh (super-node) và lặp lại quá trình.

**Kết quả phân rã**: Đồ thị được chia thành một cây phân cấp (Hierarchical Tree) các cộng đồng:
- **Level 0**: Các cộng đồng nhỏ, chi tiết cực vi mô (Micro-level).
- **Level 1**: Gộp các cộng đồng Level 0 (Meso-level).
- **Level K**: Một vài cộng đồng lớn bao trùm toàn đồ thị (Macro-level).
Mỗi cụm đỉnh $V$ ở mỗi level sẽ được LLM tóm tắt lại, tạo thành cấu trúc index hoàn hảo cho các truy vấn mang tính khái quát cao.

## 3.2. Sinh truy vấn Cypher (Text2Cypher): Ánh xạ không gian ẩn LLM sang AST đồ thị có hướng

Một tính năng cốt lõi của Neo4j GraphRAG là khả năng dịch trực tiếp câu hỏi ngôn ngữ tự nhiên thành truy vấn cơ sở dữ liệu đồ thị, được gọi là **Text2Cypher**. Quá trình này không phải là dịch chuỗi thuần túy (string-to-string translation) mà là một phép ánh xạ phức tạp từ biểu diễn ngôn ngữ sang cấu trúc cây.

### 3.2.1. Quá trình ánh xạ sang AST (Abstract Syntax Tree)

Việc tạo truy vấn Cypher bản chất là ánh xạ ngữ nghĩa từ **không gian ẩn (latent space)** của LLM sang cấu trúc **Cây cú pháp trừu tượng (AST)** có hướng của ngôn ngữ Cypher.

Quá trình này bao gồm:
1. **Schema Injection**: Lược đồ (Schema) của đồ thị (bao gồm Node labels, Relationship types, và Properties) được nhúng chặt chẽ vào Context Window của LLM.
2. **Entity Extraction & Linking**: LLM nhận diện các thực thể trong câu hỏi và liên kết (grounding) chúng với các Properties chính xác trong Schema.
3. **AST Construction**: Bức xạ câu hỏi tự nhiên thành các nhánh AST đại diện cho các mệnh đề `MATCH`, `WHERE`, `RETURN` của Cypher.
   
   *Ví dụ*: Phân tích câu hỏi *"Ai đã đạo diễn bộ phim The Matrix?"*
   - Khai triển nhánh MATCH (Pattern): `(p:Person) -[:DIRECTED]-> (m:Movie)`
   - Khai triển nhánh WHERE (Lọc): `m.title = 'The Matrix'`
   - Khai triển nhánh RETURN (Kết quả): `RETURN p.name`

### 3.2.2. Kỹ thuật tối ưu hóa độ chính xác Text2Cypher

Các mô hình LLM cơ sở (base models) thường mắc lỗi cú pháp hoặc ảo giác lược đồ (Schema hallucination) khi sinh truy vấn cho các đồ thị phức tạp. Kiến trúc GraphRAG giải quyết bằng ba kỹ thuật:

1. **Few-Shot Prompting**: Cung cấp các cặp mẫu (Natural Language $\rightarrow$ Cypher AST) trong prompt (ví dụ: `Prompt: {e[0]}\nCypher: {e[1]}`) để LLM học theo cấu trúc ánh xạ cục bộ.
2. **Fine-Tuning mô hình chuyên biệt**: Sử dụng các mô hình nhỏ gọn được tinh chỉnh (finetuned) đặc thù cho tác vụ chuyển đổi Text-to-Cypher (như bộ dữ liệu mở `neo4j/text2cypher` trên Hugging Face), giúp mô hình "hiểu sâu" ngữ pháp Cypher mà không tốn quá nhiều tài nguyên tính toán.
3. **Cypher Templates làm Hard-fallback**: Sử dụng Cypher templates cho các truy vấn có tính chu kỳ hoặc tĩnh, làm nền tảng kiểm chứng chéo (Ground truth validation) giảm rủi ro lỗi ngữ pháp.

## 3.3. Đối sánh ngữ nghĩa (Sub-graph Search): Sự giao thoa thuật toán duyệt đồ thị và Vector

Local Search (Truy vấn cục bộ tập trung vào thực thể) trong kiến trúc GraphRAG đòi hỏi sự kết hợp tinh vi giữa tìm kiếm tương tự không gian vector (Semantic Search) và thuật toán duyệt đồ thị cấu trúc (Graph Traversal). Đây là điểm giao thoa mạnh mẽ nhất giữa mô hình ngôn ngữ và toán học đồ thị.

### 3.3.1. Vector Search làm điểm neo (Anchor Search)

Khi hệ thống nhận được câu hỏi từ người dùng:
1. Nhúng câu hỏi thành vector đại diện $v_q$.
2. Thực hiện tìm kiếm K-NN với các vector đỉnh $v_i$ trong không gian nhúng (thông qua vector index của Neo4j):
   $Nodes_{anchor} = \arg\max_{v_i \in V} \frac{v_q \cdot v_i}{||v_q|| ||v_i||}$
Các đỉnh thu được (Anchor nodes) đóng vai trò là "hạt giống" (seed) để thuật toán duyệt đồ thị bắt đầu, giải quyết triệt để vấn đề "Bắt đầu tìm kiếm thông tin từ đâu trong một mạng lưới khổng lồ".

### 3.3.2. Thuật toán duyệt K-hop (K-hop Traversal)

Từ các Anchor nodes, không gian tìm kiếm được "nở" ra theo các cạnh (relationships) để thu thập bối cảnh ngữ nghĩa xung quanh:
- **1-hop**: Duyệt các đỉnh kề trực tiếp (hàng xóm bậc 1).
- **K-hop**: Duyệt lan truyền tới các đỉnh kề ở độ sâu K (thường K=2 hoặc K=3 là tối ưu cho RAG).

Thuật toán K-hop giúp GraphRAG thu thập được **ngữ cảnh ngầm ẩn** (Implicit context) mà vector search đơn thuần sẽ bỏ sót. 
*Ví dụ*: Khi hỏi về "Chiến lược AI của công ty X", Vector search có thể chỉ tìm được đỉnh "Công ty X" (Anchor). Duyệt K-hop sẽ kéo theo các đỉnh kề như "Nhà khoa học dữ liệu Y" (nhân viên) hoặc "Dự án Z" (sản phẩm), làm giàu bối cảnh (context enrichment) để LLM tổng hợp câu trả lời chính xác và đa chiều.

### 3.3.3. Thuật toán cá nhân hóa PageRank (Personalized PageRank - PPR)

Khi số lượng đỉnh xung quanh Anchor nodes quá lớn qua các bước nhảy K-hop (đặc biệt với các "siêu đỉnh" kết nối với hàng triệu đỉnh khác), hệ thống phải đối mặt với bài toán **vượt quá Context Window** của LLM.

Thuật toán **Personalized PageRank (PPR)** được sử dụng như một bộ lọc (filter) để xếp hạng (rank) độ quan trọng của các sub-graph (đồ thị con) được trích xuất:

1. Khởi tạo một mô hình Random Surfer (Người lướt đồ thị ngẫu nhiên).
2. Khác với thuật toán PageRank truyền thống phân phối xác suất đều trên toàn bộ đồ thị, PPR **thiên lệch (bias)** xác suất nhảy (teleport probability) ép luồng duyệt luôn quay trở về tập $Nodes_{anchor}$.
3. **Phương trình hội tụ PPR**:
   $PR = (1 - \alpha) M \cdot PR + \alpha P_{anchor}$
   *Trong đó*: $\alpha$ là hệ số nhảy (damping factor $\approx$ 0.15 - 0.85 tùy cấu hình); $M$ là ma trận chuyển trạng thái đồ thị; $P_{anchor}$ là vector phân phối tập trung mạnh vào các đỉnh Anchor.

**Sự giao thoa hoàn hảo**:
Thuật toán đối sánh ngữ nghĩa là sự kết hợp của một chuỗi liên hoàn: Vector Search (mở không gian) $\rightarrow$ K-hop Traversal (khai phá và thu thập ngữ cảnh) $\rightarrow$ Personalized PageRank (chắt lọc cấu trúc quan trọng nhất). Nhờ đó, Sub-graph Search của GraphRAG vượt trội hơn hẳn Naive RAG trong việc xử lý các truy vấn logic nhiều bước (Multi-hop reasoning), duy trì dòng suy luận không bị đứt gãy.