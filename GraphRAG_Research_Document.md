# GraphRAG: A Formal Framework for Graph-Augmented Retrieval-Generation in Large Language Models

## Tóm tắt (Abstract)

Retrieval-Augmented Generation (RAG) đã trở thành một kỹ thuật nền tảng để cải thiện các Mô hình Ngôn ngữ Lớn (LLMs), giúp giảm thiểu ảo giác và dựa trên dữ liệu thực tế. Mặc dù vậy, RAG truyền thống (Naive RAG) chủ yếu dựa trên vector và gặp khó khăn với các tác vụ tóm tắt toàn cục đòi hỏi sự hiểu biết toàn diện về toàn bộ tập dữ liệu. Để giải quyết vấn đề này, Microsoft Research đã giới thiệu GraphRAG (2024), một hệ thống kết hợp lập chỉ mục dựa trên đồ thị, thuật toán phát hiện cộng đồng và cơ chế Map-Reduce. Tài liệu này phân tích chi tiết GraphRAG, bao gồm kiến trúc hệ thống, cơ sở toán học, ưu điểm vượt trội so với Naive RAG và các vấn đề nghiên cứu mở hiện tại. Bản tổng hợp này đóng vai trò làm nền tảng cho các bài báo khoa học trong lĩnh vực truy xuất tri thức có cấu trúc.

---

## 1. Executive Overview (Tổng quan thực thi)

Sự phát triển vượt bậc của Large Language Models (LLMs) đã giải quyết nhiều bài toán phức tạp trong xử lý ngôn ngữ tự nhiên. Dù vậy, các mô hình này vẫn bị cản trở bởi hai điểm yếu cốt lõi: hiện tượng ảo giác (hallucination) và sự thiếu hụt kiến thức cập nhật. Retrieval-Augmented Generation (RAG) được giới thiệu như một giải pháp tiêu chuẩn nhằm khắc phục những hạn chế này (Lewis et al., 2020). Bằng cách truy xuất thông tin từ các cơ sở dữ liệu bên ngoài, RAG cung cấp ngữ cảnh thực tế cho LLMs trước khi sinh văn bản. 

Tuy nhiên, kiến trúc RAG truyền thống (Naive RAG) đang bộc lộ những giới hạn nghiêm trọng khi đối mặt với các truy vấn đòi hỏi khả năng tổng hợp thông tin ở quy mô lớn (Sensemaking). Vì phụ thuộc hoàn toàn vào tìm kiếm vector mật độ cao (Dense Retrieval), Naive RAG chỉ có thể trích xuất các đoạn văn bản (text chunks) rời rạc. Phương pháp này hoạt động tốt với các câu hỏi tra cứu thông tin cụ thể nhưng hoàn toàn thất bại khi người dùng yêu cầu phân tích sự liên kết giữa các thực thể hoặc tóm tắt một chủ đề bao quát toàn bộ tập dữ liệu.

Tài liệu này trình bày GraphRAG (Graph-Augmented Retrieval-Generation), một khung lý thuyết và phương pháp luận hoàn chỉnh nhằm giải quyết triệt để khiếm khuyết của Naive RAG (Edge et al., 2024). Thay vì lưu trữ dữ liệu dưới dạng các vector phẳng, GraphRAG sử dụng chính LLMs để trích xuất các thực thể và mối quan hệ từ văn bản gốc, sau đó xây dựng một Đồ thị tri thức (Knowledge Graph). Các thuật toán phân cụm đồ thị (Graph Clustering) tiếp tục được áp dụng để chia nhỏ mạng lưới này thành các cộng đồng ngữ nghĩa có tính liên kết chặt chẽ. Hệ thống sau đó tạo ra các bản tóm tắt cho từng cộng đồng, cung cấp một hệ thống phân cấp thông tin từ vĩ mô đến vi mô. 

Nghiên cứu này sẽ phân tích sâu sắc cấu trúc toán học và quy trình vận hành của GraphRAG. Chúng tôi chứng minh rằng việc kết hợp cấu trúc tô-pô (topological structure) của đồ thị vào quá trình truy xuất không chỉ cải thiện độ chính xác mà còn mở khóa khả năng suy luận đa bước (multi-hop reasoning) cho LLMs. 

---

## 2. Background (Cơ sở lý thuyết)

**2.1. Cơ sở của Naive RAG và Dense Retrieval**

Kiến trúc RAG tiêu chuẩn hoạt động dựa trên cơ chế đối sánh vector tĩnh. Toàn bộ kho ngữ liệu (corpus) ban đầu được phân rã thành các đoạn văn bản ngắn (chunks). Các mô hình nhúng (Embedding Models) chuyển đổi từng đoạn văn bản này thành các vector n-chiều và lưu trữ trong Vector Database. 

Khi một truy vấn (query) được gửi đến hệ thống, nó cũng được chuyển đổi thành vector trong cùng một không gian hình học. Hệ thống sẽ tính toán khoảng cách vector (thường dùng Cosine Similarity) để tìm ra top-K đoạn văn bản có độ tương đồng cao nhất. LLM sau đó nhận truy vấn gốc cùng top-K đoạn văn bản này làm ngữ cảnh (context) để tổng hợp câu trả lời. Cơ chế này đặc biệt hiệu quả trong bài toán tìm kiếm ngữ nghĩa đơn lẻ (explicit semantic matching).

**2.2. Giới hạn của mô hình không gian vector phẳng**

Dù tối ưu cho việc truy xuất cục bộ, Dense Retrieval đối mặt với ba điểm mù kiến trúc không thể vượt qua bằng cách tinh chỉnh tham số thông thường:

Thứ nhất là sự phân mảnh ngữ cảnh toàn cục (Loss of Global Context). Quá trình chunking đã vô tình phá vỡ cấu trúc tường thuật liền mạch của tài liệu. Các thực thể liên quan chặt chẽ nhưng nằm ở hai đầu của một văn bản dài sẽ bị phân ly vào các vector khác nhau. Khi truy vấn đòi hỏi một góc nhìn toàn cảnh, hệ thống không thể cung cấp đủ bối cảnh vì nó chỉ được lập trình để tìm các mảnh ghép tương đồng nhất với câu hỏi, chứ không phải các mảnh ghép liên quan nhất với nhau.

Thứ hai là sự thất bại trong suy luận đa bước (Multi-hop Reasoning Failure). Giả sử người dùng đặt câu hỏi về mối liên hệ giữa thực thể A và thực thể C. Trong tập dữ liệu, A chỉ liên kết với B, và B liên kết với C. Truy vấn về A và C sẽ không có độ tương đồng Cosine đủ lớn với đoạn văn bản chứa B. Hệ quả là hệ thống bỏ lỡ "mắt xích" trung gian quan trọng nhất để xâu chuỗi câu trả lời.

Thứ ba là hội chứng "mù thông tin tổng hợp" (The Summarization Problem). Theo Edge et al. (2024), khi người dùng đặt những câu hỏi như "Đâu là những chủ đề chính trong toàn bộ tập tài liệu này?", Naive RAG thường hoạt động rất kém. Câu hỏi dạng này không chứa các từ khóa cụ thể để đối chiếu vector, dẫn đến việc hệ thống truy xuất các đoạn văn bản nhiễu hoặc không mang tính đại diện.

**2.3. Đồ thị tri thức (Knowledge Graphs) như một giải pháp thay thế**

Để vượt qua giới hạn của không gian vector phẳng, cấu trúc mạng lưới (Network Structure) được đề xuất làm nền tảng lưu trữ kiến thức. Một Đồ thị tri thức biểu diễn dữ liệu dưới dạng tập hợp các Nút (Nodes, đại diện cho thực thể) và Cạnh (Edges, đại diện cho mối quan hệ giữa chúng). 

Trong mô hình mạng lưới này, ngữ nghĩa không chỉ nằm ở nội dung của từng nút mà còn được xác định bởi định lý đồ thị (Graph Theory). Khoảng cách giữa hai thực thể không được tính bằng sự tương đồng về mặt chữ, mà được đo lường bằng số lượng cạnh kết nối chúng. Sự chuyển đổi từ "không gian vector phẳng" sang "không gian đồ thị liên kết" chính là tiền đề lý thuyết cốt lõi để xây dựng khung GraphRAG, cho phép hệ thống duy trì tính toàn vẹn của dữ liệu trong quá trình truy xuất phức hợp.

---

## 3. Formal Definition (Định nghĩa Hình thức)

Để xây dựng một khung lý thuyết vững chắc cho GraphRAG, hệ thống tri thức cần được mô hình hóa dưới dạng một Đồ thị thuộc tính (Property Graph) có hướng hoặc vô hướng. 

**3.1. Mô hình hóa Đồ thị Tri thức (Knowledge Graph Formulation)**

Giả sử chúng ta có một tập ngữ liệu văn bản đầu vào $\mathcal{C} = \{d_1, d_2, ..., d_n\}$, trong đó $d_i$ là các tài liệu độc lập. Nhiệm vụ của GraphRAG trong giai đoạn đầu là ánh xạ tập $\mathcal{C}$ thành một đồ thị tri thức $G = (V, E)$.
*   **Tập đỉnh (Vertices/Nodes):** $V = \{v_1, v_2, ..., v_N\}$ đại diện cho các thực thể (Entities) được trích xuất từ văn bản (ví dụ: con người, tổ chức, địa điểm, khái niệm). Mỗi đỉnh $v_i$ đi kèm với một tập thuộc tính $P(v_i)$ bao gồm nhãn (label) và mô tả ngữ nghĩa (description).
*   **Tập cạnh (Edges):** $E \subseteq V \times V$ đại diện cho các mối quan hệ (Relationships) giữa các thực thể. Một cạnh $e_{ij} \in E$ kết nối $v_i$ và $v_j$. Đi kèm với mỗi cạnh là trọng số $w_{ij}$ (thể hiện độ mạnh hoặc tần suất xuất hiện của mối quan hệ) và một đoạn mô tả (relationship claim).

**3.2. Phân rã Đồ thị và Phát hiện Cộng đồng (Graph Partitioning & Community Detection)**

Để giải quyết bài toán "mù thông tin tổng hợp" (Summarization Problem) của Naive RAG, GraphRAG áp dụng lý thuyết mạng (Network Theory) nhằm nhóm các thực thể có liên kết chặt chẽ thành các cộng đồng (Communities). 

Cấu trúc phân cấp của đồ thị được xác định thông qua bài toán tối ưu hóa tính mô-đun (Modularity Optimization). Khung GraphRAG tiêu chuẩn (Edge et al., 2024) sử dụng **thuật toán Leiden** (Traag et al., 2019) – một thuật toán có khả năng khắc phục giới hạn độ phân giải (resolution limit) và tránh các cộng đồng bị ngắt kết nối nội bộ của thuật toán Louvain trước đó.

Về mặt toán học, thuật toán Leiden phân hoạch tập đỉnh $V$ thành tập hợp các cộng đồng $\mathcal{P} = \{C_1, C_2, ..., C_k\}$ sao cho các thực thể trong cùng một cụm $C_i$ có mật độ liên kết nội bộ $E_{in}$ lớn hơn nhiều so với mật độ liên kết ngẫu nhiên (expected edges). Đồ thị $G$ sau đó có thể được tổ chức thành cấu trúc phân cấp đa tầng (hierarchical multi-level graph), cho phép LLMs tóm tắt ngữ cảnh từ mức độ vi mô (từng thực thể) đến vĩ mô (toàn bộ đồ thị).

---

## 4. System Architecture (Kiến trúc Hệ thống)

Kiến trúc của GraphRAG được thiết kế thành hai chu trình độc lập nhưng bổ trợ chặt chẽ cho nhau: Chu trình Lập chỉ mục (Indexing Pipeline) và Chu trình Truy vấn (Querying Pipeline).

**4.1. Chu trình Lập chỉ mục (Indexing Pipeline)**

Mục tiêu của chu trình này là chuyển đổi văn bản phi cấu trúc (unstructured text) thành một mạng lưới tri thức có khả năng truy xuất cao. Quá trình này bao gồm 4 bước chính:

1.  **Text Chunking & Prompting:** Tương tự Naive RAG, ngữ liệu $\mathcal{C}$ được chia thành các đoạn văn bản nhỏ (chunks). Tuy nhiên, thay vì chỉ nhúng (embed) các chunks này, GraphRAG đưa chúng vào LLM với các zero-shot prompt được thiết kế chuyên biệt để nhận diện thực thể và mối quan hệ.
2.  **Element Extraction (Trích xuất phần tử):** LLM đóng vai trò là một cỗ máy trích xuất thông tin (Information Extractor), trả về các cặp thực thể và mô tả mối quan hệ giữa chúng. Đầu ra này được chuẩn hóa để hình thành tập đỉnh $V$ và tập cạnh $E$ của đồ thị $G$.
3.  **Community Clustering (Phân cụm cộng đồng):** Thuật toán Leiden được áp dụng lên đồ thị $G$ để phân tách mạng lưới thành các cấu trúc cộng đồng phân cấp $\mathcal{P}$. Các cộng đồng này đại diện cho các chủ đề ngữ nghĩa (semantic themes) từ cụ thể đến bao quát.
4.  **Community Summarization (Tóm tắt cộng đồng):** Đây là khâu đột phá của GraphRAG. Đối với mỗi cộng đồng $C_i$, LLM được yêu cầu đọc danh sách các thực thể và mối quan hệ bên trong nó để tạo ra một bản tóm tắt tự nhiên (Natural Language Summary). Các bản tóm tắt này cung cấp cái nhìn toàn cảnh về tập dữ liệu mà không bị giới hạn bởi token window của mô hình.

**4.2. Chu trình Truy vấn (Querying Pipeline)**

Tùy thuộc vào bản chất của câu hỏi đầu vào $q$, GraphRAG vận hành hai chiến lược truy xuất khác biệt để tối ưu hóa việc sinh văn bản (Generation):

*   **Local Search (Truy vấn cục bộ - Tối ưu cho Entity-centric queries):**
    Khi câu hỏi $q$ đề cập đến các thực thể cụ thể (ví dụ: "Mối quan hệ giữa nhân vật A và tổ chức B là gì?"), hệ thống sẽ sử dụng Semantic Search để định vị các đỉnh $v_i$ tương ứng trên đồ thị $G$. Từ đỉnh neo (anchor node) này, hệ thống mở rộng tìm kiếm theo vùng lân cận k-bậc (k-hop neighborhood). Tập hợp các thực thể, mối quan hệ và văn bản gốc xung quanh đỉnh neo sẽ được trích xuất làm ngữ cảnh (context) đưa vào LLM để tổng hợp câu trả lời chính xác và giàu bối cảnh.

*   **Global Search (Truy vấn toàn cục - Tối ưu cho Sensemaking queries):**
    Đối với các câu hỏi đòi hỏi tính tổng hợp (ví dụ: "Những thách thức chính được đề cập trong toàn bộ báo cáo này là gì?"), GraphRAG áp dụng mô hình điện toán **Map-Reduce** (Dean & Ghemawat, 2008) kết hợp với các bản tóm tắt cộng đồng (Community Summaries).
    *   *Phase 1 (Map):* Hệ thống chia nhỏ truy vấn $q$ và gửi song song đến các bản tóm tắt cộng đồng của đồ thị. LLM đánh giá sự liên quan của từng tóm tắt đối với $q$ và tạo ra các câu trả lời cục bộ (intermediate partial responses) kèm theo điểm đánh giá mức độ quan trọng (helpfulness score).
    *   *Phase 2 (Reduce):* Hệ thống lọc các câu trả lời cục bộ có điểm số cao nhất, ghép nối chúng cho đến khi đạt giới hạn ngữ cảnh (context window limit) và yêu cầu LLM tổng hợp lại thành một câu trả lời cuối cùng duy nhất. 

Nhờ cấu trúc Map-Reduce chạy trên nền tảng mạng lưới đồ thị, GraphRAG hoàn toàn đánh bại hiện tượng "mù thông tin tổng hợp", mang lại khả năng trả lời các câu hỏi vĩ mô mà Naive RAG không thể thực hiện (Edge et al., 2024).

---

## 5. Theoretical Contributions (Đóng góp Lý thuyết)

GraphRAG mang lại những đóng góp lý thuyết cốt lõi cho bài toán Information Retrieval (IR) và Natural Language Processing (NLP), vượt qua rào cản của mô hình RAG truyền thống (Naive RAG) trong quá trình xử lý các truy vấn tổng hợp quy mô lớn (global sensemaking).

Thứ nhất, GraphRAG chính thức hóa quá trình trừu tượng hóa tri thức (knowledge abstraction) thông qua cấu trúc phân cấp. Thay vì biểu diễn ngữ liệu dưới dạng tập hợp các đoạn văn bản (chunks) rời rạc trong không gian vector, framework này lập mô hình dữ liệu thành một đồ thị tri thức (hierarchical knowledge graph). Việc áp dụng các thuật toán phân cụm đồ thị (như thuật toán Leiden) cho phép chia cắt không gian ngữ nghĩa thành các cộng đồng (communities) độc lập nhưng có tính liên kết nội bộ cao. Phương pháp này tạo ra cơ sở toán học vững chắc cho việc tóm tắt thông tin theo nhiều mức độ chi tiết (granularity), từ vi mô (node/edge) đến vĩ mô (toàn bộ đồ thị).

Thứ hai, framework đề xuất một cơ chế nén thông tin (information compression) cấu trúc. Naive RAG thường gặp giới hạn context window và hiện tượng nhiễu ngữ cảnh (context loss) khi số lượng tài liệu tăng lên. GraphRAG giải quyết vấn đề này bằng cách chuyển đổi thông tin phi cấu trúc thành dữ liệu có cấu trúc (entity và relationship) và tổng hợp thành các community summaries. Các bản tóm tắt cộng đồng đóng vai trò như những bộ lọc nhiễu, giữ lại ngữ nghĩa trọng tâm, giúp LLM có cái nhìn toàn cảnh mà không bị quá tải thông tin.

---

## 6. Empirical Evaluation (Đánh giá Thực nghiệm)

Việc đánh giá hệ thống GraphRAG dựa trên các thiết lập thử nghiệm đối đầu (head-to-head evaluation), tập trung đo lường khả năng trả lời các câu hỏi cấp độ toàn cục (global questions) trên ngữ liệu lớn.

**6.1. Baselines và Datasets**
Các nghiên cứu thực nghiệm đánh giá GraphRAG trên nhiều tập dữ liệu đa dạng như báo cáo tài chính, kịch bản podcast và bài báo khoa học. Hiệu suất của mô hình được đối chiếu với hai baseline chính:
- **Naive RAG**: Truy xuất các đoạn văn bản dựa trên độ tương đồng cosine (cosine similarity) giữa truy vấn và chunk.
- **Map-Reduce Summarization**: Chia nhỏ văn bản để tóm tắt từng phần (Map), sau đó gộp các bản tóm tắt lại (Reduce) nhằm tạo ra câu trả lời cuối cùng.

**6.2. Metrics (Chỉ số Đánh giá)**
Khung đánh giá sử dụng LLM as a judge (thường là GPT-4) để chấm điểm câu trả lời dựa trên bốn metric cốt lõi:
- **Comprehensiveness (Độ toàn diện)**: Mức độ bao phủ thông tin, đảm bảo mọi khía cạnh của truy vấn đều được đề cập.
- **Diversity (Độ đa dạng)**: Khả năng cung cấp nhiều góc nhìn, chi tiết và ví dụ phong phú từ ngữ liệu.
- **Empowerment (Độ hỗ trợ)**: Khả năng giúp người đọc hiểu sâu bản chất vấn đề và đưa ra phán đoán.
- **Directness (Tính trực diện)**: Mức độ rõ ràng, đi thẳng vào trọng tâm, loại bỏ thông tin dư thừa.

**6.3. Kết quả Thực nghiệm**
Dữ liệu thực nghiệm chứng minh GraphRAG vượt trội hoàn toàn so với Naive RAG, đặc biệt ở hai chỉ số **Comprehensiveness** và **Diversity**. Khi xử lý truy vấn như *"Chủ đề chính trong kho tài liệu này là gì?"*, Naive RAG thường thất bại do chỉ truy xuất được thông tin cục bộ. Map-Reduce tuy hoạt động tốt hơn Naive RAG nhưng tốn kém tài nguyên và dễ làm đứt gãy các liên kết ngữ nghĩa ngầm. GraphRAG, nhờ khai thác tập hợp community summaries, đạt điểm Comprehensiveness cao nhất đồng thời tối ưu hóa chi phí token bằng cách chỉ phân tích các bản tóm tắt thay vì rà soát toàn bộ văn bản gốc.

---

## 7. Critical Research Problems (Các Vấn đề Nghiên cứu Cốt lõi)

Mặc dù giải quyết thành công bài toán global sensemaking, framework GraphRAG vẫn tồn tại các điểm nghẽn (research gaps) đòi hỏi các nghiên cứu sâu hơn.

**Chi phí xây dựng đồ thị (Graph Construction Cost):**
Việc trích xuất Knowledge Graph từ văn bản thô phụ thuộc hoàn toàn vào LLM. Quá trình quét toàn bộ corpus để nhận diện entity và trích xuất relationship tiêu tốn lượng token khổng lồ. Chi phí tính toán ở giai đoạn indexing trở thành rào cản lớn khi áp dụng GraphRAG cho các tập dữ liệu quy mô công nghiệp (hàng triệu tài liệu).

**Đồ thị tri thức tĩnh (Static Knowledge Graph):**
Hiện tại, cơ chế của GraphRAG thiết kế cho các bộ dữ liệu tĩnh. Khi có tài liệu mới bổ sung hoặc thông tin cũ thay đổi, hệ thống buộc phải cập nhật đồ thị. Việc chạy lại thuật toán phân cụm cộng đồng và tạo lại summaries từ đầu gây lãng phí tài nguyên nghiêm trọng. Bài toán cập nhật cục bộ (local updates) duy trì cấu trúc phân cấp chưa có lời giải tối ưu.

**Nhiễu trích xuất và lan truyền lỗi (Extraction Noise & Error Propagation):**
Quá trình trích xuất bằng zero-shot hoặc few-shot prompt thường sinh ra các cạnh (edges) trùng lặp, sai lệch hoặc thiếu nhất quán. Sự xuất hiện của "ảo giác" (hallucinations) ngay từ bước khởi tạo đồ thị sẽ tạo ra hiệu ứng lan truyền lỗi lên các community summaries, làm suy giảm trực tiếp độ tin cậy của câu trả lời cuối cùng.

---

## 8. Future Directions (Hướng Mở rộng trong Tương lai)

Từ các giới hạn thực tiễn, các hệ thống đồ thị kết hợp LLM có thể mở rộng theo các hướng nghiên cứu tiềm năng sau:

**Dynamic GraphRAG (GraphRAG Động):**
Nghiên cứu áp dụng các thuật toán đồ thị luồng (streaming graph algorithms) để cập nhật Knowledge Graph theo thời gian thực. Khi có luồng dữ liệu mới, hệ thống chỉ tính toán lại các node, edge và cộng đồng bị tác động trực tiếp, thay vì cấu trúc lại toàn bộ đồ thị. Phương pháp này bảo đảm tính thời sự của RAG hệ thống với chi phí vận hành tối thiểu.

**Multi-modal GraphRAG (GraphRAG Đa phương thức):**
Mở rộng không gian đồ thị bằng cách nhúng các loại dữ liệu đa phương thức. Node trong đồ thị không chỉ là văn bản mà có thể là hình ảnh, video, biểu đồ hoặc âm thanh (ví dụ: một node chứa biểu đồ doanh thu liên kết với một node giải thích bằng văn bản). Kiến trúc này giúp GraphRAG xử lý các truy vấn phân tích xuyên định dạng.

**Hybrid Retrieval Optimization (Tối ưu hóa Truy xuất Lai):**
Xây dựng cơ chế truy xuất tự thích ứng (adaptive retrieval) kết hợp giữa Vector Search và Graph Traversal. Hệ thống sử dụng một mô hình định tuyến (router) để phân loại truy vấn: nếu là truy vấn truy xuất thông tin cụ thể (local query), hệ thống kích hoạt vector search nhằm đảm bảo tốc độ; nếu là truy vấn tổng hợp (global query), hệ thống chuyển hướng sang phân tích tóm tắt cộng đồng.

**Lightweight Graph Extraction (Trích xuất Đồ thị Nhẹ):**
Phát triển và tinh chỉnh (fine-tune) các mô hình ngôn ngữ nhỏ (Small Language Models - SLMs) chuyên trách cho tác vụ Named Entity Recognition (NER) và Relation Extraction (RE). Kết hợp các công cụ NLP truyền thống với SLMs sẽ thay thế dần sự phụ thuộc vào các mô hình tham số lớn (như GPT-4), tối ưu hóa bài toán đánh đổi (trade-off) giữa chi phí khởi tạo và chất lượng đồ thị tri thức.

---

## 9. References (Tài liệu tham khảo)

1. Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., & Larson, J. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*. Microsoft Research. arXiv:2404.16130.
2. Han, H., Ma, L., Shomer, H., Wang, Y., Lei, Y., Guo, K., ... & Tang, J. (2025). *RAG vs. GraphRAG: A Systematic Evaluation and Key Insights*. arXiv:2502.11371.
3. Zhu, Z., Huang, T., Wang, K., Ye, J., Chen, X., & Luo, S. (2025). *Graph-based Approaches and Functionalities in Retrieval-Augmented Generation: A Comprehensive Survey*. arXiv:2504.10499.
4. Traag, V. A., Waltman, L., & van Eck, N. J. (2019). *From Louvain to Leiden: guaranteeing well-connected communities*. Scientific Reports, 9(1), 5233.
5. Guo, Z., Yan, L., et al. (2024). *LightRAG: Simple and Fast Retrieval-Augmented Generation.* arXiv:2410.05779.
6. Gutiérrez, B. J., Zhu, Y., et al. (2024). *HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models.* NeurIPS 2024. arXiv:2405.14831.
7. Li, D., Shen, J., et al. (2024). *SubgraphRAG: Retrieval-Augmented Generation for Open-Domain Question Answering via Subgraph Reasoning.* ICLR 2025. arXiv:2410.20724.
8. Zhao, Q., Li, C., et al. (2025). *E2GraphRAG: Eliminating Embedding-Based Graph Retrieval-Augmented Generation.* arXiv:2505.24226.
9. Wu, J., Zhu, Y., et al. (2024). *MedGraphRAG: Graph RAG for Medical Domains.* arXiv:2408.04187.
10. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.