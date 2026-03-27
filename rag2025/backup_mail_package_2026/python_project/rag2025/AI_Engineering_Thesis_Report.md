# Báo Cáo Nghiên Cứu Kỹ Thuật AI: Bước Dịch Chuyển Từ Naive RAG Sang GraphRAG Dựa Trên Nguyên Lý Đồ Thị Của Neo4j

**Tóm tắt (Abstract)**
Báo cáo kỹ thuật AI cấp độ tiến sĩ này trình bày nền tảng toán học toàn diện về sự chuyển dịch từ hệ thống truy xuất và sinh văn bản cơ bản (Naive RAG) sang mô hình tăng cường qua đồ thị tri thức (GraphRAG), lấy nền tảng từ kiến trúc GraphRAG cốt lõi của Neo4j. Báo cáo tập trung hoàn toàn vào nền tảng toán học của các đường ống (pipelines) học máy: từ giới hạn của không gian vector đẳng hướng (isotropic limits) ở các mô hình Naive RAG, phương pháp tối ưu hóa cấu trúc liên kết đồ thị (graph topology), thuật toán truy xuất lai (hybrid retrieval) dung hợp RRF, cho đến các phương pháp đo lường đánh giá mô hình bằng xác suất hậu nghiệm. Các kỹ sư AI và nhà nghiên cứu có thể dùng tài liệu này làm khuôn khổ lý thuyết để triển khai các hệ thống GraphRAG cấp doanh nghiệp phục vụ suy luận logic đa bước.

---

## 1. Nền Tảng Toán Học Của Naive RAG và Giới Hạn Đẳng Hướng

Trong kiến trúc Naive RAG, tri thức được biểu diễn dưới dạng các đoạn văn bản (chunks) và được mã hóa thành các embeddings trong không gian vector liên tục. Gọi $\mathcal{T}$ là tập hợp các đoạn văn bản (chunks), ta có ánh xạ (mapping) từ không gian ngữ nghĩa sang không gian $\mathbb{R}^d$ thông qua bộ mã hóa $\phi$ (như BERT, text-embedding-ada-002):

$$ \phi: \mathcal{T} \to \mathbb{R}^d $$

Khi người dùng đưa ra câu truy vấn $q$, mô hình tiến hành ánh xạ $v_q = \phi(q)$ và thực hiện tìm kiếm k lân cận gần nhất (k-NN) đối với tập tài liệu $v_d = \phi(d)$. Độ tương đồng phổ biến nhất được sử dụng là **Cosine Similarity** (Độ tương đồng Cosine), được định nghĩa bằng:

$$ S_C(v_q, v_d) = \frac{v_q \cdot v_d}{\|v_q\|_2 \|v_d\|_2} = \cos(\theta) $$

Vấn đề toán học cốt lõi của Naive RAG lộ rõ khi đối mặt với **suy luận đa bước (multi-hop reasoning)** do hiệu ứng giới hạn đẳng hướng (isotropic limits) của tích vô hướng (dot-products). Trong các mô hình ngôn ngữ lớn (LLMs), không gian nhúng (embedding space) thường có dạng hình nón hẹp (narrow cone-effect) gây ra bởi tính dị hướng (anisotropy) trong các tầng biểu diễn Transformer. Các tần số từ vựng thống trị sẽ chi phối vector, làm mất đi các liên kết ẩn dạng chuỗi. 

Xét một truy vấn đòi hỏi liên kết logic qua chuỗi thực thể $A \to B \to C \to \dots \to N$. Khoảng cách cosine suy biến dần do hiện tượng xô lệch (drift) trong không gian vector đa chiều. Theo định lý tập trung độ đo (Concentration of Measure) trong không gian nhiều chiều, trung bình hóa các vector hoặc tính toán k-hop trên không gian rời rạc dẫn tới một kỳ vọng toán học tiệm cận sự ngẫu nhiên:

$$ \lim_{n \to \infty} \mathbb{E}[S_C(v_A, v_n)] \approx \mathcal{O}\left(\frac{1}{\sqrt{d}}\right) $$

Nói cách khác, khi $n$ tăng, khoảng cách cosine giữa hai khái niệm được liên kết gián tiếp có xu hướng trực giao (orthogonal), khiến Naive RAG hoàn toàn mất đi khả năng duy trì định hướng cấu trúc nhân quả (causal graph alignment).

---

## 2. Hình Học Cấu Trúc Đồ Thị và Biểu Diễn Tri Thức Toán Học

Để vượt qua hạn chế "điểm mù đa bước" của không gian vector, GraphRAG thay thế cơ sở dữ liệu phi cấu trúc bằng một cấu trúc topo đồ thị tri thức chặt chẽ, được định nghĩa toán học là:

$$ \mathcal{G} = (V, E, \mathcal{R}) $$

Trong đó $V$ là tập hợp các đỉnh (thực thể - Nodes/Entities), $\mathcal{R}$ là tập hợp các loại quan hệ (Relation Types) và $E \subseteq V \times \mathcal{R} \times V$ là tập hợp các cạnh có hướng. Sự tương tác phức tạp này được mã hóa hoàn toàn thành **Ma trận Kề Thưa (Sparse Adjacency Matrices)** đối với mỗi quan hệ $r \in \mathcal{R}$:

$$ \mathbf{A}^{(r)} \in \{0, 1\}^{|V| \times |V|} $$
với các phần tử $\mathbf{A}_{i,j}^{(r)} = 1$ nếu có cạnh nối hướng từ đỉnh $v_i$ sang đỉnh $v_j$ thông qua quan hệ $r$.

Trong kiến trúc này, Large Language Model (LLM) không đóng vai trò truy xuất thuần túy mà được xem như một **Bộ Phân Tích Ngữ Nghĩa (Semantic Parser)** để khởi tạo đồ thị nội tại. Tận dụng cơ chế Self-Attention của LLM, mô hình phân rã văn bản thành các bộ ba $(h, r, t)$ (head, relation, tail). Biểu thức Attention nội sinh giải mã mối quan hệ từ ngữ cảnh:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V $$

Cơ chế self-attention định tuyến các trọng số từ vựng $h$ đến từ vựng $t$, thông qua sự kích hoạt ma trận $r$. Bằng cách đặt ngưỡng (thresholding) đối với ma trận chú ý trung gian $W_{att}$, quy trình khai thác thông tin (Information Extraction) chuyển đổi xác suất phân phối ngôn ngữ thành các siêu cạnh (hyper-edges) chắc chắn (deterministic edges) trong $A^{(r)}$.

---

## 3. Thuật Toán Mô Hình GraphRAG Theo Hệ Tiêu Chuẩn Neo4j

Paradigm GraphRAG của Neo4j xử lý dòng dữ liệu thông qua ba thuật toán cốt lõi trong đường ống (pipeline):

### 3.1 Khử Trùng Lặp Thực Thể (Entity Resolution) thông qua Leiden/Louvain
Để tổng hợp và cô đọng không gian trạng thái $\mathcal{G}$, các thuật toán phân cụm (Clustering/Community Detection) được áp dụng. Tối ưu hóa mô-đun hóa (Modularity Optimization) thông qua thuật toán Louvain hoặc Leiden liên tục phân chia đồ thị thành các cộng đồng thực thể ngữ nghĩa đồng nhất:

$$ \Delta Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \gamma \frac{k_i k_j}{2m} \right] \delta(c_i, c_j) $$

Trong đó $m = \frac{1}{2} \sum_{ij} A_{ij}$ là tổng số cạnh, $k_i$ là bậc (degree) của đỉnh $i$, $\gamma$ là hệ số độ phân giải (resolution parameter), và $\delta$ là hàm Kronecker phân định sự cùng cụm. Bằng cách gộp (collapse) các cộng đồng $c_i$ thành các Meta-Nodes, hệ thống nén topo cấu trúc, tối ưu cho ngữ cảnh có độ dài giới hạn (Context Window Limit) của LLM.

### 3.2 Ánh Xạ Không Gian Tiềm Ẩn LLM (LLM Latent Space) Thành Cypher AST
Việc chuyển đổi trực tiếp Natural Language $q \to$ Đồ thị con được thực hiện thông qua ánh xạ nội hàm (latent mapping) từ không gian tiềm ẩn của LLM sang Cây Cú Pháp Trừu Tượng (AST - Abstract Syntax Tree) của ngôn ngữ truy vấn Cypher. Bài toán này quy về tối ưu hàm ước lượng hợp lý cực đại (Maximum Likelihood Estimation - MLE) của chuỗi từ vựng Cypher $Y = (y_1, ..., y_m)$:

$$ \max_{\theta} \sum_{t=1}^m \log P(y_t | y_{<t}, q; \theta) $$
Việc giải mã tự hồi quy (autoregressive decoding) này bắt buộc phải chịu sự kiềm chế ngữ nghĩa để đảm bảo tính hợp lệ topo của cấu trúc AST đầu ra đối với lược đồ (schema) hiện tại của Neo4j.

### 3.3 Tìm Kiếm Đồ Thị Con: PageRank / K-Hop Đối Ngẫu Với Cosine
So với tìm kiếm KNN bằng khoảng cách Cosine $\arg \max_{v \in V} S_C(v_q, v)$, GraphRAG tìm kiếm đồ thị con kết hợp với phương pháp truyền lan độ tin cậy. Nếu quá trình ánh xạ nhận diện được tập đỉnh hạt giống (seed nodes) $S_0$, phân phối tầm quan trọng (importance distribution) của các node xung quanh được hội tụ bằng Random Walk hoặc Thuật toán **PageRank cá nhân hóa** (Personalized PageRank) với hệ số suy giảm (damping factor) $\alpha$:

$$ \mathbf{p} = \alpha \mathbf{A}^\top \mathbf{D}^{-1} \mathbf{p} + (1 - \alpha) \mathbf{v}_q $$

Quá trình truyền qua $K$-hop (k-hop traversal) thu thập toàn bộ các lân cận của hạt giống. Việc mở rộng đồ thị dựa trên cạnh liền kề (adjacent expansion) thiết lập nên viền (boundary) chứa đựng các mối liên kết đa chiều mà tích vô hướng tuyến tính (dot-product) trong $\mathbb{R}^d$ thất bại không thể vươn tới.

---

## 4. Công Thức Truy Xuất Lai Cấu Trúc (Hybrid Retrieval Formulation)

Sức mạnh thực sự của hệ thống Neo4j GraphRAG nằm ở khả năng tổng hợp vector dày đặc (Dense), thưa thớt (Sparse) và tri thức topo (Graph).

### 4.1 Giới hạn Dung Hợp Hạng Nghịch Đảo (Reciprocal Rank Fusion - RRF)
Với các không gian truy xuất không đồng nhất $m \in \{ \text{BM25 (Sparse)}, \text{Dense Vector}, \text{Graph K-hop} \}$, thứ hạng của tài liệu/node $d$ tại mỗi mô hình truy xuất là $r_m(d)$. Điểm RRF được hội tụ toán học như sau để loại bỏ nhiễu phân phối điểm tuyệt đối:

$$ \text{RRF}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)} $$

Với hằng số điều chỉnh hình phạt cực đại $k$ (thường chọn $k=60$). Thuật toán RRF bảo đảm tính hội tụ bất biến dưới biến đổi phi tuyến của các bộ truy xuất độc lập.

### 4.2 Tích Hợp Điểm Đa Chiều và Tối Ưu Hóa Siêu Phẳng (Hyperplane)
Giả sử ta muốn đánh giá xác suất hoặc điểm số thực tế độ phù hợp của tài liệu/node $\hat{y}$, Điểm số tích hợp (Multi-dimensional Score Integration) kết hợp điểm theo phương trình tổ hợp tuyến tính lồi:

$$ S_{final}(d, q) = w_{BM25} S_{BM25} + w_{Dense} S_{Dense} + w_{Graph} S_{Graph} $$

Hàm mục tiêu (Objective function) được xây dựng trên hàm mất mát Entropy chéo nhị phân (Binary Cross-Entropy - BCE) để cân bằng siêu phẳng:

$$ \mathcal{L}(\mathbf{w}) = - \sum_{i} \left[ y_i \log \sigma(S_{final}^{(i)}) + (1 - y_i) \log(1 - \sigma(S_{final}^{(i)})) \right] $$

Các vector trọng số siêu phẳng (Hyperplane weights) $\mathbf{w} = [w_{BM25}, w_{Dense}, w_{Graph}]^\top$ được tối ưu thông qua Gradient Descent:

$$ \mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w}^{(t)}) $$

Phương trình này cá nhân hóa và tinh chỉnh (finetune) quá trình truy xuất dự trên mật độ thực thể, độ dài truy vấn và tính chất dị hướng của bộ dữ liệu tổ chức.

---

## 5. Đo Lường Đánh Giá Hiệu Năng Trí Tuệ Nhân Tạo (AI Evaluation Metrics)

Việc đánh giá GraphRAG khác biệt hoàn toàn với Naive RAG do tính phi tuyến của cấu trúc đồ thị. Các metric cần xét đến sự tương quan hình học (graph-constrained).

### 5.1 Xếp Hạng Ràng Buộc Đồ Thị: Graph-constrained MRR và NDCG
Chỉ số MRR (Mean Reciprocal Rank) và NDCG (Normalized Discounted Cumulative Gain) truyền thống được chuẩn hóa bổ sung trọng số cấu trúc, tính đến khoảng cách đường đi ngắn nhất (shortest-path distance) $d(v_k, v_{truth})$ trong đồ thị để phạt các node xa cách:

$$ \text{GC-NDCG@K} = \frac{1}{\text{IDCG}} \sum_{k=1}^K \frac{2^{rel_k \cdot \exp(-\lambda \cdot d(v_k, v_{truth}))} - 1}{\log_2(k + 1)} $$

Sự suy giảm (decay) số mũ cực trị $\exp(-\lambda \cdot d(\dots))$ trừng phạt thẳng tay các vector có độ tương đồng bề mặt (surface semantic similarity) cao nhưng nằm cô lập hoặc đứt gãy trong cấu trúc topo.

### 5.2 Định Lượng Mức Độ Trung Thành (Faithfulness Quantification)
Tính xác thực nguyên nhân - kết quả được định lượng qua **Xác suất Hình học Topo Hậu nghiệm (Posterior Topological Probability)**. Gọi $\mathcal{G}_c$ là đồ thị tri thức nội tại của context sinh ra bởi GraphRAG, mức độ trung thành của đầu ra LLM Output $O$ được tính toán:

$$ \text{Faithfulness}(O) = \mathbb{P}(O | \mathcal{C}, \mathcal{G}_c) = \prod_{e \in \mathcal{E}_{O}} \mathbb{P}(e \in \mathcal{G}_c) $$

Một output càng bám sát vào các chuỗi liên kết nhân quả có thể được truy xuất chứng minh trực tiếp từ đồ thị tri thức thì phân phối kỳ vọng của xác suất hậu nghiệm này tiến càng gần $1$.

### 5.3 Định Lượng Ảo Giác (Hallucination Quantification)
Tình trạng ảo giác (Hallucination) xảy ra khi LLM tạo sinh nội suy ngụy biện các liên kết $e_{fake} \notin E$. Sự chệch hướng cấu trúc này được định lượng thông qua **Khoảng cách Levenshtein (Levenshtein Distance)** trên không gian chuỗi thực thể đường đi hoặc **Khoảng cách phân kỳ hình học (Geometric Divergence)** thông qua Kullback-Leibler (KL Divergence):

$$ D_{KL}(P_{\text{LLM}} || P_{\text{Graph}}) = \sum_{x \in \mathcal{X}} P_{\text{LLM}}(x) \log \frac{P_{\text{LLM}}(x)}{P_{\text{Graph}}(x)} $$

Mô hình có giá trị phân kỳ (divergence) cao minh chứng cho hiện tượng tạo ảo giác phân bố ngoài (Out-of-Distribution Hallucination). Trong kiến trúc GraphRAG, $D_{KL}$ có xu hướng giảm về mức tiệm cận không $\approx 0$ do quá trình ground-truth ràng buộc cứng bởi cấu trúc Cypher AST và đồ thị.

---

**Kết Luận (Conclusion)**

Sự chuyển dịch kiến trúc học máy từ Naive RAG sang GraphRAG không đơn thuần là quá trình thay đổi cơ sở dữ liệu vật lý sang hệ quản trị cơ sở dữ liệu đồ thị, mà là một bước nhảy vọt cơ sở lý thuyết toán học định hướng. Bằng việc định hình lại bài toán giới hạn đẳng hướng đa chiều ($R^d$) qua lăng kính hình học ma trận kề thưa, cùng với phương thức quy giải thực thể nội hàm, truyền lan tín hiệu vi phân PageRank k-hop, và dung hợp truy xuất lai siêu phẳng, hệ thống Neo4j GraphRAG giải phóng hoàn toàn LLMs khỏi điểm mù của liên kết ngữ cảnh phân mảnh. Hơn thế nữa, bằng cách ứng dụng các thông số đo lường như xác suất topo hậu nghiệm và khoảng cách phân kỳ hình học, các kỹ sư AI và nhà nghiên cứu hiện nay đã có thể định lượng minh bạch hóa độ trung thành và ràng buộc triệt để hiện tượng ảo giác AI dưới góc độ chứng minh toán học nghiêm ngặt. Hệ sinh thái này mở ra một giới hạn khai thác lý luận bậc cao trên không gian tri thức phức hợp.