# 5. Các Độ đo Đánh giá Kỹ thuật (Evaluation Metrics)

Để đánh giá hiệu năng của hệ thống RAG (Retrieval-Augmented Generation) tăng cường trên không gian đồ thị tri thức, chúng tôi đề xuất một hệ thống độ đo toàn diện. Phần này trình bày các công thức toán học chặt chẽ nhằm định lượng khả năng truy xuất đồ thị, tính trung thực của phản hồi và mức độ ảo giác.

## 5.1. Đánh giá Truy xuất Đồ thị (Graph Retrieval Metrics)

Trong không gian đồ thị $G = (V, E)$, mỗi truy vấn $q$ sẽ trả về một danh sách các nút, đường dẫn hoặc đồ thị con (subgraphs) được xếp hạng. Chúng tôi mở rộng các độ đo truyền thống để áp dụng trên topo đồ thị.

### 5.1.1. Mean Reciprocal Rank trên Không gian Đồ thị (Graph-MRR)

Graph-MRR đo lường thứ hạng của thực thể hoặc đường dẫn đồ thị liên quan đầu tiên được hệ thống trả về. Cho một tập hợp các truy vấn $Q$, với mỗi truy vấn $q \in Q$, gọi $rank_G(q)$ là vị trí của thực thể/đồ thị con đúng đầu tiên trong danh sách kết quả được xếp hạng theo khoảng cách địa lý đồ thị (graph geodesic distance).

$$ Graph-MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_G(q_i)} $$

Trong đó, $rank_G(q_i)$ được điều chỉnh bởi trọng số topo, ưu tiên các thực thể có độ trung tâm (centrality) cao trong ngữ cảnh truy vấn.

### 5.1.2. Normalized Discounted Cumulative Gain (NDCG)

Để đánh giá chất lượng của toàn bộ danh sách kết quả trả về, Graph-NDCG tính toán điểm số dựa trên mức độ liên quan của các thực thể/đường dẫn $r_i$ tại vị trí $i$. Mức độ liên quan này được xác định bởi hàm tương tự cấu trúc (structural similarity) giữa đồ thị con được truy xuất và đồ thị tri thức nền (ground-truth subgraph).

$$ DCG_p = \sum_{i=1}^{p} \frac{rel_i}{\log_2(i+1)} $$

$$ NDCG_p = \frac{DCG_p}{IDCG_p} $$

Ở đây, $rel_i \in [0, 1]$ là điểm liên quan topo của thực thể thứ $i$, tính bằng độ trùm nội dung hoặc PageRank cục bộ (Personalized PageRank) xung quanh truy vấn $q$. $IDCG_p$ là điểm DCG tối đa có thể đạt được (Ideal DCG).

## 5.2. Định lượng Tính trung thực (Faithfulness)

Tính trung thực đánh giá mức độ mà nội dung đầu ra (Output) bám sát và suy diễn trực tiếp từ ngữ cảnh được cung cấp (Context) và đồ thị tri thức (Graph). Chúng tôi mô hình hóa tính trung thực dưới dạng Xác suất Topo Hậu nghiệm (Posterior Topological Probability).

Gọi $O$ là văn bản đầu ra, $C$ là ngữ cảnh được truy xuất và $G_{sub} \subset G$ là đồ thị con làm nền tảng. Xác suất để $O$ được sinh ra một cách trung thực từ $C$ và $G_{sub}$ là:

$$ \mathcal{F} = P(O | C, G_{sub}) = \frac{P(O, C, G_{sub})}{P(C, G_{sub})} $$

Áp dụng quy tắc Bayes và giả định độc lập có điều kiện giữa cấu trúc đồ thị và văn bản ngữ cảnh, ta khai triển:

$$ P(O | C, G_{sub}) \propto P(O | C) \cdot \exp(-\lambda \cdot \mathcal{D}_{KL}(E(O) || E(G_{sub}))) $$

Trong đó:
* $P(O | C)$ là mô hình sinh ngôn ngữ (Language Model Likelihood).
* $E(\cdot)$ là hàm trích xuất tập hợp các quan hệ (triplets) từ văn bản hoặc đồ thị.
* $\mathcal{D}_{KL}$ là phân kỳ Kullback-Leibler đo lường sự khác biệt phân bố quan hệ giữa đầu ra và đồ thị gốc.
* $\lambda$ là siêu tham số kiểm soát mức độ ràng buộc cấu trúc.

Điểm trung thực càng tiến gần về 1, mô hình càng ít tự bịa đặt thông tin ngoài đồ thị.

## 5.3. Lượng hóa Ảo giác (Hallucination Quantification)

Ảo giác (Hallucination) xảy ra khi mô hình sinh ra nội dung mâu thuẫn hoặc không tồn tại trong tập dữ liệu gốc. Chúng tôi tiếp cận vấn đề lượng hóa ảo giác qua hai góc độ: ngữ nghĩa từ vựng và cấu trúc hình học.

### 5.3.1. Khoảng cách Ngữ nghĩa Levenshtein (Semantic Levenshtein Distance)

Để phát hiện sự biến tấu thực thể sinh ra ảo giác thông tin, chúng tôi áp dụng Khoảng cách Levenshtein trên không gian embedding nhúng (Embedding Space) thay vì ký tự thuần túy. Cho tập hợp các thực thể được nhắc đến trong đầu ra $E_O$ và tập hợp các thực thể đúng trong ngữ cảnh đồ thị $E_C$.

Khoảng cách ngữ nghĩa giữa một thực thể sinh ra $e_o \in E_O$ và tập $E_C$ được định nghĩa là:

$$ d_{sem}(e_o, E_C) = \min_{e_c \in E_C} \left( 1 - \text{cos\_sim}(v(e_o), v(e_c)) \right) $$

Chỉ số ảo giác ngữ nghĩa của toàn bộ câu trả lời $H_{sem}$ là trung bình khoảng cách của các thực thể vượt qua một ngưỡng dung sai $\tau$:

$$ H_{sem} = \frac{1}{|E_O|} \sum_{e_o \in E_O} d_{sem}(e_o, E_C) \cdot \mathbf{I}(d_{sem} > \tau) $$

Trong đó $v(\cdot)$ là vector biểu diễn (embedding), và $\mathbf{I}(\cdot)$ là hàm chỉ thị.

### 5.3.2. Độ lệch Hình học (Geometric Deviation)

Độ lệch hình học đánh giá sự sai lệch về mặt cấu trúc suy luận. Nếu mô hình sinh ra một chuỗi lập luận (ví dụ: $A \rightarrow B \rightarrow C$), nhưng khoảng cách ngắn nhất giữa $A$ và $C$ trong đồ thị gốc $G$ lại rất xa (hoặc không tồn tại đường đi), đó là dấu hiệu của ảo giác cấu trúc (Structural Hallucination).

Cho một đường dẫn sinh ra $P_O = (v_1, v_2, ..., v_k)$. Độ lệch hình học $\Delta_{geom}$ được tính bằng độ khác biệt giữa độ dài đường dẫn sinh ra và khoảng cách trắc địa thực tế (geodesic distance $d_G$) trên không gian cong của đồ thị:

$$ \Delta_{geom}(P_O) = \sum_{i=1}^{k-1} \left( 1 - \exp\left(-\gamma \cdot d_G(v_i, v_{i+1})\right) \right) $$

Khi $\Delta_{geom}$ có giá trị lớn, điều đó chứng tỏ mô hình đã tự ý "tạo ra" các kết nối logic không được hỗ trợ bởi đồ thị gốc tri thức, một hình thái ảo giác đặc biệt nguy hiểm trong các ứng dụng RAG đòi hỏi độ chính xác cao.