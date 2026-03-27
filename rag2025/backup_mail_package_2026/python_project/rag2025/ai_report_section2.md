# Phần 2: Cơ sở Toán học của Hình học Đồ thị và Pipeline Dữ liệu

## 2.1. Biểu diễn Toán học của Đồ thị và Ma trận Kề thưa (Sparse Adjacency Matrices)

Trong không gian biểu diễn tri thức, một hệ thống đồ thị được định nghĩa hình thức là $G = (V, E)$, trong đó $V$ là tập hợp các đỉnh (thực thể/entities) với $|V| = N$, và $E \subseteq V \times \mathcal{R} \times V$ là tập hợp các cạnh có hướng (quan hệ/relations) với $|E| = M$, tương ứng với tập hợp các loại quan hệ $\mathcal{R}$.

Về mặt đại số, cấu trúc topo của $G$ được biểu diễn toàn vẹn thông qua ma trận kề $\mathbf{A} \in \mathbb{R}^{N \times N}$. Trong thực tiễn của các hệ thống đồ thị tri thức (Knowledge Graphs) công nghiệp, $N$ thường đạt quy mô từ $\mathcal{O}(10^6)$ đến $\mathcal{O}(10^9)$. Tuy nhiên, do một thực thể chỉ kết nối hữu hạn với một tập rất nhỏ các thực thể khác, đồ thị mang tính thưa (sparsity) cực đại, nghĩa là $M \ll N^2$. Sự phân bố bậc của đồ thị (degree distribution) tuân theo định luật lũy thừa (power-law distribution):
$$ P(k) \propto k^{-\gamma} $$
với $\gamma > 1$ và $k$ là bậc của đỉnh.

Để tối ưu hóa độ phức tạp không gian từ $\mathcal{O}(N^2)$ xuống $\mathcal{O}(M)$, ma trận kề $\mathbf{A}$ bắt buộc phải được biểu diễn dưới các định dạng thưa (Sparse Matrix Formats) như Compressed Sparse Row (CSR) hoặc Coordinate Format (COO). Cụ thể, định dạng CSR ánh xạ $\mathbf{A}$ thông qua ba mảng một chiều trên bộ nhớ liên tục:
- `val`: Lưu trữ trọng số của các cạnh (độ tin cậy của quan hệ) $\in \mathbb{R}^M$.
- `col_ind`: Lưu trữ chỉ số cột (thực thể đích / tail entity) $\in \mathbb{Z}^M$.
- `row_ptr`: Chỉ mục con trỏ bắt đầu của mỗi hàng trong `val` và `col_ind` $\in \mathbb{Z}^{N+1}$.

Động học của đồ thị và toán tử truyền thông điệp (Message Passing) được mô hình hóa qua ma trận Laplacian chuẩn hóa $\mathbf{L}_{sym} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$, trong đó $\mathbf{D} = \text{diag}(\mathbf{A}\mathbf{1})$ là ma trận bậc. Lõi tính toán của quá trình ánh xạ này dựa trên phép nhân ma trận thưa - đặc (SpMM - Sparse-Dense Matrix Multiplication):
$$ \mathbf{H}^{(l+1)} = \sigma \left( \tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2} \mathbf{H}^{(l)} \mathbf{W}^{(l)} \right) $$
với $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$ biểu diễn ma trận kề có chèn thêm self-loop để bảo toàn đặc trưng nội tại của thực thể.

## 2.2. Trích xuất Bản thể học (Ontological Extraction): NER & RE dựa trên Trọng số Thống kê

Việc kiến tạo topo của $G$ từ nguồn dữ liệu văn bản thô đòi hỏi một pipeline nhận dạng thực thể (NER) và trích xuất quan hệ (RE) nội suy dựa trên các hàm mật độ xác suất.

**A. Nhận dạng Thực thể (NER) - Xác lập không gian đỉnh $V$**
Giả sử chuỗi văn bản đầu vào được token hóa thành $\mathbf{X} = (\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T)$. Phân phối xác suất biên của chuỗi nhãn thực thể $\mathbf{y} = (y_1, \dots, y_T)$ được mô hình hóa chặt chẽ qua Conditional Random Field (CRF) chiếu trên không gian vector ẩn:
$$ P(\mathbf{y} | \mathbf{X}) = \frac{1}{Z(\mathbf{X})} \exp \left( \sum_{t=1}^T \mathbf{W}_{y_t}^\top \mathbf{h}_t + \sum_{t=1}^{T-1} \mathbf{T}_{y_t, y_{t+1}} \right) $$
Trong đó, $\mathbf{h}_t \in \mathbb{R}^d$ là vector biểu diễn chuỗi tại bước $t$, $\mathbf{T}$ là ma trận chuyển đổi trạng thái Markov, và $Z(\mathbf{X})$ là hàm phân hoạch (partition function) đảm bảo chuẩn hóa tổng xác suất.

**B. Trích xuất Quan hệ (RE) - Định lượng không gian cạnh $E$**
Với hai thực thể $e_i, e_j$ đã được trích xuất, bài toán RE là bài toán phân lớp đa nhãn nhằm tìm kỳ vọng hậu nghiệm của quan hệ $r \in \mathcal{R}$. Đặt $\mathbf{h}_{e_i}, \mathbf{h}_{e_j} \in \mathbb{R}^d$ là các biểu diễn đa tầng của các thực thể. Trọng số thống kê của liên kết được lượng hóa bằng hàm tương quan song tuyến tính (Bilinear Scoring Function):
$$ s(e_i, r, e_j) = \mathbf{h}_{e_i}^\top \mathbf{W}_r \mathbf{h}_{e_j} + b_r $$
Hàm phân phối hậu nghiệm được tinh chỉnh qua hàm Softmax:
$$ P(r | e_i, e_j) = \frac{\exp(s(e_i, r, e_j))}{\sum_{r' \in \mathcal{R}} \exp(s(e_i, r', e_j))} $$
Topo đồ thị được chốt lại thông qua cơ chế lọc ngưỡng (thresholding gate) $\tau$ để loại bỏ nhiễu ngẫu nhiên:
$$ E = \{(e_i, r, e_j) \mid P(r | e_i, e_j) > \tau \} $$
Mỗi cạnh thiết lập được gán trọng số $w_{ij} = P(r | e_i, e_j)$, trực tiếp hình thành các tham số phi không (non-zero entries) trong ma trận kề thưa $\mathbf{A}$.

## 2.3. Semantic Parser (LLM) và Cơ chế Attention trong Ánh xạ $(head, relation, tail)$

Trong pipeline hiện đại, các Mô hình Ngôn ngữ Lớn (LLMs) được vận hành như một Semantic Parser, thực hiện phép biến đổi đồng cấu từ không gian chuỗi tuyến tính của ngôn ngữ tự nhiên $\mathcal{S}_{text}$ sang cấu trúc topo mạng $\mathcal{S}_{graph}$. Trái tim toán học của quá trình này là cơ chế Scaled Dot-Product Attention, về bản chất là việc xây dựng một đồ thị kề đầy đủ, động và ngầm định (implicit dynamic complete graph) giữa tất cả các token.

Từ ma trận biểu diễn đầu vào $\mathbf{X} \in \mathbb{R}^{T \times d_{model}}$, các phép chiếu affine tạo ra không gian Truy vấn ($\mathbf{Q}$), Khóa ($\mathbf{K}$), và Giá trị ($\mathbf{V}$):
$$ \mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V $$
Trọng số Attention quy định luồng thông tin đồ thị, được xác định bởi:
$$ \mathbf{A}_{attn} = \text{softmax} \left( \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} \right) \in \mathbb{R}^{T \times T} $$
Ma trận $\mathbf{A}_{attn}$ đại diện cho ma trận kề mật độ cao của ngữ cảnh. Khả năng bóc tách cấu trúc Triplet $(h, r, t)$ xảy ra nhờ vào hình học vi phân của Multi-Head Attention, nơi mỗi head $\mathbf{A}_{attn}^{(i)}$ có xu hướng hội tụ học các loại "meta-relation" khác nhau.

Dưới lăng kính tự hồi quy (autoregressive modeling), xác suất ánh xạ đồng thời tạo sinh Triplet được phân rã theo chuỗi Bayes:
$$ P((h,r,t) | \mathbf{X}) = P(h | \mathbf{X}) \cdot P(t | h, \mathbf{X}) \cdot P(r | h, t, \mathbf{X}) $$
Tại trạng thái dự đoán quan hệ $r$, cơ chế Self-Attention ở các tầng sâu phân bổ cực đại phổ chú ý (attention mass) vào các token thuộc $head$ và $tail$. Gọi $\alpha_{k, r}$ là gradient của mức độ kích hoạt từ token $r$ tới token $k$, một Triplet $(h, r, t)$ được xác nhận khi hàm năng lượng tự do (free energy) của mạng giảm thiểu:
$$ \mathcal{E}(h, r, t) = - \log \left( \sum_{k \in \text{tokens}(h \cup t)} \alpha_{k, r} \right) $$
Bằng việc vi phân và tối ưu gradient của hàm Cross-Entropy Loss, LLM học cách "cứng hóa" (freeze) các đường dẫn chú ý (attention pathways) quan trọng, biến đổi chuỗi tuyến tính 1D của văn bản thành cấu trúc không gian đa chiều của đồ thị $(head, relation, tail)$. Ở đó, quan hệ $r$ hoạt động nghiêm ngặt như một toán tử tịnh tiến (translation operator) trong đa tạp vector: $\mathbf{h} + \mathbf{r} \approx \mathbf{t}$.