# Section 4: Toán học Truy xuất Lai (Hybrid Retrieval Math)

Trong kiến trúc Retrieval-Augmented Generation (RAG) hiện đại, việc kết hợp nhiều chiến lược truy xuất (như dựa trên từ khóa, vector ngữ nghĩa và đồ thị tri thức) là yếu tố quyết định để đạt được độ phủ thông tin tối ưu và tính chính xác cao. Phần này đi sâu vào nền tảng toán học của các kỹ thuật truy xuất lai, phân tích cụ thể sự hội tụ của Reciprocal Rank Fusion (RRF), cân bằng điểm số thông qua hàm mất mát phương sai, và tối ưu hóa trọng số mô hình.

## 4.1. Phân tích Giới hạn Hội tụ của Reciprocal Rank Fusion (RRF)

Reciprocal Rank Fusion là một kỹ thuật tổng hợp không giám sát hiệu quả, được thiết kế để kết hợp kết quả từ nhiều bộ truy xuất độc lập mà không cần chuẩn hóa điểm số tương đối. Công thức chuẩn của RRF cho một tài liệu $d \in D$ được định nghĩa là:

$$RRF(d) = \sum_{q \in Q} \frac{1}{k + r_q(d)}$$

Trong đó:
- $Q$ là tập hợp các bộ truy xuất (retrievers), ví dụ: BM25, Dense, Graph.
- $r_q(d)$ là thứ hạng (rank) của tài liệu $d$ theo bộ truy xuất $q$. $r_q(d) \in \mathbb{Z}^+$.
- $k$ là một hằng số làm mượt (smoothing constant), thường được chọn là $k=60$ trong thực nghiệm.

### 4.1.1. Giới hạn Tiệm cận (Asymptotic Bounds) và Hội tụ

Xét trường hợp một bộ truy xuất đơn lẻ, đóng góp điểm số cho một tài liệu ở thứ hạng $r$ là hàm:
$$f(r) = \frac{1}{k + r}$$

Đây là một chuỗi giảm đơn điệu (monotonically decreasing sequence) với $\lim_{r \to \infty} f(r) = 0$.

Trong bối cảnh hệ thống RAG chỉ lấy top $N$ kết quả, chuỗi được giới hạn. Tuy nhiên, xét về mặt toán học đối với một không gian tài liệu vô hạn, tổng đóng góp của toàn bộ không gian (nếu được tính) là chuỗi điều hòa suy rộng:
$$\sum_{r=1}^{M} \frac{1}{k + r}$$

Khi $M \to \infty$, chuỗi này phân kỳ. Tuy nhiên, giới hạn hội tụ được quan tâm thực sự là sự hội tụ về *trọng số tương đối* giữa các thứ hạng cao.

**1. Độ nhạy thứ hạng (Rank Sensitivity):**
Đạo hàm rời rạc (Discrete derivative) của $f(r)$:
$$\Delta f(r) = f(r+1) - f(r) = \frac{1}{k+r+1} - \frac{1}{k+r} = -\frac{1}{(k+r)(k+r+1)}$$

Sự thay đổi điểm số giữa thứ hạng 1 và thứ hạng 2:
$$|\Delta f(1)| = \frac{1}{(k+1)(k+2)}$$

Khi $k$ lớn (ví dụ $k=60$), $|\Delta f(1)| \approx \frac{1}{k^2}$. Điều này chứng tỏ hằng số $k$ đóng vai trò một cơ chế *Gradient Clipping* tự nhiên, giới hạn tầm ảnh hưởng của các bộ truy xuất cực đoan (outlier retrievers). Nếu một bộ truy xuất xếp tài liệu ở hạng 1, nó không thể hoàn toàn lấn át bộ truy xuất khác xếp nó ở hạng 2 hoặc 3.

**2. Điều kiện Hội tụ Đa Hệ thống (Multi-System Convergence Limit):**
Cho $|Q|$ hệ thống truy xuất. Điểm RRF tối đa một tài liệu có thể nhận được là khi nó đứng top 1 ở tất cả hệ thống:
$$\sup(RRF) = \frac{|Q|}{k+1}$$

RRF hội tụ về một phân phối đồng nhất hơn giữa các tài liệu ở top đầu nhờ giới hạn của hằng số $k$. Thay vì bị chi phối bởi hàm mũ giảm nhanh như phân phối Zipf, RRF giữ lại sự phân biệt tuyến tính ở phần đuôi dài (long tail) của các tài liệu được truy xuất.

## 4.2. Tích hợp Điểm số và Hàm Mất mát Cân bằng Phương sai (Variance-Balanced Loss)

Trong truy xuất lai, việc kết hợp điểm số tuyến tính thường bị nhiễu do sự chênh lệch về quy mô (scale) và phương sai (variance) của các điểm số (ví dụ: điểm BM25 có thể vô hạn dương, trong khi điểm Cosine Similarity của Dense Vector nằm trong đoạn $[-1, 1]$).

Hệ thống chấm điểm lai kết hợp 3 thành phần:
$$S_{hybrid}(d) = \omega_1 \cdot \tilde{S}_{BM25}(d) + \omega_2 \cdot \tilde{S}_{Dense}(d) + \omega_3 \cdot \tilde{S}_{Graph}(d)$$

Với $\sum_{i=1}^3 \omega_i = 1$ và $\tilde{S}_j(d)$ là các điểm số đã được chuẩn hóa.

### 4.2.1. Cân bằng Phương sai (Variance Balancing)

Nếu các phân phối điểm số $S_j$ có phương sai $\sigma_j^2$ khác biệt, sự kết hợp tuyến tính sẽ bị chi phối bởi phân phối có phương sai lớn nhất. Do đó, việc chuẩn hóa Z-score là cần thiết:
$$\tilde{S}_j(d) = \frac{S_j(d) - \mu_j}{\sigma_j}$$

### 4.2.2. Hàm Mất mát (Loss Function) Cân bằng

Để huấn luyện bộ kết hợp này (thường thông qua Contrastive Learning), ta định nghĩa một hàm mất mát InfoNCE được sửa đổi để kết hợp trọng số động và kiểm soát phương sai.

Cho một câu truy vấn $q$, tài liệu dương (positive) $d^+$ và tập các tài liệu âm (negative) $D^- = \{d_1^-, d_2^-, \dots, d_m^-\}$.

Hàm Contrastive Loss chuẩn:
$$\mathcal{L}_{NCE} = -\log \frac{\exp(S_{hybrid}(q, d^+) / \tau)}{\exp(S_{hybrid}(q, d^+) / \tau) + \sum_{d^- \in D^-} \exp(S_{hybrid}(q, d^-) / \tau)}$$

Với $\tau$ là nhiệt độ (temperature).

**Hàm Mất mát Cân bằng Phương sai (Variance-Regularized Loss):**
Để ngăn chặn mô hình chỉ tối ưu hóa một hệ thống (ví dụ: chỉ gán $\omega_2 = 1$ và bỏ qua BM25/Graph), ta thêm một số hạng điều chuẩn (regularization term) để cân bằng phương sai đóng góp của mỗi nhánh:

$$\mathcal{L}_{total} = \mathcal{L}_{NCE} + \lambda \sum_{i=1}^3 \left( \mathbb{E}_{d \in \{d^+\} \cup D^-}[\omega_i \tilde{S}_i(d)] - \frac{1}{3} \mathbb{E}_{d}[S_{hybrid}(d)] \right)^2$$

Thành phần thứ hai là *Variance Balancing Penalty*. Nó ép các nhánh phải có kỳ vọng đóng góp điểm số ngang bằng nhau (tương đối), ngăn chặn một tín hiệu đơn lẻ chi phối hoàn toàn quá trình huấn luyện kết hợp. $\lambda$ là hệ số siêu tham số (hyperparameter) kiểm soát mức độ ràng buộc.

## 4.3. Huấn luyện (Weight Optimization): Phương trình Gradient Descent Siêu phẳng Lai

Việc tối ưu hóa các trọng số $\omega_i$ trên một đa tạp (manifold) $\sum \omega_i = 1, \omega_i \geq 0$ tạo thành một bài toán tối ưu trên đơn hình chuẩn (standard simplex) hay siêu phẳng lai (hybrid hyperplane).

### 4.3.1. Phép chiếu Đơn hình (Simplex Projection)

Nếu cập nhật trọng số bằng Gradient Descent tiêu chuẩn:
$$\omega^{(t+1)} = \omega^{(t)} - \eta \nabla_{\omega} \mathcal{L}_{total}(\omega^{(t)})$$

Vector $\omega^{(t+1)}$ có thể vi phạm điều kiện ràng buộc của đơn hình ($\sum \omega_i = 1$). Do đó, ta cần sử dụng **Projected Gradient Descent (PGD)**:
$$\omega^{(t+1)} = \Pi_{\Delta} \left( \omega^{(t)} - \eta \nabla_{\omega} \mathcal{L}_{total}(\omega^{(t)}) \right)$$

Trong đó $\Pi_{\Delta}(x)$ là toán tử chiếu lên đơn hình:
$$\Pi_{\Delta}(x) = \arg\min_{\omega \in \Delta} ||x - \omega||_2^2$$

### 4.3.2. Động lực học Gradient trên Siêu phẳng (Gradient Dynamics on Hyperplane)

Đạo hàm riêng của hàm mất mát đối với trọng số $\omega_i$:
$$\frac{\partial \mathcal{L}_{NCE}}{\partial \omega_i} = \frac{1}{\tau} \left( \sum_{d^- \in D^-} P(d^- | q) \tilde{S}_i(q, d^-) - \tilde{S}_i(q, d^+) \right)$$

Trong đó $P(d | q)$ là xác suất theo phân phối Softmax:
$$P(d | q) = \frac{\exp(S_{hybrid}(q, d)/\tau)}{\sum_{d' \in \{d^+\} \cup D^-} \exp(S_{hybrid}(q, d')/\tau)}$$

Phương trình này chỉ ra rằng gradient đối với $\omega_i$ chính là *kỳ vọng phần dư (expected residual)* của điểm số thành phần $\tilde{S}_i$ dưới phân phối hiện tại của mô hình. 

**Cập nhật Softmax-Parameterized (Thay thế cho Phép chiếu):**
Thay vì chiếu PGD, một phương pháp hiệu quả hơn là tham số hóa $\omega$ bằng một lớp Softmax để thỏa mãn tự nhiên ràng buộc siêu phẳng:
$$\omega_i = \frac{\exp(\theta_i)}{\sum_{j=1}^3 \exp(\theta_j)}$$

Khi đó, gradient theo $\theta$ tuân theo đạo hàm dây chuyền (chain rule):
$$\frac{\partial \mathcal{L}}{\partial \theta_j} = \sum_{i=1}^3 \frac{\partial \mathcal{L}}{\partial \omega_i} \cdot \frac{\partial \omega_i}{\partial \theta_j}$$
$$= \omega_j \left( \frac{\partial \mathcal{L}}{\partial \omega_j} - \sum_{i=1}^3 \omega_i \frac{\partial \mathcal{L}}{\partial \omega_i} \right)$$

Phương trình động lực học này đại diện cho sự thay đổi quỹ đạo trên **siêu phẳng lai (hybrid hyperplane)**. Thành phần $\sum_{i=1}^3 \omega_i \frac{\partial \mathcal{L}}{\partial \omega_i}$ đóng vai trò là một "động lượng trung bình" (mean momentum). Trọng số $\theta_j$ chỉ tăng (tức $\omega_j$ nhận thêm độ tin cậy) nếu gradient cụ thể của nó lớn hơn gradient trung bình có trọng số của toàn hệ thống. Cơ chế này tự động đào thải các luồng truy xuất không cung cấp tín hiệu tích cực vượt bậc so với hệ thống tổng thể.