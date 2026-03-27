# Phần 1: Cơ sở Toán học của Hệ thống RAG Nguyên thuỷ (Naive RAG)

## 1.1. Không gian Vector và Ánh xạ Nhúng Ngữ nghĩa (Semantic Embedding)

Trong mô thức của RAG nguyên thuỷ (Naive Retrieval-Augmented Generation), quá trình đối chiếu thông tin được thiết lập thông qua một không gian biểu diễn liên tục. Giả sử tập hợp các đơn vị văn bản đầu vào là không gian tô-pô rời rạc $\mathcal{T}$. Quá trình nhúng (embedding) thực chất là việc định nghĩa và tối ưu hóa một ánh xạ mêtric phi tuyến $f_\theta: \mathcal{T} \rightarrow \mathbb{R}^d$, trong đó $d$ là số lượng chiều của không gian ẩn (latent space) và $\theta$ đại diện cho tập trọng số của một mô hình ngôn ngữ học sâu.

Ánh xạ $f_\theta$ có nhiệm vụ bảo toàn cấu trúc tô-pô ngữ nghĩa cục bộ: các văn bản có mức độ tương đồng ngữ nghĩa cao trong $\mathcal{T}$ sẽ được ánh xạ thành các vector có khoảng cách mêtric nhỏ trong $\mathbb{R}^d$. Việc sử dụng các hàm kích hoạt phi tuyến (như GELU hay ReLU) kết hợp với cơ chế tự chú ý (self-attention) cho phép hệ thống xấp xỉ các đa tạp (manifolds) phức tạp mang tính phi tuyến tính cao của ngôn ngữ tự nhiên. Nhờ đó, đặc trưng ngữ nghĩa của văn bản được mã hoá chặt chẽ và chiếu lên một không gian Hilbert hữu hạn chiều $\mathbb{R}^d$, làm nền tảng cho các phép nội suy toán học.

## 1.2. Đo lường Độ tương đồng Cosine (Cosine Similarity) và Phân tích Hình học

Để định lượng mức độ tiệm cận về mặt ngữ nghĩa giữa vector truy vấn (query) $A \in \mathbb{R}^d$ và vector tài liệu $B \in \mathbb{R}^d$, hàm đo lường khoảng cách phổ biến nhất là độ tương đồng Cosine (Cosine Similarity). Hàm đo lường này được định nghĩa là tỷ số giữa tích vô hướng và tích các chuẩn Euclid của hai vector:

$$ S_C(A, B) = \frac{\langle A, B \rangle}{\|A\|_2 \|B\|_2} = \cos(\psi) $$

trong đó $\psi$ là góc hợp bởi hai vector $A$ và $B$ trong không gian $\mathbb{R}^d$. 

Dưới góc độ hình học phân phối từ vựng, phép đo này thực chất là việc đánh giá khoảng cách trắc địa của các điểm dữ liệu sau khi chúng được hình chiếu chuẩn hoá lên một mặt cầu đơn vị chiều $d-1$ (unit hypersphere $\mathcal{S}^{d-1}$). Trong các không gian có số chiều cực lớn, do hiệu ứng bùng nổ chiều (curse of dimensionality), khối lượng phân phối của các vector nhúng thường có xu hướng tập trung ở một vành đai mỏng tiệm cận xích đạo của mặt cầu. Phép đo $S_C(A,B)$ loại bỏ triệt để ảnh hưởng của độ lớn vector (thường đại diện cho tần suất xuất hiện và độ dài bề mặt của văn bản), tập trung tinh cất độ lệch góc – đại diện thuần tuý cho hướng phân bố đặc trưng ngữ nghĩa trong không gian.

## 1.3. Hạn chế của Tích vô hướng trong Suy luận Đa bước (Multi-hop Reasoning)

Mặc dù tích vô hướng $\langle A, B \rangle$ (là cơ sở của $S_C$) minh chứng được tính hiệu quả cao trong các bài toán đối sánh trực tiếp (single-hop retrieval), nền tảng mêtric này bộc lộ sự suy thoái nghiêm trọng khi hệ thống phải thực hiện suy luận đa bước (multi-hop reasoning).

Trong không gian vector $\mathbb{R}^d$, đặc biệt khi được tối ưu hóa theo hàm mất mát đối lập (contrastive loss) với giả định âm lấy mẫu cục bộ, các vector nhúng thường bị cưỡng ép vào một phân bố đẳng hướng (isotropic vector space). Khi một truy vấn yêu cầu đối chiếu chuỗi thông tin liên kết (ví dụ: tìm mối liên hệ $X \rightarrow Y \rightarrow Z$), việc đánh giá qua tích vô hướng vấp phải sự suy thoái thông tin (information degradation). Nếu một vector truy vấn $q$ chỉ có thể liên kết mêtric tuyến tính với tập $X$, tích vô hướng không bảo toàn được các tính chất giao hoán hay bắc cầu cho tập $Z$ ẩn phía sau. 

Hơn nữa, mức độ nhiễu loạn phân phối (distributional shift) và nhiễu phi tuyến tích luỹ sau mỗi "bước nhảy" (hop) trên mặt cầu đa tạp sẽ làm triệt tiêu nhanh chóng biên độ của các tín hiệu ngữ nghĩa hữu ích. Cấu trúc liên kết đồ thị phức tạp như quan hệ nhân quả hay logic thời gian bị ép phẳng (flattened) trên một không gian vector đẳng hướng, khiến cho các phép chiếu vô hướng trở nên bất lực trong việc khôi phục chuỗi biểu diễn phụ thuộc, từ đó dẫn đến hiện tượng ảo giác truy xuất (retrieval hallucination) trong các hệ thống Naive RAG.