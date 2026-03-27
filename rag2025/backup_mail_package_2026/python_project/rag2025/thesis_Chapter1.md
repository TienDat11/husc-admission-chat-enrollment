# Chương 1: Giới thiệu chung

## 1.1. Bối cảnh và Tính cấp thiết của đề tài

Sự phát triển vượt bậc của các Mô hình Ngôn ngữ Lớn (Large Language Models, LLMs) trong thập kỷ qua đã mở ra một kỷ nguyên mới cho trí tuệ nhân tạo, tác động sâu rộng đến nhiều khía cạnh của đời sống và khoa học kỹ thuật. Các kiến trúc như Transformer đã chứng minh khả năng xử lý và sinh ngôn ngữ tự nhiên với mức độ tinh vi chưa từng có. Tuy nhiên, LLMs vẫn tồn tại những hạn chế cốt lõi chưa thể giải quyết triệt để thông qua quá trình huấn luyện đơn thuần. Chúng thường xuyên sinh ra thông tin sai lệch (hallucination), thiếu khả năng cập nhật tri thức theo thời gian thực và đặc biệt là sự thiếu hụt nghiêm trọng trong việc truy xuất dữ liệu mang tính cục bộ hoặc nội bộ của tổ chức.

Để khắc phục những nhược điểm này, hệ thống Sinh văn bản Tăng cường Truy xuất (Retrieval-Augmented Generation, RAG) đã ra đời như một giải pháp nền tảng. Bằng cách kết hợp khả năng sinh ngôn ngữ của LLM với các kho dữ liệu tri thức bên ngoài, RAG cho phép mô hình truy xuất và tham chiếu thông tin chính xác trước khi đưa ra câu trả lời. Cơ chế này không chỉ giảm thiểu hiện tượng hallucination mà còn cung cấp khả năng truy xuất nguồn gốc thông tin, yếu tố tối quan trọng trong các ứng dụng thực tiễn đòi hỏi độ tin cậy cao.

Hiện tại, đồ án đã triển khai thành công một hệ thống RAG cơ bản (được định danh là rag2025). Hệ thống này được xây dựng trên nền tảng kiến trúc vi dịch vụ (microservices) với FastAPI đóng vai trò là gateway xử lý các yêu cầu. Việc tìm kiếm thông tin được thực hiện thông qua Vector Database Qdrant, kết hợp với mô hình nhúng BGE-M3 (BAAI General Embedding) có khả năng biểu diễn đa ngôn ngữ. Bên cạnh đó, hệ thống rag2025 cũng đã tích hợp kỹ thuật HyDE (Hypothetical Document Embeddings) để nâng cao độ chính xác trong quá trình truy xuất văn bản. Sự kết hợp này mang lại hiệu năng ổn định đối với các truy vấn tìm kiếm theo độ tương đồng ngữ nghĩa (Semantic Search). Tuy nhiên, khi hệ thống phải xử lý các tập dữ liệu lớn với cấu trúc phức tạp và đòi hỏi khả năng suy luận sâu, kiến trúc RAG hiện tại bắt đầu bộc lộ những giới hạn đáng kể về mặt phương pháp lý luận.

## 1.2. Động lực nghiên cứu

Kiến trúc Naive RAG, dựa trên tìm kiếm tương đồng vector (Vector Similarity Search), vận hành bằng cách chia nhỏ tài liệu thành các khối (chunks) độc lập, sau đó chuyển đổi chúng thành các vector nhúng (embeddings). Khi có truy vấn từ người dùng, hệ thống sẽ tìm kiếm các chunks có khoảng cách gần nhất trong không gian vector. Mặc dù phương pháp này rất hiệu quả đối với các câu hỏi tra cứu thông tin trực tiếp (factoid questions), nó lại gặp phải "điểm mù" nghiêm trọng khi đối mặt với các truy vấn suy luận đa bước (multi-hop reasoning).

Khi người dùng đặt một câu hỏi yêu cầu tổng hợp thông tin từ nhiều tài liệu khác nhau hoặc kết nối các khái niệm nằm rải rác trong cơ sở dữ liệu, việc tìm kiếm dựa trên chunks riêng lẻ thường thất bại trong việc thu thập đủ manh mối. Hơn nữa, việc chia nhỏ văn bản một cách cơ học làm mất đi ngữ cảnh toàn cục (global context) của tài liệu. Một thực thể (entity) xuất hiện ở chunk A có thể liên kết chặt chẽ với một thực thể ở chunk B, nhưng mối quan hệ này bị phá vỡ hoàn toàn trong không gian vector nếu nội dung của hai chunk không chia sẻ chung các từ khóa ngữ nghĩa. 

Sự thiếu vắng khả năng nắm bắt cấu trúc và mối quan hệ giữa các thực thể dẫn đến việc LLM nhận được các đoạn văn bản rời rạc, thiếu logic kết nối. Điều này trực tiếp làm giảm chất lượng câu trả lời, thậm chí gây nhầm lẫn khi hệ thống cố gắng lắp ghép những mảnh thông tin không tương thích. Nhận thức được giới hạn này là động lực chính để nghiên cứu và tìm kiếm một mô hình truy xuất thông tin mới, vượt qua khuôn khổ của tìm kiếm vector truyền thống. Cụ thể, việc tích hợp đồ thị tri thức (Knowledge Graph) vào quy trình RAG đang nổi lên như một hướng tiếp cận đầy triển vọng nhằm giải quyết triệt để bài toán suy luận đa bước và bảo toàn ngữ cảnh cấu trúc.

## 1.3. Mục tiêu và Phạm vi

Mục tiêu cốt lõi của đề tài nghiên cứu này là thiết kế và phát triển một kiến trúc RAG lai (Vector-Graph Hybrid), đánh dấu sự chuyển đổi từ mô hình Naive RAG sang GraphRAG. Mô hình lai này sẽ kết hợp sức mạnh tìm kiếm ngữ nghĩa của Qdrant với khả năng lưu trữ cấu trúc, truy xuất mối quan hệ phức tạp của cơ sở dữ liệu đồ thị Neo4j. 

Quá trình chuyển đổi sẽ không xây dựng lại từ đầu mà sẽ kế thừa và nâng cấp trực tiếp lên kiến trúc rag2025 hiện tại. Việc tích hợp GraphRAG đòi hỏi phải thiết kế lại pipeline trích xuất thông tin, bổ sung module nhận diện thực thể và quan hệ (Entity and Relation Extraction), sau đó đồng bộ hóa dữ liệu giữa Vector Database và Graph Database. Hệ thống sau khi nâng cấp phải đảm bảo khả năng xử lý mượt mà cả hai loại truy vấn: tìm kiếm ngữ nghĩa đơn giản và suy luận đa bước phức tạp trên không gian đồ thị.

Phạm vi của nghiên cứu tập trung vào các khía cạnh kỹ thuật sau:
1. Nghiên cứu cơ sở lý thuyết về GraphRAG và các kỹ thuật trích xuất đồ thị tri thức từ văn bản thô sử dụng LLM.
2. Thiết kế mô hình dữ liệu (Schema) cho Neo4j phù hợp với đặc thù dữ liệu tuyển sinh và đào tạo đại học.
3. Tích hợp Neo4j vào hệ thống rag2025, xây dựng API giao tiếp giữa FastAPI, Qdrant và Neo4j.
4. Phát triển thuật toán định tuyến truy vấn (Query Routing) để hệ thống tự động quyết định khi nào cần sử dụng Vector Search, khi nào cần Graph Search, hoặc kết hợp cả hai.
5. Đánh giá hiệu suất của hệ thống GraphRAG mới so với mô hình Naive RAG ban đầu thông qua các độ đo tiêu chuẩn (metrics).

## 1.4. Đóng góp khoa học và thực tiễn

Đề tài mang lại những đóng góp rõ rệt trên cả phương diện học thuật và ứng dụng thực tiễn.

Về mặt học thuật, nghiên cứu cung cấp một đánh giá toàn diện và hệ thống hóa về các điểm yếu của Naive RAG trong bài toán suy luận đa bước. Việc đề xuất và thử nghiệm kiến trúc lai Vector-Graph đóng góp thêm dữ liệu thực nghiệm quan trọng vào lĩnh vực Xử lý Ngôn ngữ Tự nhiên (NLP) ứng dụng. Đồng thời, nghiên cứu cũng đề xuất một quy trình trích xuất và liên kết thực thể (Entity Linking) tối ưu cho miền dữ liệu hẹp, góp phần hoàn thiện các phương pháp tự động hóa xây dựng đồ thị tri thức.

Về mặt thực tiễn, kết quả trực tiếp của đề tài là việc nâng cấp thành công hệ thống rag2025, giải quyết dứt điểm các hạn chế trong quá trình truy xuất dữ liệu. Hệ thống mới (GraphRAG) sẽ sở hữu khả năng trả lời các câu hỏi phức tạp, đòi hỏi tổng hợp thông tin từ nhiều văn bản, quy chế hoặc hướng dẫn khác nhau. Điều này trực tiếp nâng cao trải nghiệm người dùng cuối, cải thiện độ chính xác và tính thuyết phục của câu trả lời. Hơn thế nữa, mã nguồn và kiến trúc tích hợp giữa FastAPI, Qdrant và Neo4j có thể được đóng gói và chuyển giao như một giải pháp tham khảo chuẩn mực cho các tổ chức, doanh nghiệp đang có nhu cầu xây dựng hệ thống hỏi đáp thông minh dựa trên tri thức nội bộ. 

Tóm lại, việc chuyển đổi từ Naive RAG sang GraphRAG không chỉ là một bước tiến về mặt công nghệ cho hệ thống rag2025 mà còn là minh chứng cho sự linh hoạt và khả năng mở rộng không ngừng của các kiến trúc ứng dụng LLM trong tương lai.