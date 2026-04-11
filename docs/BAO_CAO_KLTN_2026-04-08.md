# Báo Cáo KLTN 8/4/2026

## I. Công việc đã làm

### 1. Phương pháp thực hiện

#### a. Chuẩn hóa dữ liệu

Đề tài sử dụng bộ dữ liệu **ViTextVQA**, là bài toán Visual Question Answering tiếng Việt với trọng tâm là nội dung văn bản xuất hiện trong ảnh. Dữ liệu đầu vào gồm ảnh, câu hỏi và kết quả OCR theo từng vùng chữ.

Ở bước chuẩn hóa, dữ liệu được xử lý theo các bước sau:

- Chuẩn hóa Unicode cho văn bản theo dạng chuẩn thống nhất để giảm lỗi lệch dấu và lỗi encoding.
- Thay thế khoảng trắng bất thường như `non-breaking space`, đồng thời gộp nhiều khoảng trắng liên tiếp về một khoảng trắng duy nhất.
- Chuẩn hóa câu hỏi và text OCR trước khi embedding để bảo đảm text truy hồi và text trong graph cùng một chuẩn biểu diễn.
- Thực hiện **word segmentation tiếng Việt** trước khi embedding. Trong pipeline hiện tại, backend tokenizer được thiết kế theo cơ chế `auto`, ưu tiên `underthesea`; nếu môi trường không có `underthesea` thì fallback sang **PyVi (`ViTokenizer`)**. Như vậy về bản chất, hệ thống dùng bộ tách từ tiếng Việt để biến câu hỏi và text OCR thành chuỗi phù hợp hơn cho embedding retrieval.

Với một văn bản đầu vào `x`, hàm chuẩn hóa text có thể tóm tắt như sau:

\[
x_{norm} = \text{CollapseWhitespace}(\text{UnicodeNormalize}(x))
\]

Sau đó, nếu bật tách từ tiếng Việt:

\[
x_{seg} = \text{Tokenizer}(x_{norm})
\]

Trong đó `Tokenizer` có thể là `underthesea` hoặc `PyVi` tùy môi trường thực thi. Chuỗi `x_seg` là đầu vào cho mô hình embedding `BAAI/bge-m3`.

#### b. Xây dựng spatial graph

Sau khi có kết quả OCR theo từng bbox, hệ thống chuyển mỗi bbox thành một node cơ sở, rồi xây graph theo thứ tự đọc **từ trên xuống dưới, từ trái sang phải**. Với bộ dữ liệu này, cách sắp xếp theo **top-left -> right -> bottom** được sử dụng vì phù hợp hơn với cấu trúc văn bản trong ảnh và cho hiệu quả thực nghiệm tốt hơn khi truy hồi ngữ cảnh.

Quy trình xây graph gồm 3 bước chính:

##### Bước 1. Tạo base nodes

Mỗi bbox OCR được chuyển thành một node chứa:
- `text`
- `bbox = [x1, y1, x2, y2]`
- tọa độ tâm `cx, cy`
- `width, height`

Các node cơ sở được sắp xếp theo:

\[
\text{order}(v) = (y_1(v), x_1(v))
\]

tức là ưu tiên cao hơn cho node nằm cao hơn, sau đó đến node nằm bên trái hơn.

##### Bước 2. Merge ngang

Hai bbox được gộp ngang nếu đồng thời thỏa:

- `next_box.x1 >= current_box.x2`
- độ chồng lấp theo trục dọc đủ lớn
- khoảng cách ngang đủ nhỏ
- tỷ lệ chiều cao không quá chênh lệch

Cụ thể trong cấu hình hiện tại:

- `y_overlap_for_horizontal_merge = 0.55`
- `max_horizontal_gap = 28`
- `max_height_ratio = 2.2`

Điều kiện merge ngang có thể viết:

\[
\text{MergeH}(a,b) =
[\text{yOverlap}(a,b) \ge 0.55]
\land
[\text{hGap}(a,b) \le 28]
\land
\left[\frac{\max(h_a,h_b)}{\min(h_a,h_b)} \le 2.2\right]
\]

##### Bước 3. Merge dọc

Sau khi merge ngang, các node tiếp tục được xét merge theo chiều dọc nếu:

- bbox phía dưới thực sự nằm thấp hơn bbox hiện tại
- độ chồng lấp theo trục ngang đủ lớn
- khoảng cách dọc đủ nhỏ
- tâm theo trục `x` không lệch quá nhiều
- độ rộng giữa hai bbox không chênh lệch mạnh

Tham số hiện tại:

- `x_overlap_for_vertical_merge = 0.60`
- `x_overlap_for_vertical_merge_on_max = 0.45`
- `max_vertical_gap = 10`
- `max_width_ratio = 2.2`
- `max_center_x_diff_ratio = 0.30`

Điều kiện merge dọc:

\[
\text{MergeV}(a,b) =
[\text{xOverlap}(a,b) \ge 0.60]
\land
[\text{xOverlapOnMax}(a,b) \ge 0.45]
\land
[\text{vGap}(a,b) \le 10]
\land
\left[\frac{\max(w_a,w_b)}{\min(w_a,w_b)} \le 2.2\right]
\land
[|cx_a-cx_b| \le 0.30 \cdot \min(w_a,w_b)]
\]

##### Bước 4. Tạo cạnh không gian

Trên tập node đã merge, mỗi node được nối với:

- **RIGHT_NEIGHBOR**: node bên phải gần nhất, đủ chồng lấp theo chiều dọc
- **DOWN_NEIGHBOR**: node phía dưới gần nhất, đủ chồng lấp theo chiều ngang

Tham số:

- `y_overlap_for_right_edge = 0.30`
- `x_overlap_for_down_edge = 0.30`
- `max_right_gap = 180`
- `max_down_gap = 160`

Như vậy graph cuối cùng là một **spatial text graph**, trong đó node là vùng text đã merge, còn edge biểu diễn quan hệ đọc gần nhất theo chiều ngang và chiều dọc.

#### c. Xây embedding cho node text và node crop

Mỗi node trong graph được gắn hai loại embedding:

- **text embedding** bằng `BAAI/bge-m3`
- **image crop embedding** bằng `openai/clip-vit-base-patch32`

Text embedding được tính trên text đã chuẩn hóa và tách từ. Crop embedding được tính trên vùng ảnh crop tương ứng với bbox node. Hai embedding này được lưu offline để tăng tốc độ truy hồi khi chạy batch trên tập test.

#### d. Làm giàu graph cho CSE

Trước khi truy hồi, graph được làm giàu bằng các thống kê offline:

##### Độ tương đồng embedding giữa hai node nối cạnh

Với hai vector embedding \(u\) và \(v\), hệ thống dùng cosine similarity chuẩn hóa:

\[
\text{cos}(u,v)=\frac{u\cdot v}{\|u\|\|v\|}
\]

\[
\text{sim}_{norm}(u,v)=\frac{1+\text{cos}(u,v)}{2}
\]

Từ đó:

- `conf_off_text`: độ giống nhau giữa text embedding của source và target
- `conf_off_image`: độ giống nhau giữa crop embedding của source và target

Điểm confidence offline của cạnh:

\[
\text{conf\_off} =
\frac{w_t \cdot \text{conf\_off\_text} + w_i \cdot \text{conf\_off\_image}}
{w_t + w_i}
\]

với:

- `conf_text_weight = 0.4`
- `conf_image_weight = 0.6`

##### Hub penalty

Mỗi node được gắn thêm độ hub:

\[
\text{hub}(v)=\log(1+\deg(v))
\]

trong đó:

\[
\deg(v)=\deg_{in}(v)+\deg_{out}(v)
\]

Hub penalty được dùng để giảm ưu tiên những node có quá nhiều kết nối, tránh lan rộng vào các vùng quá chung chung.

#### e. Truy hồi text-first CSE

Đây là phần quan trọng nhất trong pipeline.

##### Bước 1. Tính độ liên quan giữa query và từng node

Từ câu hỏi \(q\), hệ thống tạo:

- `query_text_embedding` bằng `bge-m3`
- `query_image_embedding` bằng CLIP text encoder

Với mỗi node \(v\), tính:

- `rel_text(q,v)`: độ giống giữa query text embedding và text embedding của node
- `rel_image(q,v)`: độ giống giữa query image-text embedding và crop image embedding của node

Điểm liên quan cuối cùng:

\[
\text{rel}(q,v)=
\frac{\lambda_t \cdot \text{rel\_text}(q,v) + \lambda_i \cdot \text{rel\_image}(q,v)}
{\lambda_t+\lambda_i}
\]

với cấu hình hiện tại:

- `rel_text_weight = 0.4`
- `rel_image_weight = 0.6`

##### Bước 2. Chọn top-k seed nodes

Các node seed được chọn bằng cách sắp xếp tất cả text node theo `rel(q,v)` giảm dần, rồi lấy:

\[
\text{Seeds}(q)=\text{TopK}_{v \in V_{text}} \ \text{rel}(q,v)
\]

Cấu hình thực nghiệm hiện tại trong batch test:

- `top_k = 5`

Tức là từ mỗi câu hỏi, hệ thống lấy 5 text node liên quan nhất làm điểm khởi đầu cho quá trình mở rộng.

##### Bước 3. Mở rộng CSE

Từ mỗi seed node, hệ thống duyệt theo các cạnh của graph trong số hop giới hạn. Với mỗi cạnh từ source sang target, điểm mở rộng được tính:

\[
\text{score}(e: u \rightarrow v)=
\alpha \cdot \text{conf\_off}(u,v)
\;+\;
\text{rel}(q,v)
\;-\;
\lambda_{hub}\cdot \text{hub}(v)
\]

Trong đó:

- \(\alpha\) là trọng số của confidence offline
- \(\text{rel}(q,v)\) là độ liên quan của node đích với câu hỏi
- \(\lambda_{hub}\) là hệ số phạt node hub

Tham số batch test hiện tại:

- `hops = 3`
- `top_m = 5`
- `threshold = 0.35`
- `alpha = 0.5`
- `lambda_hub = 0.05`

Quy tắc mở rộng:

- tại mỗi hop, với mỗi node frontier, chỉ giữ tối đa `top_m = 5` cạnh tốt nhất
- chỉ nhận các cạnh có `score >= threshold`
- dừng sớm nếu số node vượt `max_nodes = 100` hoặc số cạnh vượt `max_edges = 200`

##### Bước 4. Chấm điểm subgraph

Mỗi seed tạo ra một subgraph riêng. Với node seed:

\[
\text{final\_score}(v_{seed}) = \text{rel}(q,v_{seed})
\]

Với node được mở rộng:

\[
\text{final\_score}(v)=\max \text{score}(e)
\]

Điểm subgraph được tính bằng trung bình của top-3 node score cao nhất trong subgraph:

\[
\text{subgraph\_score} =
\frac{1}{3}\sum_{i=1}^{3} s_i
\]

nếu subgraph có ít hơn 3 node thì lấy trung bình trên số node hiện có.

Sau đó các subgraph được sắp xếp theo `subgraph_score` giảm dần.

#### f. Đưa ngữ cảnh vào mô hình VLM

Từ các subgraph đã truy hồi, hệ thống lấy:

- tối đa `max_subgraphs = 2`
- mỗi subgraph lấy tối đa `max_nodes_per_subgraph = 3`
- tối đa `max_crops = 4` crop OCR

Prompt đưa vào mô hình gồm:

- câu hỏi gốc
- danh sách node OCR quan trọng
- `node_id`
- `bbox`
- `text`
- `final_score`
- `rel`

Đầu vào thị giác gồm:

- ảnh gốc
- các crop OCR quan trọng

Nhờ vậy, mô hình không chỉ nhìn ảnh mà còn được cung cấp ngữ cảnh retrieval rõ ràng hơn.

#### g. Mô hình sử dụng

Mô hình chính dùng trong giai đoạn này là:

- **Qwen2.5-VL-7B**

Cấu hình thực nghiệm batch test:

- `temperature = 0.7`
- `max_new_tokens = 256`

Ngoài ra pipeline cũng đã được tổ chức để có thể thử nghiệm thêm với Gemini và Vintern, tuy nhiên kết quả chính trong giai đoạn báo cáo này tập trung vào Qwen2.5-VL-7B.

### 2. Nguồn dữ liệu

Bộ dữ liệu sử dụng là **ViTextVQA**.

Quy mô dữ liệu hiện dùng:

- Tập `dev`: **5.155** câu hỏi
- Tập `test`: **10.028** câu hỏi

Đây là bộ dữ liệu phù hợp với hướng nghiên cứu vì yêu cầu mô hình phải khai thác được cả nội dung text trong ảnh lẫn quan hệ không gian giữa các vùng văn bản.

## II. Kết quả thực hiện

### 1. Kết quả hệ thống

Đến thời điểm hiện tại, đề tài đã xây dựng được pipeline hoàn chỉnh theo chuỗi:

\[
\text{OCR} \rightarrow \text{Spatial Graph} \rightarrow \text{Embedding} \rightarrow \text{Text-first CSE} \rightarrow \text{Qwen2.5-VL-7B}
\]

Pipeline đã chạy được trên toàn bộ tập test của ViTextVQA và hỗ trợ suy luận hàng loạt với cơ chế:

- load model một lần
- lưu kết quả online sau từng câu
- resume khi bị ngắt
- dọn cache để phù hợp hơn với môi trường GPU hạn chế bộ nhớ

### 2. Kết quả định lượng

Với cấu hình thực nghiệm nội bộ tốt nhất trong giai đoạn hiện tại, hệ thống đạt khoảng:

- **Exact Match (EM)**: `0.3422`
- **F1**: `0.5606`

Ở một cấu hình ổn định khác, kết quả đạt khoảng:

- **EM**: `0.3069`
- **F1**: `0.5176`

Các kết quả trên cho thấy việc bổ sung OCR retrieval và spatial graph giúp nâng chất lượng trả lời rõ rệt, đặc biệt với câu hỏi cần bám sát văn bản trong ảnh.

### 3. Nhận xét

- Mô hình Qwen2.5-VL-7B cho kết quả tốt hơn khi được cung cấp ngữ cảnh retrieval thay vì chỉ dùng ảnh gốc.
- Text-first CSE phù hợp với ViTextVQA do câu hỏi thường bám trực tiếp vào nội dung chữ trong ảnh.
- Việc sắp xếp và merge theo thứ tự đọc top-left -> right -> bottom giúp graph sát hơn với bố cục đọc tự nhiên của dữ liệu.
- Thành phần `rel_image` và `conf_off_image` từ CLIP hỗ trợ tốt cho các trường hợp OCR text ngắn hoặc mơ hồ.

## III. Công việc dự tính

### 1. Cải thiện chất lượng retrieval

- Tối ưu thêm các tham số `top_k`, `hops`, `top_m`, `threshold`, `alpha`, `lambda_hub`.
- Thử nghiệm thêm các chiến lược seed selection khác ngoài top-k theo `rel`.
- Đánh giá sâu hơn vai trò của `rel_text` và `rel_image` trên từng nhóm câu hỏi.

### 2. Cải thiện độ ổn định pipeline

- Bổ sung fallback khi thiếu `graph_enriched`, `text_embeddings` hoặc `crop_embeddings` để tránh dừng cả batch.
- Tối ưu thêm bộ nhớ cho suy luận trên GPU 16GB-24GB.
- Chuẩn hóa quy trình chạy trên môi trường Vast.ai để giảm lỗi phụ thuộc thư viện.

### 3. Hoàn thiện thực nghiệm khóa luận

- So sánh với các backend khác như Gemini và Vintern.
- Phân tích lỗi theo từng nhóm câu hỏi.
- Hoàn thiện bảng kết quả, hình minh họa pipeline và phần thảo luận trong báo cáo chính thức.
