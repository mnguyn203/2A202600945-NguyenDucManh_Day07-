# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Đức Mạnh 
**MSSV:** 2A202600945
**Nhóm:** F2
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là hai văn bản có hướng vector gần nhau trong không gian nhúng, biểu thị chúng có chung chủ đề hoặc ý nghĩa ngữ nghĩa, bất kể độ dài thực tế của chúng.

**Ví dụ HIGH similarity:**
- Sentence A: "Thời tiết hôm nay thật đẹp."
- Sentence B: "Hôm nay trời rất trong xanh và mát mẻ."
- Tại sao tương đồng: Hai câu đều nói về chủ đề thời tiết tốt trong ngày hôm nay.

**Ví dụ LOW similarity:**
- Sentence A: "Thời tiết hôm nay thật đẹp."
- Sentence B: "Giá vàng trong nước đang tăng mạnh."
- Tại sao khác: Hai câu hoàn toàn không liên quan đến nhau về mặt chủ đề hay ngữ nghĩa (thời tiết vs. tài chính).

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chỉ đo góc giữa hai vector (ý nghĩa ngữ nghĩa) và bỏ qua độ dài của chúng (độ dài văn bản). Euclidean distance bị ảnh hưởng bởi độ dài, khiến một văn bản dài và văn bản ngắn về cùng chủ đề có khoảng cách lớn, không phản ánh đúng sự tương đồng.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* num_chunks = ceil((10,000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11)
> *Đáp án:* 23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Nếu overlap=100: num_chunks = ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = 25 chunks (số chunk sẽ tăng lên). Chúng ta muốn overlap nhiều hơn để đảm bảo ranh giới cắt không làm mất ngữ cảnh liền mạch giữa các chunk, tránh việc cắt đứt đôi một ý quan trọng.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Python Official Documentation for Building Data/RAG Applications.

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain này vì lab dùng Python để xây dựng hệ thống RAG. Bộ tài liệu lấy từ Python Official Documentation nên nguồn rõ ràng, nội dung ổn định và có cấu trúc heading/section (Markdown) rất phù hợp để thử nghiệm các chiến thuật băm dữ liệu thông minh.

### Data Inventory

| # | Tên tài liệu | Topic | Nguồn | Số ký tự | Metadata chính |
|---|--------------|-------|-------|----------|----------------|
| 1 | python_data_structures.md | data_structures | docs.python.org | 25,910 | source_url, topic, difficulty, keywords |
| 2 | python_modules.md | modules | docs.python.org | 26,216 | source_url, topic, difficulty, keywords |
| 3 | python_errors_exceptions.md | errors | docs.python.org | 25,666 | source_url, topic, difficulty, keywords |
| 4 | python_classes.md | oop | docs.python.org | 37,950 | source_url, topic, difficulty, keywords |
| 5 | python_stdlib.md | standard_library | docs.python.org | 15,100 | source_url, topic, difficulty, keywords |
| 6 | python_venv.md | environment | docs.python.org | 7,908 | source_url, topic, difficulty, keywords |
| 7 | python_input_output.md | io | docs.python.org | 22,341 | source_url, topic, difficulty, keywords |
| 8 | python_argparse.md | cli | docs.python.org | 95,464 | source_url, topic, difficulty, keywords |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `doc_id` | string | python_venv | Gom chunk theo tài liệu gốc. |
| `topic` | string | environment | Dùng cho metadata filtering theo chủ đề trước khi tính similarity. |
| `keywords` | list[string] | venv, pip | Hỗ trợ nhúng thẳng vào chunk (Context Injection) để tăng khả năng khớp ngữ nghĩa. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Strategy Của Tôi

**Loại:** Custom Strategy (`HeaderAwareChunker` / `Context-Aware Code Chunker`)

**Mô tả cách hoạt động:**
> Sử dụng RegEx để bóc tách JSON Metadata (lấy context_prefix) và cắt file theo các thẻ Markdown Headers (`##`, `###`). Đặc biệt, thuật toán nhận diện và không bao giờ cắt đôi các đoạn code block (` ```python `), đồng thời luôn ghim nội dung text giải thích ngay sát phía trên vào chung chunk với đoạn code đó.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Tài liệu lập trình luôn chứa văn bản giải thích đi kèm code mẫu. Cắt đếm ký tự bình thường sẽ làm đứt đôi đoạn code khiến LLM nhận sai cú pháp. Cắt theo Header và bảo vệ Code đảm bảo RAG agent luôn nhận được ngữ cảnh trọn vẹn (Giải thích + Code ví dụ).

### So Sánh 4 Strategies Trong Nhóm F2

Nhóm F2 có 4 thành viên, so sánh trên cùng dataset Python Official Documentation, cùng 5 benchmark queries và dùng chung embedding backend `ollama:qwen3-embedding:0.6b` để đánh giá công bằng.

| Thứ tự | Thành viên | Strategy | Chunk count | Avg length | Top-1 relevant | Top-3 relevant | Avg top-1 score |
|--------|------------|----------|-------------|------------|----------------|----------------|-----------------|
| 1 | Lê Quốc Anh | FixedSizeChunker(chunk_size=700, overlap=50) | 391 | 692.4 | 5/5 | 5/5 | 0.6733 |
| 2 | Lý Hải Long | SentenceChunker(max_sentences_per_chunk=3) | 478 | 521.4 | 5/5 | 5/5 | 0.6732 |
| 3 | Nguyễn Đức Khang | RecursiveChunker(chunk_size=700) | 474 | 525.0 | 5/5 | 5/5 | 0.6833 |
| 4 | **Nguyễn Đức Mạnh (Tôi)** | **HeaderAwareChunker(chunk_size=700)** | **519** | **479.3** | **5/5** | **5/5** | **0.6783** |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Về điểm số (Score), RecursiveChunker của bạn Khang nhỉnh hơn một chút (0.6833 vs 0.6783). Tuy nhiên, về mặt **bảo tồn ngữ cảnh (coherence)**, chiến thuật `HeaderAwareChunker` của tôi là hoàn hảo nhất cho bộ tài liệu này vì nó tôn trọng tuyệt đối cấu trúc Markdown và không phá vỡ bất kỳ khối Code nào.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng Biểu thức chính quy (Regex `re.split`) theo các dấu ngắt câu phổ biến `[.!?]`. Lọc bỏ khoảng trắng thừa, sau đó lặp qua danh sách các câu và gom lại thành từng nhóm theo `max_sentences_per_chunk`.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Hoạt động bằng đệ quy. Cố gắng chia tách văn bản bằng ký tự phân tách lớn nhất (ví dụ `\n\n`). Nếu mảng kết quả vẫn có đoạn lớn hơn `chunk_size`, nó tiếp tục gọi lại chính mình để cắt bằng ký tự nhỏ hơn (như `\n` rồi đến ` `). Base case là khi đoạn đã đủ nhỏ hoặc hết separator.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Lưu trữ nội dung, metadata và id vào một dictionary dictionary nội bộ (`self._store`). Chạy hàm nhúng (embedding) để sinh vector. Hàm search duyệt qua toàn bộ kho, tính Cosine Similarity giữa câu hỏi và từng chunk, sau đó sort giảm dần để lấy top_k.

**`search_with_filter` + `delete_document`** — approach:
> Xử lý Filter trước (tiền lọc) bằng cách duyệt qua `metadata` của record, bỏ qua những record không khớp điều kiện trước khi tính toán Similarity (rất tối ưu). Xóa document bằng cách `del self._store[doc_id]`.

### KnowledgeBaseAgent

**`answer`** — approach:
> Nhận câu hỏi, gọi `store.search` để lấy ngữ cảnh. Sau đó nối các chunk tìm được vào biến `context` và đưa vào cấu trúc Prompt Template cứng: "Use the following context to answer...". Cuối cùng đẩy Prompt này cho `llm_fn`.

### Test Results

```
test_chunking.py .............                                                                    [ 30%]
test_store.py ......................                                                              [ 83%]
test_agent.py .......                                                                             [100%]
======================================== 42 passed in 0.15s ========================================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

Sử dụng AI model (`ollama:qwen3-embedding:0.6b`) để kiểm tra:

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Python is used to build RAG systems. | Python connects embeddings, vector stores, and application logic. | high | 0.620 | Có |
| 2 | A vector store retrieves similar embeddings. | A database can search vectors by similarity. | high | 0.787 | Có |
| 3 | Customer support uses knowledge base articles. | The support team answers repeated customer questions. | high | 0.685 | Có |
| 4 | Deep learning uses neural networks. | Cooking recipes list ingredients and steps. | low | 0.338 | Có |
| 5 | Metadata filters can narrow search results. | Brown bears live in northern forests. | low | 0.142 | Có |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Bất ngờ nhất là Pair 4 vẫn có điểm 0.338 dù khác hẳn domain. Điều này cho thấy Embedding bắt được nét tương đồng về "cấu trúc mô tả một quy trình" dù từ vựng khác biệt.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên thuật toán Custom `HeaderAwareChunker` kết hợp AI `ollama:qwen3-embedding:0.6b`.

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Python list comprehension khác gì so với for loop khi tạo list mới? | List comprehension tạo list mới bằng cú pháp ngắn gọn gồm expression và for clause; for loop làm cùng việc nhưng dài hơn và cần append thủ công. |
| 2 | Python module import hoạt động như thế nào, và Module Search Path ảnh hưởng gì? | `import` nạp definitions từ module; Module Search Path quyết định Python tìm module ở thư mục script, PYTHONPATH và thư mục cài đặt chuẩn theo thứ tự nào. |
| 3 | try-except-finally / exception handling trong Python xử lý lỗi ra sao? | `try` chạy code có thể lỗi, `except` bắt exception phù hợp, `finally` chạy cleanup dù có lỗi hay không; `raise` dùng để phát sinh hoặc phát lại exception. |
| 4 | Class, instance, attribute, method trong Python OOP khác nhau thế nào? | Class định nghĩa kiểu object; instance là object tạo từ class; attribute là dữ liệu gắn với object/class; method là function thuộc class. |
| 5 | Virtual environment và pip giúp quản lý dependency conflict như thế nào? | Virtual environment tách package theo từng project; pip cài, nâng cấp và quản lý package trong môi trường đó. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Python list comprehension... | `python_data_structures_chunk_...`: 5.1.3 List Comprehensions | 0.68 | Có | Giải thích cấu trúc list comprehension ngắn gọn hơn for loop. |
| 2 | Python module import... | `python_modules_chunk_...`: 6.1.1. Executing modules as scripts | 0.66 | Có | Giải thích về sys.path và quá trình tìm kiếm file module. |
| 3 | try-except-finally... | `python_errors_exceptions_chunk_...`: 8.3. Handling Exceptions | 0.72 | Có | Nêu rõ luồng try-except, và block finally luôn chạy cuối cùng. |
| 4 | Class, instance, attribute... | `python_classes_chunk_...`: 9.3.2. Class Objects | 0.67 | Có | Nêu định nghĩa 4 khái niệm OOP theo sát tài liệu. |
| 5 | Virtual environment và pip... | `python_venv_chunk_...`: 12.3. Managing Packages with pip | 0.70 | Có | Cách pip cài thư viện cô lập vào thư mục .venv. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi học được từ bạn Khang rằng thuật toán `RecursiveChunker` tuy đơn giản nhưng lại rất hiệu quả và cân bằng. Nó không tốn quá nhiều dòng code phức tạp như regex của tôi nhưng vẫn đạt điểm retrieval trung bình cao nhất (0.6833).

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Nhóm khác đã khéo léo kết hợp Metadata Filter trước khi search. Ví dụ, họ ép query tìm trong topic "oop" trước, điều này giúp loại bỏ hoàn toàn các văn bản nhiễu, làm giảm gánh nặng cho thuật toán tính toán Similarity.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ nghiên cứu thêm phương pháp "Small-to-Big Retrieval": băm tài liệu thành các câu cực nhỏ để nhúng vector (tăng độ chính xác khi tìm kiếm), nhưng khi trả kết quả về cho LLM thì trả cả một đoạn lớn chứa câu đó để giữ nguyên ngữ cảnh.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
