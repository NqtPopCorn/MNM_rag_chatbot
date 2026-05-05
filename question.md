# Bộ câu hỏi kiểm tra RAG Chatbot

> **Tài liệu nguồn:**
> - **Bài 1** — *Nghiên cứu và Phát triển Hệ thống Giám sát Môi trường Thông minh dựa trên IoT và Học sâu* (Nguyễn Văn A, Trần Thị B — HUST)
> - **Bài 2** — *Nâng cao hiệu quả hệ thống RAG sử dụng kỹ thuật Hybrid Search và Tái xếp hạng đa tầng cho dữ liệu văn bản pháp quy Việt Nam* (Ngô Quang Trường, Nguyễn Thị Khoa Học — SGU)
> - **Bài 3** — *Ứng dụng Mạng Nơ-ron Tích chập (CNN) trong Tự động Nhận diện Bệnh hại trên Lá Lúa qua Hình ảnh Quang học* (Trần Quang Minh, Lê Thị Thanh Thảo — VNU)

---

## A. Kiểm tra chức năng cơ bản

Mục tiêu: xác nhận retrieval đúng chunk, đọc được số liệu, xử lý câu hỏi ngoài ngữ cảnh tài liệu.

### Câu 1 — Truy xuất thực tế (Bài 1)

> Hệ thống IoT trong bài báo sử dụng vi điều khiển và cảm biến gì? Dữ liệu được lấy mẫu bao nhiêu phút một lần?

**Kỳ vọng:** ESP32, DHT22, PMS5003, mỗi 5 phút.  
**Kiểm tra:** chunk retrieval đơn giản, metadata chính xác.

---

### Câu 2 — So sánh số liệu (Bài 1)

> So sánh chỉ số RMSE của mô hình LSTM với Linear Regression và Random Forest. Mô hình nào tốt nhất?

**Kỳ vọng:** LSTM RMSE = 2.45, Random Forest = 4.28, Linear Regression = 6.12 → LSTM tốt nhất.  
**Kiểm tra:** đọc và tổng hợp bảng số liệu.

---

### Câu 3 — Truy xuất công thức (Bài 2)

> Công thức tính Hybrid Score trong bài báo về RAG là gì? Giá trị alpha được chọn là bao nhiêu?

**Kỳ vọng:** `S_hybrid = α·S_vector + (1−α)·S_BM25`, với α = 0.7.  
**Kiểm tra:** retrieval đúng chunk chứa công thức toán học.

---

### Câu 4 — Kiến thức cross-paper (Bài 1 + Bài 3)

> Cả hai bài báo về IoT/LSTM và nhận diện bệnh lúa đều đề cập đến việc xử lý vấn đề gì của mạng nơ-ron truyền thống?

**Kỳ vọng:** Vanishing gradient (triệt tiêu đạo hàm) — LSTM dùng cổng gate, ResNet50 dùng shortcut connection.  
**Kiểm tra:** khả năng tổng hợp từ nhiều tài liệu.

---

### Câu 5 — Bẫy hallucination (Ngoài tài liệu)

> Blockchain được ứng dụng như thế nào trong nghiên cứu nhận diện bệnh lúa?

**Kỳ vọng:** "I don't know based on the provided documents." (không xuất hiện trong bất kỳ bài nào).  
**Kiểm tra:** strict mode không được hallucinate thông tin ngoài tài liệu.

---

## B. Kiểm tra chức năng History-aware

Mục tiêu: kiểm tra `contextualize()`, memory window, và khả năng chuyển chủ đề. **Gửi 5 câu này theo thứ tự liên tiếp trong cùng một cuộc hội thoại.**

### Câu 6 — Turn 1: Thiết lập ngữ cảnh (Bài 3)

> Bài báo về nhận diện bệnh lúa sử dụng kiến trúc mạng nào và đạt độ chính xác bao nhiêu?

**Kỳ vọng:** ResNet50, độ chính xác 96.5%.  
**Kiểm tra:** turn đầu tiên, thiết lập context cho các turn tiếp theo.

---

### Câu 7 — Turn 2: Đại từ follow-up (Bài 3)

> Tại sao họ lại chọn nó thay vì VGG16 hay MobileNetV2?

**Kỳ vọng:** "nó" được rewrite thành "ResNet50" → lý do là shortcut connections giải quyết vanishing gradient.  
**Kiểm tra:** `contextualize()` rewrite đại từ thành standalone query.

---

### Câu 8 — Turn 3: Tham chiếu ngầm (Bài 3)

> Mô hình đó được huấn luyện trên GPU gì và mất bao nhiêu epoch?

**Kỳ vọng:** NVIDIA RTX 4090, 50 epoch.  
**Kiểm tra:** memory window còn nhớ chủ đề sau 2 lượt hội thoại.

---

### Câu 9 — Turn 4: Chuyển chủ đề tường minh (Bài 1)

> Chuyển sang bài báo về IoT — giao thức truyền thông nào được dùng và tại sao?

**Kỳ vọng:** MQTT — nhẹ, ít băng thông, phù hợp thiết bị nhúng.  
**Kiểm tra:** context switch không bị nhầm lẫn giữa hai bài.

---

### Câu 10 — Turn 5: Tổng hợp đa lượt (Bài 1+2+3)

> Trong cả ba nghiên cứu chúng ta vừa thảo luận, nghiên cứu nào có dataset lớn nhất và nhỏ nhất?

**Kỳ vọng:** lớn nhất = IoT (8,640 mẫu), nhỏ nhất = RAG (1,000 câu hỏi), CNN = 5,000 ảnh.  
**Kiểm tra:** tổng hợp xuyên suốt toàn bộ hội thoại kết hợp retrieval.

---

## C. Benchmark RAG vs CoRAG

Mục tiêu: so sánh hiệu năng (độ chính xác, số vòng truy xuất, latency) giữa hai chế độ chain. Chạy từng câu ở cả hai mode và so sánh kết quả.

### Câu 11 — Câu đơn giản (Bài 2)

> Hit Rate@5 của phương pháp Hybrid + Re-ranking là bao nhiêu?

**Kỳ vọng:** 0.89.  
**Benchmark:** CoRAG không cần nhiều vòng → so sánh latency với RAG trên câu đơn.

---

### Câu 12 — Câu đa bước cross-paper (Bài 1+2+3)

> Hãy so sánh chiến lược giải quyết vấn đề vanishing gradient trong cả ba bài báo về (IOT, RAG, Nhận diện Bệnh hại trên Lá Lúa) và đánh giá phương pháp nào hiệu quả nhất theo kết quả thực nghiệm.

**Kỳ vọng:** LSTM dùng gate, ResNet50 dùng shortcut connection — RAG Hybrid không đề cập.  
**Benchmark:** CoRAG nên dùng 2–3 vòng; kiểm tra trace log có query đúng không.

---

### Câu 13 — Câu mơ hồ (Bài 1+2+3)

> Các bài báo này đề xuất hướng phát triển tiếp theo là gì?

**Kỳ vọng:** Edge Computing (Bài 1), GraphRAG (Bài 2), tích hợp Android/iOS + Drone (Bài 3).  
**Benchmark:** CoRAG nên dùng follow-up query để gom đủ context từ 3 bài.

---

### Câu 14 — Câu suy luận áp dụng (Bài 2)

> Nếu hệ thống RAG của bài báo thứ hai được áp dụng cho văn bản y tế thay vì pháp luật, những thành phần nào cần thay đổi và tại sao?

**Kỳ vọng:** thay embedding model (vietnamese-bi-encoder → medical domain), bộ dữ liệu training Re-ranking, tokenizer tách từ chuyên ngành.  
**Benchmark:** RAG có thể thiếu context; CoRAG nên lấy đủ rồi tổng hợp — so sánh chất lượng câu trả lời.

---

### Câu 15 — Bẫy ngoài tài liệu (Không có trong bài nào)

> Transformer architecture hoạt động như thế nào? Hãy giải thích chi tiết cơ chế self-attention.

**Kỳ vọng:** "I don't know based on the provided documents." trong strict mode.  
**Benchmark:** cả RAG và CoRAG đều phải fail gracefully — CoRAG không được dùng nhiều vòng để "tìm kiếm" thứ không tồn tại.

---

## Bảng tóm tắt

| # | Nhóm | Tài liệu | Mục tiêu kiểm tra |
|---|------|----------|-------------------|
| 1 | Cơ bản | Bài 1 | Chunk retrieval cơ bản |
| 2 | Cơ bản | Bài 1 | Đọc bảng số liệu |
| 3 | Cơ bản | Bài 2 | Retrieval công thức |
| 4 | Cơ bản | Bài 1+3 | Tổng hợp cross-paper |
| 5 | Cơ bản | — | Chống hallucination |
| 6 | History | Bài 3 | Turn đầu, thiết lập context |
| 7 | History | Bài 3 | Contextualize đại từ |
| 8 | History | Bài 3 | Memory window |
| 9 | History | Bài 1 | Context switch |
| 10 | History | Bài 1+2+3 | Tổng hợp đa lượt |
| 11 | Benchmark | Bài 2 | Câu đơn — latency |
| 12 | Benchmark | Bài 1+2+3 | Câu đa bước — CoRAG trace |
| 13 | Benchmark | Bài 1+2+3 | Câu mơ hồ — follow-up query |
| 14 | Benchmark | Bài 2 | Suy luận áp dụng |
| 15 | Benchmark | — | Fail gracefully ngoài tài liệu |