# 🤖 RAG Chatbot

Chatbot hỏi đáp tài liệu PDF với khả năng **swap linh hoạt** giữa Gemini API và local models (Ollama, LM Studio).

---

## Cấu trúc thư mục

```
rag-chatbot/
├── core/               # Logic nghiệp vụ (LLM factory, embedding, chain, vectorstore)
├── config/             # Settings (Pydantic) + Prompt templates
├── models/             # Wrapper từng provider: gemini, ollama, lmstudio
├── ui/                 # Streamlit components: sidebar, chat, badges
├── papers/             # File PDF đầu vào
├── vector_db/          # FAISS index được lưu ở đây
├── app.py              # Entry point Streamlit
├── build_db.py         # CLI: build DB offline
├── chat.py             # CLI: chat trong terminal
└── requirements.txt
```

---

## Cài đặt

```bash
pip install -r requirements.txt
cp .env.example .env
# Điền GOOGLE_API_KEY vào .env nếu dùng Gemini
```

---

## Chạy ứng dụng

### Streamlit UI
```bash
streamlit run app.py
```

### Build DB từ CLI
```bash
# Dùng Gemini embedding (mặc định)
python build_db.py

# Dùng Ollama embedding
python build_db.py --embed-provider ollama --embed-model nomic-embed-text

# Tùy chỉnh chunk
python build_db.py --chunk-size 1000 --chunk-overlap 150
```

### Chat từ terminal
```bash
# Gemini
python chat.py

# Ollama
python chat.py --provider ollama --model llama3.2

# LM Studio
python chat.py --provider lmstudio --model your-model-name
```

---

## Swap Model

### Trong UI (Streamlit)
Dùng sidebar → chọn **Provider** và **Model** → chain tự build lại.

### Thêm provider mới vào code
1. Tạo `models/<provider>.py` với hàm `get_<provider>_llm()` và `get_<provider>_embeddings()`
2. Thêm 1 `elif` vào `core/llm.py` và `core/embeddings.py`
3. Thêm provider vào danh sách `LLM_PROVIDERS` trong `ui/sidebar.py`

---

## LƯU Ý quan trọng

> **Embedding model phải nhất quán** giữa lúc build DB và lúc query.  
> Nếu đổi embedding provider/model → bấm **"Build lại TOÀN BỘ Database"** trong sidebar.