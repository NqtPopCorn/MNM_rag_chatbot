import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load biến môi trường
load_dotenv()
FAISS_DB_FOLDER_PATH = os.getenv("FAISS_DB_FOLDER_PATH", "./faiss_db_gemini")
UPLOAD_DIR = "./papers"

# Đảm bảo thư mục lưu trữ tồn tại
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# --- CẤU HÌNH GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="Private Knowledge Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Chatbot RAG - Tải lên & Hỏi đáp")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- CÁC HÀM XỬ LÝ DỮ LIỆU ---

def add_files_to_db(file_paths, chunk_size, chunk_overlap):
    """CHỈ đọc file mới, cắt nhỏ và THÊM vào DB hiện tại"""
    try:
        # 1. Chỉ load các file vừa được upload
        documents = []
        for file_path in file_paths:
            loader = UnstructuredFileLoader(file_path)
            documents.extend(loader.load())

        if not documents:
            return False, "Không đọc được nội dung file PDF."

        # 2. Cắt nhỏ (Chunking) theo thông số UI
        MARKDOWN_SEPARATORS = ["\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n", "\n___+\n", "\n\n", "\n", " ", ""]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            strip_whitespace=True,
            separators=MARKDOWN_SEPARATORS
        )
        splits = text_splitter.split_documents(documents)

        # 3. Khởi tạo Embedding model
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

        # 4. Thêm vào DB (Append) thay vì tạo mới toàn bộ
        if os.path.exists(FAISS_DB_FOLDER_PATH):
            # Nếu đã có DB -> Load lên và thêm document mới vào
            vectorstore = FAISS.load_local(
                folder_path=FAISS_DB_FOLDER_PATH, 
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            vectorstore.add_documents(splits) # <-- Hàm quan trọng để append
        else:
            # Nếu chưa có DB -> Tạo mới
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        
        # 5. Lưu lại xuống ổ cứng
        vectorstore.save_local(FAISS_DB_FOLDER_PATH)
        
        # Xóa cache bộ nhớ để chatbot nhận diện dữ liệu mới ngay lập tức
        st.cache_resource.clear() 
        return True, f"Thành công! Đã nhúng và thêm {len(splits)} chunks mới vào Database."
    except Exception as e:
        return False, f"Lỗi hệ thống: {str(e)}"

def rebuild_all_db(chunk_size, chunk_overlap):
    """Đọc lại TOÀN BỘ thư mục ./papers và tạo lại DB từ đầu (Backup)"""
    try:
        loader = DirectoryLoader(
            path=UPLOAD_DIR, glob="**/*.pdf", loader_cls=UnstructuredFileLoader, use_multithreading=True
        )
        doc = loader.load()
        if not doc: return False, "Thư mục trống."

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            add_start_index=True, strip_whitespace=True
        )
        splits = text_splitter.split_documents(doc)
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        vectorstore.save_local(FAISS_DB_FOLDER_PATH)
        st.cache_resource.clear() 
        return True, f"Đã build lại TOÀN BỘ Database với {len(splits)} chunks."
    except Exception as e:
        return False, f"Lỗi: {str(e)}"

@st.cache_resource(show_spinner=False)
def get_rag_chain():
    if not os.path.exists(FAISS_DB_FOLDER_PATH):
        return None
    try:    
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        vectorstore = FAISS.load_local(
            folder_path=FAISS_DB_FOLDER_PATH, 
            embeddings=embeddings,
            allow_dangerous_deserialization=True 
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold", # tìm kiếm theo điểm tương đồng dựa trên phương pháp cosine
            search_kwargs={"k": 5, "score_threshold": 0.3} # lấy 5 vector có ngưỡng điểm lọc thấp nhất 0.3
        )
        template = (
            "You are a strict, citation-focused assistant for a private knowledge base.\n"
            "RULES:\n"
            "1) Use ONLY the provided context to answer.\n"
            "2) If the answer is not clearly contained in the context, say: "
            "\"I don't know based on the provided documents.\"\n"
            "3) Do NOT use outside knowledge, guessing, or web information.\n"
            "4) If applicable, cite sources as (source: page:start_index) using the metadata.\n"
            "5) Answer in the SAME language as the question. Do NOT translate unless the question asks for translation.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1)
        return ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    except:
        return None

# --- SIDEBAR: TẢI FILE & CONFIG CHUNK ---
with st.sidebar:
    st.header("📂 Thêm tài liệu mới")
    
    # 1. Giao diện upload
    uploaded_files = st.file_uploader("Chọn file PDF", type=["pdf"], accept_multiple_files=True)
    
    # 2. Cấu hình Chunk (Đặt ngay dưới file để config trước khi bấm thêm)
    st.subheader("⚙️ Cấu hình Chunk")
    c_size = st.slider("Chunk Size", 500, 3000, 1200, key='c_size')
    c_overlap = st.slider("Chunk Overlap", 0, 500, 200, key='c_overlap')
    
    # 3. Nút xử lý TỪNG FILE
    if uploaded_files:
        if st.button("🚀 Gắn file này vào Database", type="primary", use_container_width=True):
            with st.spinner("Đang xử lý và nhúng file mới..."):
                saved_paths = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_paths.append(file_path)
                
                # Gọi hàm CHỈ ADD FILE MỚI, dùng chunk config hiện tại
                success, message = add_files_to_db(saved_paths, c_size, c_overlap)
                if success: st.success(message)
                else: st.error(message)

    st.divider()
    
    # 4. Nút Re-build toàn bộ (Dành cho trường hợp muốn làm mới lại từ đầu)
    with st.expander("🛠️ Công cụ nâng cao"):
        st.warning("Nút bên dưới sẽ đọc lại TOÀN BỘ file trong thư mục và tính phí Embedding lại từ đầu.")
        if st.button("🔄 Build lại TOÀN BỘ Database"):
            with st.spinner("Đang build lại..."):
                success, message = rebuild_all_db(c_size, c_overlap)
                if success: st.success(message)
                else: st.error(message)

# --- MAIN GIAO DIỆN CHAT ---
rag_chain = get_rag_chain()

if rag_chain is None:
    st.info("👋 Database chưa được tạo. Hãy upload file PDF ở cột bên trái để bắt đầu!")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Hỏi gì đó về tài liệu..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = st.write_stream(rag_chain.stream(prompt))
            st.session_state.messages.append({"role": "assistant", "content": response})