import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()   
CHROMA_DB_FOLDER_PATH=os.getenv("CHROMA_DB_FOLDER_PATH")
FAISS_DB_FOLDER_PATH=os.getenv("FAISS_DB_FOLDER_PATH")

def main():
    print("1. Đang đọc file PDF từ thư mục ./papers...")
    loader = DirectoryLoader(
        path="./papers",
        glob="**/*.pdf",
        loader_cls=UnstructuredFileLoader,
        show_progress=True,
        use_multithreading=True
    )
    doc = loader.load()

    MARKDOWN_SEPARATORS = [
        "\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n", "\n___+\n", "\n\n", "\n", " ", ""
    ]

    print("2. Đang chia nhỏ tài liệu (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, # kích thước tối đa mỗi chunk
        chunk_overlap=200, # số lượng kí tự chồng lặp giữa các chunk, giúp giữ ngữ cảnh liên tục
        add_start_index=True, # thêm metadata để dẫn nguồn truy xuất
        strip_whitespace=True, # loại bỏ khoảng trắng thừa
        separators=MARKDOWN_SEPARATORS # kí tự phân tách các chunk, ưu tiên theo index 0->n
    )
    splits = text_splitter.split_documents(doc)

    print("3. Đang khởi tạo Gemini Embedding API...")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    print("4. Đang tạo Vector Database và lưu xuống ổ cứng...")
    # Dùng from_documents để tạo MỚI database từ các splits
    # Tùy chọn các vector store
    # vectorstore = Chroma.from_documents(
    #     documents=splits, 
    #     embedding=embeddings,
    #     persist_directory="./chroma_db_gemini"
    # )
    vectorstore = FAISS.from_documents(
        documents=splits, 
        embedding=embeddings
    )
    vectorstore.save_local(FAISS_DB_FOLDER_PATH)

    print(f"✅ HOÀN TẤT! Dữ liệu đã được lưu vào thư mục {FAISS_DB_FOLDER_PATH}")

if __name__ == "__main__":
    main()