from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()   

loader = DirectoryLoader(
    path="./papers",
    glob="**/*.pdf",
    loader_cls=UnstructuredFileLoader,
    show_progress=True,
    use_multithreading=True # tăng tốc độ load nhiều file
)

doc = loader.load()

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n", 
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200, # kích thước tối đa mỗi chunk
    chunk_overlap=200, # số lượng kí tự chồng lặp giữa các chunk, giúp giữ ngữ cảnh liên tục
    add_start_index=True, # thêm metadata để dẫn nguồn truy xuất
    strip_whitespace=True, # loại bỏ khoảng trắng thừa
    separators=MARKDOWN_SEPARATORS # kí tự phân tách các chunk, ưu tiên theo index 0->n
)

splits = text_splitter.split_documents(doc)

# Tạo/Khai báo embedding model
print("Đang khởi tạo Gemini Embedding API...")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Tạo store
print("Đang tạo Vector Database ...")
# Bên dưới, store đẩy các split cho embedding model rồi lưu lại vector vào db
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings,
    persist_directory="./chroma_db_gemini"
    # distance_strategy=
)
print("Hoàn tất!")

# QUICK TEST: Tìm kiếm tương đồng (similarity search) với câu hỏi mẫu
# query = "Nội dung chính của tài liệu này là gì?"
# results = vectorstore.similarity_search(query, k=2) # short-hand thay vì phải tự tạo embedding query
# ----------------------------------------------------

retriver = vectorstore.as_retriever(
    # chỉ lấy những chunk có điểm tương đồng ở ngưỡng tự quy định
    # search_type="similarity_score_threshold",
    # seach_params={"k": 5}
)

template =(
    "You are a strict, citation-focused assistant for a private knowledge base.\n"
    "RULES:\n"
        "1) Use ONLY the provided context to answer.\n"
        "2) If the answer is not clearly contained in the context, say: "
        "\"I don't know based on the provided documents.\"\n"
        "3) Do NOT use outside knowledge, guessing, or web information.\n"
        "4) If applicable, cite sources as (source: page) using the metadata. \n\n"
    "Context: \n{context}\n\n"
    "Question: {question}"
)

promt = ChatPromptTemplate.from_template(template)
print("Đang khởi tạo Gemini LLM...")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    temperature=0.1
)

rag_chain = (
    {'context': retriver, "question": RunnablePassthrough()} # lấy parameter gắn vào question
    | promt
    | llm
    | StrOutputParser()
)

question=input("Question: ")
answer = rag_chain.invoke(question)

print(answer)