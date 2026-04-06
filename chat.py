import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
CHROMA_DB_FOLDER_PATH=os.getenv("CHROMA_DB_FOLDER_PATH")
FAISS_DB_FOLDER_PATH=os.getenv("FAISS_DB_FOLDER_PATH")

def main():
    print("1. Đang kết nối với Vector Database đã lưu...")
    # Phải khai báo lại embedding để Chroma biết cách nhúng câu hỏi của user
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    
    # Load database ĐÃ TỒN TẠI 
    # Tùy chọn vector store
    # vectorstore = Chroma(
    #     persist_directory="./chroma_db_gemini", 
    #     embedding_function=embeddings
    # )
    vectorstore = FAISS.load_local(
        folder_path=FAISS_DB_FOLDER_PATH, 
        embeddings=embeddings,
        allow_dangerous_deserialization=True 
    )

    # Cấu hình Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.5}
    )

    # 3. Cấu hình Prompt
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
    prompt = ChatPromptTemplate.from_template(template)

    # 4. Khởi tạo LLM
    print("2. Đang khởi tạo Gemini LLM...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.1
    )

    # 5. Tạo Chain bằng LCEL (LangChain Expression Language)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n✅ HỆ THỐNG SẴN SÀNG! (Gõ 'exit' hoặc 'quit' để thoát)\n" + "-"*50)

    # Vòng lặp vô hạn giúp bạn hỏi liên tục mà không cần chạy lại script
    while True:
        question = input("\n🧑 Question: ")
        if question.lower() in ['exit', 'quit']:
            print("Tạm biệt!")
            break
            
        print("🤖 Trả lời: ", end="", flush=True)
        # Sử dụng stream để text hiện ra từ từ cho giống chatbot thật
        for chunk in rag_chain.stream(question):
            print(chunk, end="", flush=True)
        print("\n")

if __name__ == "__main__":
    main()