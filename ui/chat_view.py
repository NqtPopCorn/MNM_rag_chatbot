import streamlit as st
from langchain_core.runnables import Runnable

def render_chat(rag_chain: Runnable, provider: str, model: str):
    """Render vùng chat: hiển thị history + xử lý câu hỏi mới, có kèm avatar và tên model."""
    
    # Hiển thị lịch sử
    for msg in st.session_state.messages:
        # Tùy biến avatar cho từng role
        avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
        
        with st.chat_message(msg["role"], avatar=avatar):
            # Nếu là tin nhắn của assistant và có lưu thông tin model
            if msg["role"] == "assistant" and "model_info" in msg:
                st.caption(f"⚡ **Trả lời bởi:** `{msg['model_info']}`")
            
            st.markdown(msg["content"])

    # Input người dùng
    if user_input := st.chat_input("Hỏi gì đó về tài liệu..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            model_info_str = f"{provider.upper()} - {model}"
            st.caption(f"⚡ **Trả lời bởi:** `{model_info_str}`")
            
            # Streaming kết quả
            response = st.write_stream(rag_chain.stream(user_input))
            
        # Lưu vào history kèm theo thông tin model
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "model_info": model_info_str
        })

def clear_chat_button():
    # Thêm style='secondary' hoặc icon để nút trông xịn hơn
    if st.button("🗑️ Làm mới cuộc trò chuyện", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()