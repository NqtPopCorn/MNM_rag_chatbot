import streamlit as st

def model_badge(provider: str, model: str):
    """Hiển thị badge provider + model đang dùng ở header chat với UI đẹp hơn."""
    color_map = {
        "gemini": ("🔵", "#4285F4"),  # Màu xanh Google
        "ollama": ("🟢", "#34A853"),  # Màu xanh lá
        "lmstudio": ("🟠", "#FBBC05"),
    }
    
    icon, border_color = color_map.get(provider, ("⚪", "#9AA0A6"))
    
    # CSS thẻ Badge để giao diện header nhìn sang trọng hơn
    html_str = f"""
    <div style="
        background-color: transparent; 
        padding: 12px 16px; 
        border-radius: 8px; 
        margin-bottom: 24px; 
        border-left: 5px solid {border_color};
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    ">
        <span style="font-size: 15px; color: #333;">
            {icon} Đang sử dụng <b>{provider.upper()}</b> với model: <code style="color: #FF4B4B;">{model}</code>
        </span>
    </div>
    """
    st.markdown(html_str, unsafe_allow_html=True)


def status_banner(db_exists: bool):
    if not db_exists:
        # Dùng st.warning với icon để báo hiệu rõ ràng hơn
        st.warning(
            "**Cơ sở dữ liệu (Database) chưa sẵn sàng!** 👋 \n\n"
            "Vui lòng tải lên tài liệu PDF của bạn ở thanh bên (sidebar) và nhấn 'Thêm vào Database' để bắt đầu.", 
            icon="🚨"
        )
        st.stop()