import streamlit as st
from main import chatbot

# Page Configuration
st.set_page_config(
    page_title="Imani - Insurance Guide",
    page_icon="🟢",
    layout="centered"
)

# Custom CSS for WhatsApp-like styling (Dark-Mode Proof)
st.markdown("""
<style>
    /* WhatsApp Web Background Pattern */
    .stApp {
        background-color: #efeae2 !important;
        background-image: url("https://user-images.githubusercontent.com/15075759/28719144-86dc0f70-73b1-11e7-911d-60d70fcded21.png") !important;
        background-repeat: repeat !important;
        background-blend-mode: multiply;
    }
    
    /* Hide Streamlit Header */
    header {visibility: hidden;}
    
    /* Make all chat text Dark/Black */
    [data-testid="stChatMessage"] {
        color: #111111 !important;
    }
    [data-testid="stChatMessage"] * {
        color: #111111 !important;
    }

    /* Style the chat bubbles */
    [data-testid="stChatMessage"] {
        padding: 8px 12px !important;
        border-radius: 7.5px !important;
        margin-bottom: 8px !important;
        box-shadow: 0 1px 0.5px rgba(11,20,26,.13) !important;
        max-width: 65% !important;
        clear: both;
        display: inline-block;
        width: max-content;
    }
    
    /* User Message Bubble (Green, pushed to right) */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #dcf8c6 !important;
        border-top-right-radius: 0px !important;
        float: right;
        margin-left: auto !important;
    }
    
    /* Assistant Message Bubble (White, pushed to left) */
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: #ffffff !important;
        border-top-left-radius: 0px !important;
        float: left;
        margin-right: auto !important;
    }
    
    /* Fix Input Box to look like WhatsApp bottom bar */
    .stChatInputContainer {
        background-color: #f0f2f5 !important;
        padding: 10px !important;
        border-radius: 24px !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Settings
with st.sidebar:
    st.title("🌍 Settings")
    
    # Dialect Selection
    selected_language = st.selectbox(
        "Select Dialect",
        options=[
            "Tunisian Arabic (Tounsi)",
            "Algerian (Dziri)",
            "Moroccan (Darija)",
            "English (Standard)"
        ],
        index=0
    )
    
    # Clear Chat Button
    if st.button("Clear Chat", type="primary"):
        chatbot.clear_history()
        st.session_state.messages = []
        st.rerun()

# Header
st.title("Imani 🤖")
st.markdown(f"**Current Dialect:** {selected_language}")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Handling
if prompt := st.chat_input("Ask Imani..."):
    # Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add to History
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate Response
    with st.spinner("Imani is typing..."):
        try:
            # Call Backend
            response_data = chatbot.chat(prompt, language=selected_language)
            bot_response = response_data.get("response", "An error occurred.")
            
            # Display Bot Message
            with st.chat_message("assistant"):
                st.markdown(bot_response)
            
            # Add to History
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
