import streamlit as st
from main import chatbot, analyze_damage_image
import speech_recognition as sr
from gtts import gTTS
import io
from audio_recorder_streamlit import audio_recorder
import base64

# --- VOICE ENGINE HELPER FUNCTIONS ---
def text_to_audio_autoplay(text, lang='ar'):
    """Converts text to speech and auto-plays it."""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        b64 = base64.b64encode(fp.read()).decode()
        md = f"""
            <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        return md
    except Exception as e:
        return ""

def transcribe_audio(audio_bytes, language_code="ar-TN"):
    """Converts spoken audio into text."""
    recognizer = sr.Recognizer()
    audio_file = io.BytesIO(audio_bytes)
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language=language_code)
            return text
    except Exception as e:
        return f"⚠️ Audio error: {str(e)}"

# --- UI CONFIGURATION ---
st.set_page_config(page_title="OLEA Service Client", page_icon="🟢", layout="centered", initial_sidebar_state="collapsed")

# Custom CSS for Authentic WhatsApp Web Styling
st.markdown("""
<style>
    .stApp {
        background-color: #efeae2 !important;
        background-image: url("https://user-images.githubusercontent.com/15075759/28719144-86dc0f70-73b1-11e7-911d-60d70fcded21.png") !important;
        background-repeat: repeat !important;
        background-blend-mode: multiply;
    }
    .block-container { padding-top: 0rem !important; padding-bottom: 5rem !important; }
    header {visibility: hidden;}
    [data-testid="stChatMessage"] { color: #111111 !important; }
    [data-testid="stChatMessage"] * { color: #111111 !important; }
    [data-testid="stChatMessage"][data-baseweb="block"]:nth-child(odd) {
        background-color: #ffffff !important; border-radius: 0px 8px 8px 8px; margin-bottom: 10px; padding: 10px; box-shadow: 0 1px 1px rgba(0,0,0,0.1);
    }
    [data-testid="stChatMessage"][data-baseweb="block"]:nth-child(even) {
        background-color: #d9fdd3 !important; border-radius: 8px 0px 8px 8px; margin-bottom: 10px; padding: 10px; box-shadow: 0 1px 1px rgba(0,0,0,0.1); display: flex; flex-direction: row-reverse;
    }
</style>
""", unsafe_allow_html=True)

user_avatar = "👤"
olea_avatar = "🟢"

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    selected_language = st.selectbox("Choose your Dialect:", ["Tunisian Arabic (Tounsi)", "Moroccan (Darija)", "Algerian (Dziri)", "English", "French"])
    if st.button("🗑️ Clear Chat History"):
        chatbot.clear_history()
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Asslema! Bienvenue chez OLEA. Kifech najem n3awnek lyoum?"}]

# Render History
for message in st.session_state.messages:
    avatar_to_use = olea_avatar if message["role"] == "assistant" else user_avatar
    with st.chat_message(message["role"], avatar=avatar_to_use):
        st.markdown(message["content"])

# --- FEATURE 1: IMAGE UPLOAD (VISION AI) ---
st.write("📸 **Upload Crash Photo:**")
uploaded_file = st.file_uploader("Upload Image (JPG, PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Crash Photo", use_column_width=True)
    if st.button("🔍 Run AI Damage Assessment & Fraud Check", use_container_width=True):
        with st.spinner("Analyzing pixels and deepfake anomalies..."):
            base64_img = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
            assessment = analyze_damage_image(base64_img, selected_language)
            
            # Generate Voice for Vision Assessment
            tts_lang = 'ar' if 'Arabic' in selected_language or 'Dziri' in selected_language or 'Darija' in selected_language else 'en'
            audio_html = text_to_audio_autoplay(assessment, lang=tts_lang)
            
            with st.chat_message("assistant", avatar=olea_avatar):
                st.markdown(assessment)
                if audio_html:
                    st.markdown(audio_html, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": assessment})

# --- FEATURE 2: VOICE MICROPHONE ---
st.write("🎙️ **Tap to Speak:**")
audio_bytes = audio_recorder(text="", recording_color="#e8b923", neutral_color="#008069", icon_size="2x")

prompt = None
if audio_bytes:
    with st.spinner("Listening..."):
        stt_lang = "ar-TN" if "Tunisian" in selected_language else "ar-DZ" if "Algerian" in selected_language else "ar-MA" if "Moroccan" in selected_language else "en-US"
        prompt = transcribe_audio(audio_bytes, language_code=stt_lang)
        if "⚠️" in prompt:
            st.error(prompt)
            prompt = None

# --- TEXT FALLBACK & RESPONSE GENERATION ---
text_input = st.chat_input("Message...", max_chars=500)
if text_input:
    prompt = text_input

if prompt:
    # 1. User Message
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Agent Response
    with st.spinner("OLEA is typing & thinking..."):
        try:
            response_data = chatbot.chat(prompt, language=selected_language)
            bot_response = response_data.get("response", "An error occurred.")
            
            # 3. Voice Generation
            tts_lang = 'ar' if 'Arabic' in selected_language or 'Dziri' in selected_language or 'Darija' in selected_language else 'en'
            audio_html = text_to_audio_autoplay(bot_response, lang=tts_lang)
            
            # 4. Render Bot Message + Audio
            with st.chat_message("assistant", avatar=olea_avatar):
                st.markdown(bot_response)
                if audio_html:
                    st.markdown(audio_html, unsafe_allow_html=True)
            
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
