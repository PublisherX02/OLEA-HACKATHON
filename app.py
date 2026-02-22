import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import io
import base64
import requests
import os

API_URL = os.environ.get("API_URL", "http://localhost:8000")

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

def correct_insurance_stt(raw_text):
    """
    Hackathon God-Mode : Intercepte les erreurs du Speech-to-Text Google
    et les force dans le vocabulaire de l'assurance tunisienne.
    """
    # Dictionnaire de correction : "Ce que Google entend" -> "Ce que le client voulait dire"
    corrections = {
        "kahraba": "karhba",        # Électricité -> Voiture
        "sayara": "karhba",         # Voiture (MSA) -> Voiture (Tounsi)
        "hadith": "accident",       # Histoire/Accident (MSA) -> Accident
        "hades": "accident",
        "aksidon": "accident",
        "taamin": "assurance",      # Assurance (MSA) -> Assurance
        "ta'min": "assurance",
        "zojaj": "bllar",           # Verre (MSA) -> Vitre/Pare-brise (Tounsi)
        "mraya": "rétroviseur",     # Miroir -> Rétroviseur
        "is3af": "dépannage",       # Secours -> Dépannage
        "moteur": "moteur",
        "parchoc": "parchoc",
        "dharba": "accident",       # Coup -> Accident
        "كهرباء": "كرهبة",          # Fixes Arabic Kahraba -> Karhba
        "كهراباء": "كرهبة",
        "حادث": "كسيدة"             # Optional: MSA to Tounsi
    }
    
    text_lower = raw_text.lower()
    
    # Remplacement ultra-rapide
    for wrong, right in corrections.items():
        text_lower = text_lower.replace(wrong, right)
        
    return text_lower


def transcribe_audio(audio_bytes, language_code="ar-TN"):
    """Converts spoken audio into text with noise reduction."""
    recognizer = sr.Recognizer()
    
    # ENHANCEMENT: Applied from Notebook for noisy hackathon environments
    recognizer.energy_threshold = 4000
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    
    audio_file = io.BytesIO(audio_bytes)
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            raw_text = recognizer.recognize_google(audio_data, language=language_code)
            
            # --- HACKATHON STT INTERCEPTOR (Dialect Fixes) ---
            final_text = correct_insurance_stt(raw_text)
                
            return final_text
            
    # ENHANCEMENT: Advanced error catching from notebook
    except sr.UnknownValueError:
        return "⚠️ Could not understand audio. Please speak clearly."
    except sr.RequestError as e:
        return f"⚠️ API Error (Check Wi-Fi): {e}"
    except Exception as e:
        return f"⚠️ Audio error: {str(e)}"

# --- UI CONFIGURATION ---
st.set_page_config(page_title="OLEA Service Client", page_icon="olea.png", layout="centered", initial_sidebar_state="expanded")

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
    #MainMenu {visibility: hidden;}
    header {background: transparent !important;}
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
olea_avatar = "olea.png"

# --- SIDEBAR (TOOLS & SETTINGS) ---
with st.sidebar:
    st.header("⚙️ Settings")
    selected_language = st.selectbox("Choose your Dialect:", ["Tunisian Arabic (Tounsi)", "Moroccan (Darija)", "Algerian (Dziri)", "English", "French"])
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.divider()
    
    st.header("📎 Attachments & Voice")
    # Feature 1: Voice Microphone (Required to keep)
    audio_bytes = st.audio_input("🎙️ Record Voice Note:")
    st.caption("💡 *Astuce : Essayez de parler en arabe le plus possible pour une meilleure reconnaissance.*")

# --- MAIN CHAT AREA ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Asslema ! Ena Imani, l'assistante commerciale mta3 OLEA. Bch najem na3tik a7sen Pack Assurance 'Ala Kifak', nst7a9 chwaya ma3loumet: chnouwa ismek, 9adeh dakhel fi l'3am (Revenu Annuel), 9adeh andek sghar w kbar fi kfaltek, 9adeh andek men karhba, w est-ce que 3malt des sinistres (accidents) 9bal ?"}]

# Render History
for message in st.session_state.messages:
    avatar_to_use = olea_avatar if message["role"] == "assistant" else user_avatar
    with st.chat_message(message["role"], avatar=avatar_to_use):
        st.markdown(message["content"])

# --- PROCESS SIDEBAR INPUTS ---
prompt = None

# --- PROCESS INPUTS ---
prompt = None
is_voice_prompt = False  # The flag to control the audio response!

# 1. Check for Voice Input
if audio_bytes:
    with st.spinner("Listening..."):
        stt_lang = "ar-TN" if "Tunisian" in selected_language else "ar-DZ" if "Algerian" in selected_language else "ar-MA" if "Moroccan" in selected_language else "en-US"
        prompt = transcribe_audio(audio_bytes.getvalue(), language_code=stt_lang)
        if "⚠️" in prompt:
            st.error(prompt)
            prompt = None
        else:
            is_voice_prompt = True

# 2. Check for Text Input (This overwrites voice if both happen)
text_input = st.chat_input("Message...", max_chars=500)
if text_input:
    prompt = text_input
    is_voice_prompt = False

# 3. Execute the Chat
if prompt:
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Imani is processing securely..."):
        try:
            payload = {"message": prompt, "language": selected_language}
            
            # Send to FastAPI Backend
            response = requests.post(f"{API_URL}/api/chat", json=payload, timeout=180)
            response.raise_for_status()
            bot_response = response.json().get("response", "No response generated.")
            
            # ONLY generate audio if the user used the microphone
            audio_html = ""
            if is_voice_prompt:
                tts_lang = 'ar' if 'Arabic' in selected_language or 'Dziri' in selected_language or 'Darija' in selected_language else 'en'
                audio_html = text_to_audio_autoplay(bot_response, lang=tts_lang)
            
            with st.chat_message("assistant", avatar=olea_avatar):
                st.markdown(bot_response)
                if audio_html:
                    st.markdown(audio_html, unsafe_allow_html=True)
            
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            
        except Exception as e:
            error_msg = f"❌ Network Error: Could not reach the AI Backend. Details: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
