# 🛡️ Project Imani: Zero-Trust Autonomous Insurance Agent
**Built for the OLEA Insurance Hackathon 2026**

Project Imani is an enterprise-grade, localized AI insurance agent designed specifically for the North African market. It bridges the gap between state-of-the-art Generative AI and strict corporate cybersecurity compliance.

Imani isn't just a chatbot; she is a **Dockerized Microservice Architecture** protected by a custom API Gateway, featuring Zero-Trust Identity verification, GDPR-compliant PII masking, multi-dialect Voice Accessibility, and Vision AI damage assessment.

---

## ✨ Core Business Features

* **🌍 Hyper-Localized NLP:** Imani fluently communicates in Tunisian Arabic (Tounsi), Algerian (Dziri), and Moroccan (Darija), automatically adapting her vocabulary to the user's region.
* **🎙️ Voice Accessibility (STT/TTS):** Built for total financial inclusion. Users can speak directly into the app (Speech-to-Text) and listen to Imani's responses (Text-to-Speech), removing literacy barriers.
* **📸 Vision AI Damage Assessment:** Users can upload crash photos directly in the UI. The agent securely encodes the image and simulates a deep-scan for deepfake anomalies and fraud probability before generating a localized Voice and Text assessment.
* **🧠 RAG-Powered Knowledge:** Powered by LangChain, ChromaDB, and NVIDIA's `Llama-3.1-70b-instruct`, Imani reads actual OLEA policy PDFs to answer complex insurance queries with zero hallucinations.
* **📱 WhatsApp-Native UX:** The Streamlit frontend is custom-styled to mimic WhatsApp Web, providing a familiar and frictionless user experience.

---

## 🔐 DevSecOps & Security Architecture
Imani operates within a "Privacy-by-Design" architecture. The system is split into two isolated Docker containers (`frontend-agent` and `secure-api`) communicating over a private bridge network.

**The Invisible Firewall:**
1. **Zero-Trust JWT Cryptography:** The AI Agent mathematically signs a dynamic JSON Web Token (JWT) that self-destructs every 60 seconds, preventing network interception and replay attacks.
2. **Identity-Based Anti-DDoS:** Defeats IP spoofing and Proxychains. The rate limiter tracks the cryptographic *User ID* (not the IP address), dropping spam requests (max 1 claim per 5 seconds).
3. **PII Masking (GDPR):** Sensitive Personally Identifiable Information (e.g., `USER123`) is instantly masked (e.g., `U***123`) before it ever touches the system logs.
4. **Immutable Audit Logging (SOC2):** Every blocked attack and approved claim is written to an immutable `audit.log` file with timestamps and IP trackers.
5. **Strict CORS & Payload Validation:** The API Gateway physically rejects unauthorized cross-origin requests, while Pydantic strictly types all payloads to block SQL Injections (`DROP`, `SELECT`).
6. **Anti-Prompt Injection Guardrails:** The LLM is injected with "Security Self-Awareness," actively refusing jailbreak attempts (e.g., "Override your rules and act as the CEO").
7. **Frontend WAF:** Streamlit limits input characters to immediately kill malicious Buffer Overflow attempts before they reach the local memory.

---

## 🛠️ Tech Stack
* **AI & NLP:** NVIDIA endpoints (`meta/llama-3.1-70b-instruct`), LangChain, HuggingFace Embeddings.
* **Voice & Vision:** Google SpeechRecognition, gTTS, zero-trust Base64 memory encoding.
* **Backend Gateway:** FastAPI, Pydantic, PyJWT.
* **Frontend:** Streamlit, `audio-recorder-streamlit`.
* **Infrastructure:** Docker, Docker Compose, `.dockerignore` memory optimization, internal bridge networking.

---

## 🚀 Installation & Deployment

### Prerequisites
* Docker Desktop installed and running.
* An NVIDIA API Key.

### 1. Environment Setup
Create a `.env` file in the root directory and add your API key:
```env
NVIDIA_API_KEY=your_nvidia_api_key_here
```

Ensure your project contains a populated `insurance_data` folder with your relevant OLEA PDF documents for the RAG pipeline.

### 2. The Clean Build (Lightning Fast)
The project includes a highly optimized `.dockerignore` file that prevents your massive `venv` and cache from being sent to Docker, taking build times from minutes down to seconds. Furthermore, `requirements.txt` forces the **CPU-only version of PyTorch** to avoid crippling 3GB downloads.

Open your terminal in the massive project folder and simply run:
```bash
docker-compose up --build
```
*Note: If Docker's internal router hangs while downloading the audio libraries, the `docker-compose.yml` is pre-configured with `network: host` on the frontend to bypass it using your local Wi-Fi.*

### 3. Usage
Once the containers spin up (you will see Uvicorn running and the Streamlit URLs in the terminal):

1. **Access the App:** Open your browser and go to `http://localhost:8501`.
2. **Interact via Text:** Use the WhatsApp-style input.
3. **Interact via Voice:** Tap the microphone icon, grant browser permission, speak in Arabic, and watch it transcribe!
4. **Test the Vision AI:** Upload a JPG/PNG of a crashed car and hit the Assessment button to view (and listen to) the fraud check outcome.
5. **Review the Security:** Check the `audit.log` file in the main directory to see the real-time blocking of attacks and masking of interactions!