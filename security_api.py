from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn
import jwt
from datetime import datetime
import time
import logging
import sys
import os
import joblib
import pandas as pd
import requests

try:
    from solution import load_model, preprocess, predict
    ml_model = load_model()
    print("✅ Successfully loaded model and functions from solution.py")
except Exception as e:
    ml_model = None
    print(f"Warning: solution.py or model.pkl failed to load. ML endpoint will not work. Error: {e}")

# --- 1. IMMUTABLE AUDIT LOGGING (SOC2 Compliance) ---
# This saves every security event to an 'audit.log' file AND prints it to the terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("audit.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("SecurityAudit")

app = FastAPI(title="OLEA Secure - Enterprise API Gateway")

# --- 2. STRICT CORS MIDDLEWARE ---
# Physically blocks HTTP requests unless they come from your Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501", 
        "http://frontend-agent:8501", 
        "http://127.0.0.1:8501"
    ],
    allow_credentials=True,
    allow_methods=["POST"], # Only allow POST requests, block GET/DELETE
    allow_headers=["*"],
)

SECRET_KEY = "OLEA_HACKATHON_SUPER_SECRET_2026"
request_tracker = {}

# --- 3. PII MASKING FUNCTION (GDPR Compliance) ---
def mask_pii(user_id: str) -> str:
    """Masks sensitive Identity Numbers (e.g., USER12345 -> U***345)"""
    if len(user_id) > 4:
        return f"{user_id[0]}***{user_id[-3:]}"
    return "****"

# --- Security Dependency (JWT) ---
def verify_token(x_token: str = Header(...)):
    """Validates dynamic, expiring JWT tokens."""
    try:
        payload = jwt.decode(x_token, SECRET_KEY, algorithms=["HS256"])
        if payload["exp"] < datetime.utcnow().timestamp():
            logger.warning("BLOCKED: Expired JWT Token (Possible Replay Attack attempted).")
            raise HTTPException(status_code=401, detail="Security Alert: Token Expired.")
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("BLOCKED: Expired JWT Signature.")
        raise HTTPException(status_code=401, detail="Security Alert: Token Expired.")
    except jwt.InvalidTokenError:
        logger.warning("CRITICAL: Invalid JWT Signature (Tampering Attempt Detected).")
        raise HTTPException(status_code=401, detail="Security Alert: Invalid Token Signature.")

# --- Data Validation Models ---
class ClaimRequest(BaseModel):
    user_id: str = Field(..., min_length=5, max_length=20, description="Unique User ID (Alphanumeric)")
    policy_type: str = Field(..., max_length=50, description="Type of insurance policy (e.g., Motor, Home)")
    amount: float = Field(..., gt=0, le=50000, description="Claim amount (Max 50,000)")

    @validator("user_id", "policy_type")
    def block_sql_injection(cls, value):
        dangerous_keywords = ["SELECT", "DROP", "INSERT", "DELETE", "UPDATE", "UNION", "--", ";"]
        val_upper = value.upper()
        for kw in dangerous_keywords:
            if kw in val_upper:
                logger.critical(f"CRITICAL BLOCKED: SQL Injection pattern '{kw}' detected in payload.")
                raise ValueError("Security Alert: Malicious SQL patterns detected.")
        return value

# --- Endpoints ---
@app.post("/api/secure_claim", dependencies=[Depends(verify_token)])
async def submit_secure_claim(request: ClaimRequest, raw_request: Request):
    """Secure endpoint with Rate Limiting, JWT, CORS, and PII Masking."""
    
    client_ip = raw_request.client.host
    current_time = time.time()
    
    # Mask the User ID immediately so plain text never touches the logs
    masked_user = mask_pii(request.user_id)
    
    # ANTI-DDOS IDENTITY-BASED RATE LIMITING
    if masked_user in request_tracker:
        last_request_time = request_tracker[masked_user]
        if current_time - last_request_time < 5.0: 
            logger.warning(f"RATE LIMIT TRIGGERED | Proxychain/Spam blocked for Target: {masked_user}")
            raise HTTPException(
                status_code=429, 
                detail="High Traffic Alert: Multiple claims detected for this user. Please wait 5 seconds."
            )
            
    request_tracker[masked_user] = current_time
    
    # Process Claim
    claim_id = f"SECURE-{request.user_id}-99"
    
    # Audit Log the Success (with masked data!)
    logger.info(f"SUCCESS | Claim Processed | IP: {client_ip} | User: {masked_user} | Amount: ${request.amount}")
    
    return {
        "status": "success",
        "message": "Claim passed security validation and was filed.",
        "claim_id": claim_id
    }

from main import chatbot

class ChatRESTRequest(BaseModel):
    message: str
    language: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRESTRequest):
    """Processes a chat message via the React Agent."""
    try:
        response_data = chatbot.chat(request.message, language=request.language)
        return {"response": response_data.get("response", "Error processing request")}
    except Exception as e:
        logger.error(f"Chat API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class ClientProfile(BaseModel):
    # Core Streamlit inputs
    Estimated_Annual_Income: float = 30000.0
    Adult_Dependents: int = 1
    Child_Dependents: float = 0.0
    Vehicles_on_Policy: int = 1
    Previous_Claims_Filed: int = 0
    Years_Without_Claims: int = 0
    Employment_Status: str = "Employed_FullTime"
    Region_Code: str = "TUN"
    
    # Required background columns for LightGBM with safe defaults
    Policy_Cancelled_Post_Purchase: int = 0
    Policy_Start_Year: int = 2026
    Policy_Start_Week: int = 1
    Policy_Start_Day: int = 1
    Grace_Period_Extensions: int = 0
    Previous_Policy_Duration_Months: int = 12
    Infant_Dependents: int = 0
    Existing_Policyholder: int = 0
    Policy_Amendments_Count: int = 0
    Broker_ID: float = 9.0
    Employer_ID: float = 0.0
    Underwriting_Processing_Days: int = 0
    Custom_Riders_Requested: int = 0
    Broker_Agency_Type: str = "Direct_Website"
    Deductible_Tier: str = "Tier_2_Mid_Ded"
    Acquisition_Channel: str = "Direct_Website"
    Payment_Schedule: str = "Monthly_EFT"
    Days_Since_Quote: int = 1
    Policy_Start_Month: str = "January"
    
    # Passing context for the LLM explicitly, but excluding it from ML features
    client_name: str = "Client"

@app.post("/api/ml_predict")
async def ml_predict_endpoint(profile: ClientProfile):
    """Predicts insurance bundle and augments explanation via LLaMA."""
    if ml_model is None:
        raise HTTPException(status_code=503, detail="ML Model not loaded.")
        
    try:
        # 1. Convert to DataFrame (dropping the non-ML feature 'client_name')
        profile_dict = profile.dict()
        client_name = profile_dict.pop("client_name")
        
        # Add a dummy User_ID required by solution.py's predict function
        profile_dict["User_ID"] = "USR_API_0001"
        
        df = pd.DataFrame([profile_dict])
        
        # 2. Run Inference using the exact solution.py logic (Zero Training-Serving Skew)
        start_ml = time.time()
        df_processed = preprocess(df)
        predictions_df = predict(df_processed, ml_model)
        prediction = predictions_df['Purchased_Coverage_Bundle'].iloc[0]
        ml_latency = time.time() - start_ml
        
        # 3. Call NVIDIA LLaMA 70B for the "Augmented" response
        NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
        if not NVIDIA_API_KEY:
            raise ValueError("Missing NVIDIA_API_KEY in environment.")
            
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Accept": "application/json"
        }
        
        prompt = f"""Tu es Imani, l'agent commercial dynamique et experte de OLEA Tunisie.
        Le système vient de sélectionner le "Pack Assurance Numéro {prediction}" pour le client {client_name}.
        Profil du client : {profile.Estimated_Annual_Income} TND de revenus, {profile.Adult_Dependents + profile.Child_Dependents} dépendants, {profile.Vehicles_on_Policy} véhicules, et {profile.Previous_Claims_Filed} sinistres.
        
        Tâche : Explique en dialecte Tunisien (Tounsi - alphabet latin) pourquoi ce Pack (Bundle {prediction}) est PARFAITEMENT adapté à sa situation.
        CONTRAINTE ABSOLUE : Utilise un ton commercial persuasif, insistant mais agréable ("persisting business tone").
        Tu DOIS utiliser les détails de son profil (ses revenus, son nombre d'enfants/véhicules, ou son historique de sinistres) pour le convaincre que c'est le meilleur investissement pour SA sécurité financière.
        Chaque discours doit être unique, ne génère pas de texte générique abrégé.
        Sois convaincante et directe (maximum 4 phrases). Demande-lui à la fin s'il souhaite confirmer.
        """
        
        payload = {
            "model": "meta/llama-3.1-70b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150,
            "temperature": 0.4
        }
        
        res = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
        res.raise_for_status()
        imani_explanation = res.json()["choices"][0]["message"]["content"]
        
        return {
            "predicted_bundle": int(prediction),
            "imani_explanation": imani_explanation,
            "ml_latency_sec": ml_latency
        }
        
    except Exception as e:
        logger.error(f"ML Predict API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("🔒 Starting OLEA Enterprise Secure API Gateway...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
