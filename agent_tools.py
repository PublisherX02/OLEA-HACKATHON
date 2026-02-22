import random
from typing import List, Dict, Any
from langchain.tools import tool
import requests
import json
import jwt
import os
from datetime import datetime, timedelta

class InsuranceDatabase:
    """
    Simulated Insurance Database for tracking policies and claims.
    """
    def __init__(self):
        # Hardcoded dummy database
        self.users = {
            "USER123": {"policy_type": "Motor", "status": "Active", "coverage": 50000},
            "USER456": {"policy_type": "Home", "status": "Active", "coverage": 100000},
            "USER789": {"policy_type": "Health", "status": "Expired", "coverage": 0}
        }
        self.claims = {}

    def file_claim(self, user_id: str, policy_type: str, amount: float) -> str:
        """
        Simulates filing an insurance claim internally (Legacy/Fallback).
        Now primarily used by the Secure API, not directly by the tool.
        """
        if user_id not in self.users:
            return f"❌ User {user_id} not found in database."
            
        if self.users[user_id]["policy_type"] != policy_type:
            return f"❌ User {user_id} does not have a {policy_type} policy."

        # claims > 5000 require human review
        if amount > 5000:
            return f"⚠️ Claim amount {amount} exceeds automatic approval limit. Sent for human review. Reference: REV-{random.randint(1000, 9999)}"
        
        # Generate fake claim ID
        claim_id = f"CLM-{random.randint(10000, 99999)}"
        self.claims[claim_id] = {
            "user_id": user_id,
            "amount": amount,
            "status": "Approved"
        }
        return f"✅ Claim filed successfully! Your Claim ID is {claim_id}."

    def check_policy(self, user_id: str) -> str:
        """
        Checks the status of a user's insurance policy.
        
        Args:
            user_id (str): The ID of the user.
            
        Returns:
            str: Policy details or error message.
        """
        user = self.users.get(user_id)
        if not user:
            return f"❌ User {user_id} not found."
        
        return f"📋 Policy Details for {user_id}:\n- Type: {user['policy_type']}\n- Status: {user['status']}\n- Coverage: ${user['coverage']}"

# Instantiate Database
db = InsuranceDatabase()

@tool
def file_claim_tool(user_id: str, policy_type: str, amount: float) -> str:
    """
    Use this tool to file an insurance claim for a user.
    Requires user_id (str), policy_type (str), and amount (float).
    Returns the claim status and ID.
    """
    api_url = os.getenv("API_URL", "http://localhost:8000/api/secure_claim")
    
    # Generate a dynamic JWT token that expires in 60 seconds
    secret_key = "OLEA_HACKATHON_SUPER_SECRET_2026"
    token_payload = {
        "service": "imani_autonomous_agent",
        "exp": datetime.utcnow() + timedelta(seconds=60)
    }
    dynamic_token = jwt.encode(token_payload, secret_key, algorithm="HS256")
    
    headers = {
        "Content-Type": "application/json",
        "X-Token": dynamic_token
    }
    
    payload = {
        "user_id": user_id,
        "policy_type": policy_type,
        "amount": amount
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            response_data = response.json()
            return f"✅ {response_data['message']} Claim ID: {response_data['claim_id']}"
            
        elif response.status_code == 401:
            return "🚨 Security System Blocked Request: Unauthorized Access (Invalid or Expired JWT Token)."
            
        elif response.status_code == 422:
            return f"🚨 Security System Blocked Request: Validation Error (Anti-SQL Injection or Limit Exceeded)."
            
        elif response.status_code == 429:
            return "⏳ High Traffic Alert: The OLEA servers are currently experiencing heavy load. Please wait a few moments and try your claim again."
            
        else:
            return f"❌ Error filing claim. Server returned status {response.status_code}."
            
    except requests.exceptions.ConnectionError:
        return "❌ Error: Could not connect to the Secure API Gateway. Is the server running?"
    except Exception as e:
        return f"❌ Unexpected Error: {str(e)}"

@tool
def check_policy_tool(user_id: str) -> str:
    """
    Use this tool to check the status and details of a user's insurance policy.
    Requires user_id (str).
    Returns policy information.
    """
    return db.check_policy(user_id)

@tool
def predict_insurance_bundle_tool(input_json: str) -> str:
    """
    Use this tool ONLY when you have collected ALL the required profiling information from the user.
    The input MUST be a JSON string with these exact keys:
    {"income": 35000.0, "adult_dep": 1, "child_dep": 2, "vehicles": 1, "claims": 0, "user_name": "Ahmed"}
    - income: float (Revenu Annuel Estimé en TND)
    - adult_dep: int (nombre d'adultes à charge)
    - child_dep: int (nombre d'enfants à charge)
    - vehicles: int (nombre de véhicules)
    - claims: int (nombre de sinistres précédents)
    - user_name: str (prénom du client)
    Returns the personalized sales pitch to deliver to the user.
    """
    try:
        data = json.loads(input_json)
        income = float(data["income"])
        adult_dep = int(data["adult_dep"])
        child_dep = int(data["child_dep"])
        vehicles = int(data["vehicles"])
        claims = int(data["claims"])
        user_name = str(data["user_name"])
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return f"❌ Paramètres manquants ou incorrects. Assure-toi de passer un JSON valide avec les clés: income, adult_dep, child_dep, vehicles, claims, user_name. Détail: {str(e)}"

    api_url = os.getenv("API_URL", "http://localhost:8000")
    
    payload = {
        "Estimated_Annual_Income": income,
        "Adult_Dependents": adult_dep,
        "Child_Dependents": float(child_dep),
        "Vehicles_on_Policy": vehicles,
        "Previous_Claims_Filed": claims,
        "client_name": user_name
    }
    
    try:
        response = requests.post(f"{api_url}/api/ml_predict", json=payload, timeout=60)
        
        if response.status_code == 200:
            data_resp = response.json()
            return data_resp.get("imani_explanation", "Erreur lors de la génération du pitch.")
        else:
            return f"❌ Erreur ML backend. Status {response.status_code}: {response.text}"
            
    except Exception as e:
        return f"❌ Erreur de connexion au backend: {str(e)}"

@tool
def book_olea_appointment_tool(user_name: str) -> str:
    """
    Use this tool when the user explicitly agrees to subscribe to the proposed insurance bundle.
    Pass only the client's first name (user_name) as a plain string.
    Returns a confirmation that a form was sent to OLEA to book an appointment.
    """
    return f"✅ Parfait ! Rendez-vous confirmé pour {user_name}. Un formulaire de souscription automatique vient d'être envoyé aux services OLEA. Tu n'as plus besoin de te déplacer pour la paperasse — OLEA s'occupe de tout pour toi !"

# Export list of tools
insurance_tools = [file_claim_tool, check_policy_tool, predict_insurance_bundle_tool, book_olea_appointment_tool]
