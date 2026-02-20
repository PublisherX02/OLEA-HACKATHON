from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field, validator
import uvicorn
import re

app = FastAPI(title="Imani Secure - Insurance API Gateway")

# --- Security Dependency ---
def verify_token(x_token: str = Header(...)):
    """
    Validates the 'x-token' header.
    If the token does not match the secure key, raises 401 Unauthorized.
    """
    if x_token != "Imani_Secure_2026":
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid Security Token")
    return x_token

# --- Data Validation Models ---
class ClaimRequest(BaseModel):
    user_id: str = Field(..., min_length=5, description="Unique User ID (Alphanumeric)")
    policy_type: str = Field(..., description="Type of insurance policy (e.g., Motor, Home)")
    amount: float = Field(..., gt=0, le=50000, description="Claim amount (Max 50,000)")

    @validator("user_id", "policy_type")
    def block_sql_injection(cls, value):
        dangerous_keywords = ["SELECT", "DROP", "INSERT", "DELETE", "UPDATE", "UNION", "--", ";"]
        val_upper = value.upper()
        for kw in dangerous_keywords:
            if kw in val_upper:
                raise ValueError("Security Alert: Malicious SQL patterns detected.")
        return value

# --- Endpoints ---
@app.post("/api/secure_claim", dependencies=[Depends(verify_token)])
async def submit_secure_claim(request: ClaimRequest):
    """
    Secure endpoint to submit an insurance claim.
    Requires:
    - Valid 'x-token' header.
    - Valid JSON payload complying with 'ClaimRequest' schema.
    """
    
    # Process the verified claim
    # In a real system, this would call the backend service or DB.
    # For now, we return the dummy success response.
    
    claim_id = f"SECURE-{request.user_id}-99"
    
    return {
        "status": "success",
        "message": "Claim passed security validation and was filed.",
        "claim_id": claim_id,
        "details": {
            "user": request.user_id,
            "type": request.policy_type,
            "amount": request.amount
        }
    }

# --- Runner ---
if __name__ == "__main__":
    print("🔒 Starting Imani Secure API Gateway...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
