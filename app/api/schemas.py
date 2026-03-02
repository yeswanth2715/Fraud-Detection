from pydantic import BaseModel


class TransactionRequest(BaseModel):
    User: int
    Card: int
    Year: int
    Month: int
    Day: int
    Time: str
    Amount: str
    Use_Chip: str
    Merchant_Name: str
    Merchant_City: str
    Merchant_State: str
    Zip: int
    MCC: int
    Errors: str


class PredictionResponse(BaseModel):
    fraud_probability: float
    risk_level: str