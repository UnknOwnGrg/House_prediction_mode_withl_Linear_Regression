from pydantic import BaseModel
from typing import Dict,Any

class PredictionRequest(BaseModel):
    total_sqft: float
    bath: int
    bhk: int
    location: str

class PredictionResponse(BaseModel):
    predicted_price: float
    formula_breakdown: Dict[str, Any]
    is_mock: bool