import os
import json
from fastapi import FastAPI , HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib as jb
from .schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="Real Estate Predictor - Initialized Model Server")

# Configure CORS for React Frontend on port 5174
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
metadata = None

ARTIFACTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "model_metadata.json")

def load_model_artifacts():
    global model ,metadata
    if os.path.exists(MODEL_PATH) and os.path.exists(METADATA_PATH):
        try:
            model = jb.load(MODEL_PATH)
            with open(METADATA_PATH, "r") as f:
                metadata= json.load(f)
            print("Successfully loaded model and metadata from artifacts!")

        except Exception as e:
            print("Warning: Model artifacts not found.Serving mock predictions.")
            model, metadata = None, None

@app.on_event('startup')
def startup_event():
    load_model_artifacts()

@app.get("/api/model-info")
def get_model_info():
    if model is not None and metadata is not None:
        return {
            "status": "loaded",
            "metrics": metadata.get("metrics", {}),
            "locations": metadata.get("locations", []),
            "features_count": len(metadata.get("features", [])),
            "intercept": metadata.get("intercept", 0.0),
            "coefficients": metadata.get("coefficients", {}),
            "is_mock": False
        }
    else:
#         Fallback if model isn't trained yet
        return {
            "status": "not_trained",
            "message": "Model artifacts not found.Run train.py to train and export the model.",
            "locations":  ["Indiranagar", "1st Phase JP Nagar", "Whitefield", "Yelahanka", "Electronic City", "Thanisandra", "Kanakpura Road", "Sarjapur Road"],
            "is_mock": True
        }

@app.post("/api/predict")
def predict(request: PredictionRequest):
    global model, metadata

    # Reload model if it was trained after startup
    if model is None:
        load_model_artifacts()

    if model is not None and metadata is not None:
        try:
            features = metadata["features"]
            x = np.zeros(len(features))

            # Set continuous features
            x[0] = request.total_sqft
            x[1] = request.bath
            x[2] = request.bhk

            # One-hot encode the location if it exists
            #
            loc_column = request.location
            loc_coeff = 0.0

            if loc_column in features:
                loc_idx = features.index(loc_column)
                x[loc_idx] = 1.0
                loc_coeff = metadata["coefficients"].get(loc_column, 0.0)

            # Perform prediction
            prediction = model.predict([x])[0]

            # Avoid negatice prediction
            predicted_price = max(0.0, float(prediction))

            # Calculate breakdown contributions
            intercept = metadata.get("intercept", 0.0)
            sqft_coeff = metadata["coefficients"].get("total_sqft", 0.0)
            bath_coeff = metadata["coefficients"].get("bath", 0.0)
            bhk_coeff = metadata["coefficients"].get("bhk", 0.0)

            formula_breakdown = {
                "intercept": intercept,
                "sqft_contribution": request.total_sqft * sqft_coeff,
                "bath_contribution": request.bath * bath_coeff,
                "bhk_contribution": request.bhk * bhk_coeff,
                "location_contribution": loc_coeff,
                "sqft_coeff": sqft_coeff,
                "bath_coeff": bath_coeff,
                "bhk_coeff": bhk_coeff,
                "location_coeff": loc_coeff
            }

            return PredictionResponse (
                predicted_price = predicted_price,
                formula_breakdown=formula_breakdown,
                is_mock=False
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction errorr : {str(e)}")
    else:
        # Serve dummy prediction in a realistic range if model artifacts are not present
        # Price in lakhs: e.g. 1 BHK in Electronic City is 35 lakhs, 3 BHK in Indiranagar is 200 lakhs
        # A simple mock linear formula: Price = 10 + sqft * 0.05 + bhk * 15 + bath * 5 + location_factor
        loc_factors = {
            "Indiranagar": 80.0,
            "1st Phase JP Nagar": 45.0,
            "Whitefield": 20.0,
            "Yelahanka": 15.0,
            "Electronic City": -5.0,
            "Thanisandra": 0.0,
            "Kanakpura Road": 5.0,
            "Sarjapur Road": 10.0
        }
        loc_factor = loc_factors.get(request.location, 0.0)
        predicted_price = 10.0 + (request.total_sqft * 0.06) + (request.bhk * 10.0) + (request.bath * 8.0) + loc_factor
        predicted_price = max(10.0 , predicted_price)

        formula_breakdown = {
            "intercept": 10.0,
            "sqft_contribution": request.total_sqft * 0.06,
            "bath_contribution": request.bath * 8.0,
            "bhk_contribution": request.bhk * 10.0,
            "location_contribution": loc_factor,
            "sqft_coeff": 0.06,
            "bath_coeff": 8.0,
            "bhk_coeff": 10.0,
            "location_coeff": loc_factor
        }

        return PredictionResponse(
            predicted_price= float(predicted_price),
            formula_breakdown=formula_breakdown,
            is_mock=True
        )


if __name__ == "__main__" :
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)