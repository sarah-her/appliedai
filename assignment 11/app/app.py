from fastapi import FastAPI
from pydantic import BaseModel
from xgboost import XGBRegressor
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="Sales Forecast API")

# Load the trained XGBoost model
model = XGBRegressor()
model.load_model("model/model.json")

# Define input schema
class SalesInput(BaseModel):
    sales_lags: list[float]

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sales Forecast API!"}

# Prediction endpoint
@app.post("/predict")
def predict_sales(data: SalesInput):
    # Convert list to numpy array with correct shape (1, 6)
    X_input = np.array(data.sales_lags).reshape(1, -1)

    # Make prediction
    prediction = model.predict(X_input)[0]

    return {"predicted_sales_next_month": float(prediction)}
