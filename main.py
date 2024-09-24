from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
# Load the trained model
model = pickle.load(open("physical_properties.pkl", "rb"))

# Load the training data to get the feature names (optional)
# train_df = pd.read_csv('aluminum_wire_properties_10000.csv')
# feature_names = train_df.columns.drop(['Tensile Strength (MPa)', 'Yield Strength (MPa)', 'Elongation (%)'])

# Initialize the FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input data model
class InputData(BaseModel):
    Al: float = Field(..., alias="Al (%)")
    Mg: float = Field(..., alias="Mg (%)")
    Si: float = Field(..., alias="Si (%)")
    Zn: float = Field(..., alias="Zn (%)")
    Temperature: float = Field(..., alias="Temperature (Â°C)")
    Rolling_Speed: float = Field(..., alias="Rolling Speed (m/s)")
    Heat_Treatment: int = Field(..., alias="Heat Treatment (Yes=1, No=0)")


# Prediction endpoint
import numpy as np


@app.post("/predict")
async def predict(input_data: InputData):
    input_values = [
        input_data.Al,
        input_data.Mg,
        input_data.Si,
        input_data.Zn,
        input_data.Temperature,
        input_data.Rolling_Speed,
        input_data.Heat_Treatment
    ]

    # Get the prediction
    prediction = model.predict([input_values])

    # Convert prediction to a list if it's an array
    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()

    # Ensure the prediction is a single scalar value (if applicable)
    predicted_value = float(prediction[0]) if isinstance(prediction[0], (np.float32, np.float64)) else prediction[0]

    return {"Predicted results": predicted_value}


# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to Aluminium Physical Properties Prediction API!"}
