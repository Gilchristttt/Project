from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Charger le modèle et le scaler sauvegardés
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# API FastAPI
app = FastAPI()

# Schéma des données à envoyer
class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Endpoint pour la prédiction
@app.post("/predict/")
def predict(data: PatientData):
    # Convertir les données en tableau numpy
    patient = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure,
                         data.SkinThickness, data.Insulin, data.BMI,
                         data.DiabetesPedigreeFunction, data.Age]])
    # Standardiser
    patient_scaled = scaler.transform(patient)
    # Prédiction
    prediction = model.predict(patient_scaled)
    return {"diabetes_risk": int(prediction[0])}
