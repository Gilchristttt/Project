from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image


model = load_model("Model_1.keras")


app = FastAPI()


def preprocess_image(file, taille_image=256):
    img = Image.open(io.BytesIO(file)).convert("RGB")  
    img = img.resize((taille_image, taille_image))  
    img_array = np.array(img) / 255.0  
    return np.expand_dims(img_array, axis=0)  

@app.get("/")
def root():
    return {"message": "Bienvenue dans l'API de détection d'images !"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    content = await file.read()
    input_data = preprocess_image(content)


    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction[0])  

    
    class_names = ["Fake","Real"]
    result = class_names[predicted_class]

    return {
        "prediction": result,
        "confidence": float(prediction[0][predicted_class])
    }
