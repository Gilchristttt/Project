from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

Model = tf.keras.models.load_model("Model_1.keras")

class_names = ['Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_healthy']

@app.get("/ping")
async def ping():
    return "hello"


def read_f_as_arr(data) ->np.ndarray:
    im = np.array(Image.open(BytesIO(data)))
    return im

@app.post("/predict")
async def prediction(file: UploadFile = File(...)):
    
    image_np  = read_f_as_arr(await file.read())
    img_red = np.expand_dims(image_np, axis=0)
    predictions = Model.predict(img_red)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {"class ": predicted_class, "confidence":float(confidence) }

