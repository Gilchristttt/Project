import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image


model = load_model("Model_1.keras")


def predict_image(img):
    img = img.resize((256, 256))  
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    prediction = model.predict(img_array)
    class_names = ["Fake","Real"]
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    if confidence < 0.49:
        return f"C'est une image générée par un Algorithme d'IA, avec une confiance de {confidence*100: .2f}% "
    elif confidence > 0.52:
        return f"C'est une image réelle, avec une confiance de {confidence*100:.2f}%  "
    else:
        return f"Plus ou moins une image rélle, avec une confiance de {confidence*100: .2f}%  "

interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Détection d'images générée par l'IA'"
)


interface.launch("share=True")
