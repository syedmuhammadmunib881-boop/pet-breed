
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("pet_breed_model.h5")
    return model

model = load_model()

st.set_page_config(page_title="Pet Breed Recognition", page_icon="🐾", layout="centered")
st.title("🐾 Pet Breed Recognition")
st.write("Upload a pet image and I'll guess the breed!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    img_array = image.resize((224, 224))
    img_array = np.array(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    decoded = decode_predictions(prediction, top=1)[0][0]

    breed_name = decoded[1]
    confidence = decoded[2] * 100

    st.subheader(f"Prediction: {breed_name}")
    st.write(f"Confidence: {confidence:.2f}%")
