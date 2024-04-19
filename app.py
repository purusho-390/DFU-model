import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

np.set_printoptions(suppress=True)

# Load the model
model = load_model("/content/drive/MyDrive/Project/Models/keras_model.h5", compile=False)

# Load the labels
class_names = open("/content/drive/MyDrive/Project/Models/labels.txt", "r").readlines()

st.title("DFU Infection Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    st.write(f"Prediction: {class_name[2:]}")
    st.write(f"Confidence Score: {confidence_score}")
