import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Conv2D , MaxPooling2D , Flatten , Activation , BatchNormalization , Dropout
from tensorflow.keras.optimizers import Adam , Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import load_model

# Load the trained EfficientNetB0 model
channels = 3
img_shape = (224, 224, channels)
class_count = 5

base_model = tf.keras.applications.DenseNet121(include_top= False, weights= "imagenet", input_shape= img_shape, pooling= 'max')

model = Sequential([
    base_model])

model=load_model('model.h5')

# Class labels
class_labels = ['Lilly', 'Lotus', 'Orchid', 'SunFlower', 'Tulip']

def predict(image):
    # Preprocess the image
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    predictions = model.predict(image_array)
    predicted_idx = np.argmax(predictions[0])
    return class_labels[predicted_idx]

st.title("Flower Classification with EfficientNetB0")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Convert image to RGB if it is RGBA
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Get the classification result
    prediction = predict(image)
    st.write(f"Prediction: {prediction}")

st.write("Upload an image to classify it into Lilly, Lotus, Orchid, SunFlower, or Tulip.")
