import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load YOLOv8s-seg model
model = YOLO('best (2).pt')

st.title("YOLOv8 Segmentation with Streamlit")

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

    # Perform inference
    results = model(image)

    # Retrieve and display the segmentation mask
    result_image = results[0].plot()  # The function may vary depending on ultralytics version
    st.image(result_image, caption='Model Output Image', use_column_width=True)

st.write("Upload an image to see the YOLOv8s-seg model in action!")
