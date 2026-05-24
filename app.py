import streamlit as st
from predict_remedy import predict_and_remedy
from PIL import Image
import tempfile

st.set_page_config(page_title="Crop Disease Detector", page_icon="🌱")

st.title("🌱 AI Crop Disease Detection & Recommendation System")

st.write("Upload a leaf image to detect disease and get treatment.")

uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save temp image
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    # Show image
    image = Image.open(temp_path)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict
    disease, confidence, remedy = predict_and_remedy(temp_path)

    st.subheader("Result")
    st.success(f"Disease: {disease}")
    st.info(f"Confidence: {confidence}%")
    st.warning(f"Remedy: {remedy}")