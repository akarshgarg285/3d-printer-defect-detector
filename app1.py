import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="3D Print Crack Detector", page_icon="🔍")

st.title("🔍 3D Printer Defect Detection")
st.write("### Project ")

@st.cache_resource
def load_my_model():
    try:
        # Looks for the model in the same folder on GitHub
        return tf.keras.models.load_model("crack_detector_model.keras")
    except Exception as e:
        st.error(f"Model file not found in repository.")
        return None

model = load_my_model()
class_names = ['Negative', 'Positive']

uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Print Surface', use_container_width=True)
    
    # Preprocessing to match 128x128 training
    img_resized = image.resize((128, 128))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)

    if st.button('Start AI Inspection'):
        if model is not None:
            predictions = model.predict(img_array)
            # Use softmax for probability distribution
            score = tf.nn.softmax(predictions[0])
            result = class_names[np.argmax(score)]
            confidence = 100 * np.max(score)
            
            st.write(f"Result: **{result}** ({confidence:.2f}%)")
        else:
            st.error("Model not loaded.")

st.markdown("---")
st.caption("Developed by Akarsh Garg & Aakrit Jain ")
