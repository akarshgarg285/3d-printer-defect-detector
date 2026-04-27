import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="3D Print Crack Detector",
    page_icon="🔍",
    layout="centered"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_value=True)

st.title("🔍 3D Printer Defect Detection")
st.write("### Project by Akarsh Garg & Aakrit Jain")
st.write("Upload a surface photo to check for structural cracks using our trained CNN model.")

# 2. Model Loading
# We use st.cache_resource so the model only loads once, making the app fast.
@st.cache_resource
def load_my_model():
    try:
        # This looks for the file in your GitHub root folder
        return tf.keras.models.load_model("crack_detector_model.keras")
    except Exception as e:
        st.error(f"Error: Model file 'crack_detector_model.keras' not found in repository.")
        return None

model = load_my_model()
class_names = ['Negative', 'Positive'] # Negative: No Crack | Positive: Crack

# 3. File Uploader
uploaded_file = st.file_uploader("Upload a JPG/PNG image of the print", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Print Surface', use_container_width=True)
    
    # 4. Preprocessing (matching your 128x128 training)
    img_height, img_width = 128, 128
    img_resized = image.resize((img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0) # Create batch axis

    # 5. Prediction Logic
    if st.button('Start AI Inspection'):
        if model is not None:
            with st.spinner('Analyzing patterns...'):
                predictions = model.predict(img_array)
                # Apply softmax to get probability scores
                score = tf.nn.softmax(predictions[0])
                
                result = class_names[np.argmax(score)]
                confidence = 100 * np.max(score)

                st.write("---")
                if result == "Positive":
                    st.error(f"## 🚨 Crack Detected!")
                    st.write(f"**Confidence Score:** {confidence:.2f}%")
                    st.warning("Action Recommended: Inspect the printer's bed leveling and extrusion temperature.")
                else:
                    st.success(f"## ✅ Surface Clear")
                    st.write(f"**Confidence Score:** {confidence:.2f}%")
                    st.info("The AI did not find significant structural defects in this sample.")
        else:
            st.error("Model not loaded. Check GitHub repository for .keras file.")

# Footer
st.markdown("---")
st.caption("Developed for BITS Pilani, Hyderabad Campus - 2026")