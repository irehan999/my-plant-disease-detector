import streamlit as st
import tensorflow as tf
import numpy as np
import os
import google.generativeai as genai

# ------------------- Configure the Gemini API -------------------
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY environment variable.")
    genai.configure(api_key=GEMINI_API_KEY)
    model_gemini = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"[Gemini API Error] {e}")
    model_gemini = None

def get_gemini_response(plant_disease):
    if model_gemini is None:
        return "❌ Gemini API is not configured."
    prompt = f"Give a detailed explanation of the plant disease '{plant_disease}', including causes, symptoms, and treatment/prevention."
    try:
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error getting Gemini response: {e}"

# ------------------- TensorFlow Model Prediction -------------------
def model_prediction(test_image):
    model_tf = tf.keras.models.load_model("trained_plant_disease_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Make batch of 1
    predictions = model_tf.predict(input_arr)
    return np.argmax(predictions)

# ------------------- Streamlit UI -------------------
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About"])

# ------------------- Home Page -------------------
if app_mode == "Home":
    st.header("🌿 Intelligent Plant Disease Detection with Preventive Guidance Using Deep Learning")
    st.image("media/thumbnail.png", use_container_width=True)
    st.markdown("""
    Welcome! This system identifies plant diseases from images using deep learning.

    ### How It Works:
    1. Select a plant image from the test folder **or upload your own image**.
    2. We'll predict the disease using a CNN model.
    3. You’ll get a detailed explanation powered by Google Gemini AI.
    """)

    # List images from test/images folder
    image_folder = "test/"
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    selected_image = st.selectbox("Select a test image:", image_files)

    uploaded_file = st.file_uploader("Or upload your own plant image (jpg, jpeg, png):", type=["jpg", "jpeg", "png"])

    image_path = None
    use_uploaded = False

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        use_uploaded = True
    elif selected_image:
        image_path = os.path.join(image_folder, selected_image)
        st.image(image_path, caption=selected_image, use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing... 🔍"):
            if use_uploaded:
                temp_path = "temp_uploaded_image.png"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                result_index = model_prediction(temp_path)
                os.remove(temp_path)
            elif image_path:
                result_index = model_prediction(image_path)
            else:
                st.warning("Please select or upload an image.")
                st.stop()

            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                          'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                          'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                          'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                          'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                          'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                          'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                          'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                          'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                          'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                          'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                          'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                          'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                          'Tomato___healthy']

            predicted_disease = class_name[result_index]
            st.success(f"🩺 Detected Disease: **{predicted_disease}**")

            disease_info = get_gemini_response(predicted_disease)
            st.write(disease_info)

# ------------------- About Page -------------------
elif app_mode == "About":
    st.title("About Our Project: Intelligent Plant Disease Detection with Preventive Guidance Using Deep Learning")

    st.subheader("🌱 Project Overview")
    st.markdown("""
    This project develops an AI system to detect plant diseases from leaf images using a Convolutional Neural Network (CNN). The system classifies diseases, provides real-time treatment advice, and offers a user interface for image uploads and results. It aims to aid farmers with accessible, AI-driven diagnostics and preventive strategies.
    """)

    st.subheader("🧠 Core Technology")
    st.markdown("""
    The core technologies of this project are a Convolutional Neural Network (CNN) for image-based disease detection and the Gemini API to provide detailed disease information and treatment advice. The CNN model, built with TensorFlow, analyzes plant leaf images, while Gemini enhances the system by generating explanations of diseases, their causes, symptoms, and treatments.
    """)

    st.subheader("📊 Dataset")
    st.markdown("""
    The CNN model is trained on the **New Plant Diseases Dataset**, sourced from [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset). This dataset contains images of healthy and diseased plant leaves.
    """)

    st.subheader("🔍 Disease Coverage")
    st.markdown("""
    The system recognizes 38 different categories of plant health, covering both healthy plants and those affected by common diseases. This wide coverage enables comprehensive diagnostic support for a range of plant conditions.
    """)

    st.subheader("💡 Enhanced Diagnostic Capabilities")
    st.markdown("""
    To enhance the diagnostic capabilities, the system integrates the Gemini AI. After identifying a disease, Gemini generates detailed information, including explanations, causes, symptoms, and treatment/prevention strategies. This bridges the gap between image detection and actionable advice.
    """)

    st.subheader("🎯 Project Goal")
    st.markdown("""
    Our goal is to bridge the gap between image-based disease detection and practical agricultural knowledge. We aim to offer a user-friendly AI solution that contributes to healthier crops, reduced losses, and sustainable farming practices.
    """)

    st.subheader("👨‍💻 Developed By")
    st.markdown("""
    **Rehan Irfan**  
    📧 [irehan046@gmail.com](mailto:irehan046@gmail.com)  
    """)

    

    st.subheader("📚 References")
    st.markdown("""
    - Took reference from: [YouTube Playlist](https://youtube.com/playlist?list=PLvz5lCwTgdXDNcXEVwwHsb9DwjNXZGsoy&si=le-PZGGW5a9VttCZ)
    """)
