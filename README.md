# 🌿 Intelligent Plant Disease Detection with Preventive Guidance Using Deep Learning

## 📖 Overview

This project presents an AI-powered system designed to detect and diagnose plant diseases through image classification using a Convolutional Neural Network (CNN). In addition to detection, the system provides detailed treatment and prevention strategies through Google Gemini AI integration.

The motivation is to assist farmers and agricultural professionals by providing a smart, accessible, and real-time disease diagnosis tool, reducing crop loss and supporting sustainable agriculture practices.

---

## 🚀 How It Works

### 🔍 Step-by-Step Workflow:

1. **User selects a leaf image from the test dataset**.
2. The pre-trained CNN model classifies the disease from the image.
3. Gemini AI generates a human-readable explanation including:

   * Disease name
   * Causes
   * Symptoms
   * Prevention/Treatment tips

---

## 🧠 Technologies Used

| Tech              | Purpose                                                    |
| ----------------- | ---------------------------------------------------------- |
| Python            | Backend and ML logic                                       |
| Streamlit         | Frontend/UI for web interface                              |
| TensorFlow        | Deep learning CNN model                                    |
| Google Gemini API | Natural language generation (explanation of plant disease) |
| NumPy, Pillow     | Image handling                                             |

---

## 🧪 Dataset

* **Name:** New Plant Diseases Dataset (via Kaggle)
* **Size:** \~87,000 images
* **Classes:** 38 categories (includes healthy and diseased classes across multiple plant types)
* [Kaggle Link](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

---

## 🏗 Project Structure

```
├── app.py                       # Main Streamlit app
├── trained_plant_disease_model.h5  # CNN model (90MB)
├── test/                        # Sample test images for selection
│   ├── Apple_scab.jpg
│   ├── Tomato_early_blight.png
│   └── ...
├── media/                      # Thumbnail or UI images
│   └── thumbnail.png
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview and details
```

---

## 🌐 Deployment

Deployed using [Streamlit Cloud](https://streamlit.io/cloud). This platform allows free hosting of Streamlit applications. As uploading custom images isn't required, users can choose from sample test images directly.

---

## 💬 How Gemini Integration Works

After predicting the plant disease using the CNN model, the disease name is passed to Google Gemini’s LLM (Large Language Model). A descriptive explanation is generated and displayed in real-time, including symptoms and actionable treatment plans.

Example prompt sent to Gemini:

```python
Give a detailed explanation of the plant disease 'Tomato___Early_blight', including causes, symptoms, and treatment/prevention.
```

---

## 📚 References

* Dataset: [New Plant Diseases Dataset (Kaggle)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
* Model inspiration and flow: [YouTube Playlist](https://youtube.com/playlist?list=PLvz5lCwTgdXDNcXEVwwHsb9DwjNXZGsoy&si=le-PZGGW5a9VttCZ)

---

## 🔗 Final Notes

* No image upload needed — test images are preloaded for simplicity
* Model is deployed locally or on cloud (Streamlit) and can be extended for real-world deployment using APIs or mobile apps

---

> This project bridges AI technology with agriculture — making real-world impact on food production and sustainability. 🌱
