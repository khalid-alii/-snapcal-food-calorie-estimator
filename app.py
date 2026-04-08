import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

st.set_page_config(
    page_title="Food Calorie Estimator",
    page_icon="🍽️",
    layout="centered"
)

NUTRITION_DATA = {
    "pizza":        {"calories": 266, "protein": 11.0, "carbs": 33.0, "fat": 10.0},
    "hamburger":    {"calories": 295, "protein": 17.0, "carbs": 24.0, "fat": 14.0},
    "french_fries": {"calories": 312, "protein":  3.4, "carbs": 41.0, "fat": 15.0},
    "fried_rice":   {"calories": 163, "protein":  3.5, "carbs": 28.0, "fat":  4.3},
    "sushi":        {"calories": 143, "protein":  5.8, "carbs": 27.0, "fat":  1.0},
    "steak":        {"calories": 242, "protein": 26.0, "carbs":  0.0, "fat": 15.0},
    "omelette":     {"calories": 154, "protein": 10.0, "carbs":  1.6, "fat": 12.0},
    "ice_cream":    {"calories": 207, "protein":  3.5, "carbs": 24.0, "fat": 11.0},
    "ramen":        {"calories": 436, "protein": 18.0, "carbs": 54.0, "fat": 16.0},
    "pancakes":      {"calories": 291, "protein":  7.9, "carbs": 37.0, "fat": 13.0}
}

CLASS_NAMES = [
    "french_fries", "fried_rice", "hamburger", "ice_cream",
    "omelette", "pancakes", "pizza", "ramen", "steak", "sushi"
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenet_best.h5")

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(model, img_array):
    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_index]) * 100
    predicted_label = CLASS_NAMES[predicted_index]
    return predicted_label, confidence, predictions[0]

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("SnapCal - Food Calorie Estimator")
st.caption("Upload a photo of your food to get its nutritional values per 100g")

model = load_model()

tab1, tab2 = st.tabs(["Upload Image", "Take Photo"])

with tab1:
    uploaded_file = st.file_uploader(
        "Choose a food image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        image = Image.open(uploaded_file)

with tab2:
    camera_photo = st.camera_input("Take a photo of your food")
    if camera_photo:
        image = Image.open(camera_photo)

if 'image' in dir() and image is not None:

    st.image(image, caption="Your food image", use_column_width=True)

    with st.spinner("Analysing your food..."):
        img_array = preprocess_image(image)
        label, confidence, all_preds = predict(model, img_array)

    nutrition = NUTRITION_DATA.get(label)
    display_name = label.replace("_", " ").title()

    st.divider()

    # Result header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"{display_name}")
        st.caption("Nutritional values per 100g")
    with col2:
        st.metric("Confidence", f"{confidence:.1f}%")

    st.progress(confidence / 100)

    st.divider()

    # Macro cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Calories",  f"{nutrition['calories']} kcal")
    c2.metric("Protein",   f"{nutrition['protein']}g")
    c3.metric("Carbs",     f"{nutrition['carbs']}g")
    c4.metric("Fat",       f"{nutrition['fat']}g")

    st.divider()

    # Top 3 predictions
    with st.expander("See all predictions"):
        top3_indices = np.argsort(all_preds)[::-1][:3]
        for i, idx in enumerate(top3_indices):
            name = CLASS_NAMES[idx].replace("_", " ").title()
            prob = all_preds[idx] * 100
            st.write(f"**{i+1}. {name}** — {prob:.1f}%")
            st.progress(float(all_preds[idx]))

    st.divider()
    st.caption("⚠️ Values shown are for 100g of food. This app currently recognises 10 food types. Nutritional data sourced from USDA FoodData Central.")