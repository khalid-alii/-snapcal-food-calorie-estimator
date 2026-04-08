# SnapCal — Food Calorie Estimator

An AI-powered food recognition app that identifies food from photos 
and returns nutritional values per 100g.

## Features
- Classifies 10 food types using MobileNetV2
- Returns calories, protein, carbs, and fat per 100g
- Upload a photo or use your camera directly

## Models compared
| Model | Test Accuracy |
|-------|--------------|
| MobileNetV2 | 0.533 |
| ResNet50 | 0.3813 |
| EfficientNetB0 | 0.1267 |
| InceptionV3 | 0.8453 |

## Tech stack
- TensorFlow / Keras — model training
- Streamlit — web application
- Google Colab (A100 GPU) — training environment

## Dataset
Food-101 (ETH Zurich) — 10 classes, 500 images each

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Limitations
- Recognises 10 food types only
- Nutritional values shown per 100g, not per actual portion