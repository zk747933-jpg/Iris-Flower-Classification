import streamlit as st
import numpy as np
import pickle

# Load trained model
with open("iris_dataset.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🌸 Iris Flower Predictor")

# User input
sepal_length = st.slider("Sepal Length (cm)", 4.3, 7.9, 5.4)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.4, 3.4)
petal_length = st.slider("Petal Length (cm)", 1.0, 6.9, 1.3)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Prediction
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = model.predict(input_data)

    species = ["Setosa", "Versicolor", "Virginica"]

    st.success(f"🌼 Predicted Iris Species: **{species[prediction[0]]}**")