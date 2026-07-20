
import streamlit as st   
import pickle
import numpy as np

with open("iris_dataset_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Iris Flower Prediction")
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 0.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.4, 0.1)
petal_length = st.slider("Petal Length (cm)", 1.0, 6.9, 0.1)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.1)

if st.button("Prediction"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    species= ['setosa', 'versicolor', 'virginica']
    st.write(f"Predicted Species: {[prediction[0]]}")