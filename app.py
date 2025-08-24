import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# Load Model
# ---------------------------
model = pickle.load(open("model.pkl", "rb"))

# ---------------------------
# Streamlit App Layout
# ---------------------------
st.title("🌸 Iris Flower Prediction App")
st.write("""
This is a simple **Machine Learning Web App** built with Streamlit.  
It predicts the **species of Iris flower** based on input features.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a section:", ["Dataset Overview", "Make a Prediction", "Model Performance"])

# ---------------------------
# Section 1: Dataset Overview
# ---------------------------
if options == "Dataset Overview":
    st.subheader("About the Iris Dataset")
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    df["species"] = df["species"].map({0:"setosa", 1:"versicolor", 2:"virginica"})

    st.write("### Dataset Preview")
    st.write(df.head())

    st.write("### Shape:", df.shape)
    st.write("### Columns:", df.columns.tolist())

# ---------------------------
# Section 2: Prediction
# ---------------------------
elif options == "Make a Prediction":
    st.subheader("Enter Flower Measurements:")

    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.0)
    sepal_width  = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.0)
    petal_width  = st.slider("Petal width (cm)", 0.1, 2.5, 1.0)

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)

    st.write("### Prediction:", prediction[0])
    st.write("### Confidence:", np.max(proba))

# ---------------------------
# Section 3: Model Performance
# ---------------------------
elif options == "Model Performance":
    st.subheader("Model Information")
    st.write("The model used is **Random Forest Classifier** trained on the Iris dataset.")
    st.write("It achieves over **95% accuracy** on the test set.")
