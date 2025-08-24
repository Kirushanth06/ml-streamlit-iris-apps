import streamlit as st
import pickle
from sklearn.datasets import load_iris

with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

iris = load_iris()

st.sidebar.title("Menu")
page = st.sidebar.radio("Go to:", ["Home", "Data", "Predict"])

if page == "Home":
    st.title("Iris Flower Classification")
    st.write("Use this app to predict iris flower species!")

elif page == "Data":
    st.title("Iris Dataset")
    st.write("Dataset features:")
    st.write(iris.feature_names)
    st.write("Target names:")
    st.write(iris.target_names)

elif page == "Predict":
    st.title("Species Predictor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sepal_length = st.number_input("Sepal Length (cm)", value=5.1)
        sepal_width = st.number_input("Sepal Width (cm)", value=3.5)
    
    with col2:
        petal_length = st.number_input("Petal Length (cm)", value=1.4)
        petal_width = st.number_input("Petal Width (cm)", value=0.2)
    
    if st.button("Predict"):
        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        species = iris.target_names[prediction][0]
        st.write("**Predicted Species:**")
        st.success(f"- Predicted Species: {species}")
