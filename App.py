import streamlit as st
import joblib
import pandas as pd


model  =  joblib.load("Classifier.pkl")

def predict(values):
    prediction =  model.predict(values)
    return prediction


def main():
    st.title("First ML Deployment")
    st.write("Iris prediction")
    feature1 = st.number_input("SepalLengthCm")
    feature2 = st.number_input("SepalWidthCm")
    feature3 = st.number_input("PetalLengthCm")
    feature4 = st.number_input("PetalWidthCm")
    input_data = pd.DataFrame( { "SepalLengthCm" : [feature1],
                                 "SepalWidthCm" : [feature2],
                                 "PetalLengthCm" : [feature3],
                                 "PetalWidthCm" : [feature4] })
    if st.button("Predict"):
        prediction = predict(input_data)
        st.write("Prediction : ",prediction)

if __name__=="__main__":
    main() 
