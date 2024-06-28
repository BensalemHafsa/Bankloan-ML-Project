import streamlit as st
import joblib
import pandas as pd


model  =  joblib.load("Classifier.pkl")

def predict(values):
    prediction =  model.predict(values)
    return prediction


def main():
    st.title("ML Project Deployment")
    st.write("Bankloan Prediction")
    feature1 = st.number_input("Age")
    feature2 = st.number_input("Experience")
    feature3 = st.number_input("Income")
    feature4 = st.number_input("ZIP.Code")
    feature5 = st.number_input("Family")
    feature6 = st.number_input("CCAvg")
    feature7 = st.number_input("Education")
    feature8 = st.number_input("Mortgag")
    feature9 = st.number_input("Securities.Account")
    feature10 = st.number_input("CD.Account")
    feature11 = st.number_input("Online")
    feature12 = st.number_input("CreditCard")
    input_data = pd.DataFrame( { "Age" : [feature1],
                                 "Experience" : [feature2],
                                 "Income" : [feature3],
                                  "ZIP.Code" : [feature4],
                                  "Family" : [feature5],
                                  "CCAvg" : [feature6],
                                  "Education" : [feature7],
                                  "Mortgag" : [feature8],
                                  "Securities.Account" : [feature9],
                                  "CD.Account" : [feature10],
                                  "Online" : [feature11],
                                  "CreditCard" : [feature12]})
                                
    if st.button("Predict"):
        prediction = predict(input_data)
        st.write("Prediction : ",prediction)

if __name__=="__main__":
    main() 
