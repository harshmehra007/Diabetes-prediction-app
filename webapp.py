import numpy as np
import streamlit as st
import pickle

loaded_model = pickle.load(open('trained_model.sav','rb'))

def df(input_data):

    input_data_array = np.asarray(input_data)

    input_reshape = input_data_array.reshape(1,-1)

    pridiction  = loaded_model.predict(input_reshape)

    print(pridiction)

    if (pridiction[0] == 0):
     return('sugar nhi h')
    else:
     return('sugar h')

# app title
st.title("Diabetes Prediction App")

# input fields
st.subheader("Enter Patient Details:")
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=25)

# prediction button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    prediction = loaded_model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ The person is likely Diabetic")
    else:
        st.success("✅ The person is NOT Diabetic")