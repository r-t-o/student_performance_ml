import streamlit as st
import joblib
# Load trained model
model = joblib.load("model.pkl")

st.title("🎓 Student Performance Predictor")

st.header("Enter Student Details")

study_hours = st.slider("Study Hours", 0, 12, 5)
sleep_hours = st.slider("Sleep Hours", 0, 12, 6)
attendance = st.slider("Attendance (%)", 0, 100, 75)

if st.button("Predict Performance"):
    input_data = [[study_hours, sleep_hours, attendance]]
    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Score: {prediction:.2f}")

    # Add category
    if prediction >= 75:
        st.write("Performance: Excellent")
    elif prediction >= 60:
        st.write("Performance: Good")
    elif prediction >= 40:
        st.write("Performance: Average")
    else:
        st.write("Performance: Needs Improvement")

