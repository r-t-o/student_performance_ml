🎓 Student Performance Prediction System

## Overview
This project predicts student performance based on:
- Study Hours
- Sleep Hours
- Attendance
It uses Machine Learning to estimate a student's performance score and categorize it.

## Features
- Predicts performance using Linear Regression
- Provides performance category (Excellent / Good / Average / Needs Improvement)
- Simple and interactive web app using Streamlit

## Tech Stack
- Python
- Scikit-learn
- Pandas
- Streamlit
- Joblib

## How it Works
1. Data is used to train a Linear Regression model
2. Model is saved using joblib
3. Streamlit app takes user input
4. Model predicts performance score
5. Output is displayed with category

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py

## 🌐 Live Demo
https://studentperformanceml-j8tr6dnsdenesaqx6icctr.streamlit.app/
