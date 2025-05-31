# NextCrime_with_SoftwareTesting

Predicts next-day crime types in Chicago using NLP and time series models via a Streamlit app. Includes RNN, LSTM, and software testing extension.
This project is a machine learning application designed to predict the **next day's crime category** in a given **community area** of Chicago. It combines **sequence modeling** and **natural language processing (NLP)** to make informed predictions based on historical data and textual descriptions of crimes.



## 📌 Features
- Predicts crime type for the next day using:
  - Last 5 days’ crime categories
  - Arrest status
  - Crime location descriptions
  - Latest textual crime description (NLP)
- Supports multiple deep learning models:
  - RNN, LSTM, GRU, Bidirectional LSTM, Transformer
- User-friendly web interface built with **Streamlit**
- Trained on the official **Chicago Crimes Dataset**
- Encodes categorical data using LabelEncoders & TF-IDF
- Modular architecture with separate modules for preprocessing, prediction, and interface



## 🧠 Technologies Used
- Python
- TensorFlow / Keras
- NumPy / Pandas
- Scikit-learn (for preprocessing)
- Streamlit (for web interface)
- TF-IDF (text vectorization)



## 🧪 Software Testing (Extension)
As part of the project extension, we plan to integrate **automated software testing** for both backend and interface components:

- ✅ **Unit Testing** (using `unittest` or `pytest`) for:
  - Preprocessing pipeline
  - Prediction logic
- ✅ **Integration Testing** for end-to-end workflow
- ✅ **Streamlit interface testing** using tools like `pytest-streamlit` (optional)

This will improve code reliability, reusability, and maintainability — making the project ready for real-world deployment or further enhancements.


## 📂 Folder Structure
.
├── app.py                    
├── models/                   
├── data/                     
├── src/
│   ├── data_preprocessing.py 
│   └── predict.py            
├── requirements.txt
└── README.md
