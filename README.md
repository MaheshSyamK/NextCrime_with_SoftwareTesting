Chicago Crime Prediction Project

Overview

This project analyzes the Chicago Crime dataset from Kaggle to predict the next day's crime category based on a sequence of the last 5 days' crime categories. It includes data preprocessing, exploratory data analysis (EDA), visualization, and modeling using RNN, LSTM, GRU, Bidirectional LSTM, and Transformer models. The project is deployed using a Streamlit app.

Directory Structure

chicago_crime_prediction/
├── data/
│   └── Crimes_2001_to_Present.csv
├── models/
│   └── (saved models)
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── models.py
│   ├── train.py
│   └── predict.py
├── app.py
├── main.py
├── requirements.txt
└── README.md

Setup





Download Dataset: Download the Chicago Crime dataset from Kaggle and place it in the data/ folder as Crimes_2001_to_Present.csv.
Install Dependencies: Run pip install -r requirements.txt.
Run the Project: Execute python main.py to preprocess data, perform EDA, and train models.
Run Streamlit App: Execute streamlit run app.py to launch the prediction interface.

Usage





EDA: Visualizations are saved in the output/ folder.



Training: Models are trained and saved in the models/ folder.



Prediction: Use the Streamlit app to input a sequence of 5 days' crime categories and select a model to predict the next day's crime category.

Model Details





Input: Sequence of 5 days' crime categories (THEFT, BATTERY, ASSAULT) for a community area.



Output: Predicted crime category for the next day.



Models: RNN, LSTM, GRU, Bidirectional LSTM, Transformer.



Metrics: Accuracy, precision, recall, F1-score (printed during training).

Requirements

See requirements.txt for dependencies.