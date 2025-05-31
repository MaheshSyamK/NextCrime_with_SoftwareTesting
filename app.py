import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import numpy as np
from src.predict import Predictor
from src.data_preprocessing import DataPreprocessor

@st.cache_resource
def load_preprocessor():
    preprocessor = DataPreprocessor('data/Crimes_2001_to_Present.csv')
    preprocessor.load_data()
    preprocessor.preprocess()
    return preprocessor

def main():
    st.title("Chicago Crime Prediction")
    st.write("Predict the next day's crime category in a community area using past crime sequences and text descriptions.")

    # Load and cache the preprocessor
    preprocessor = load_preprocessor()
    crime_encoder = preprocessor.get_crime_encoder()
    location_encoder = preprocessor.get_location_encoder()
    tfidf = preprocessor.get_tfidf()

    # User input
    st.subheader("Input for Prediction")
    community_area = st.number_input("Community Area", min_value=1, max_value=77, value=1)
    st.write("Enter the last 5 days' crime categories:")
    crime_options = crime_encoder.classes_.tolist()
    sequence = []
    for i in range(5):
        crime = st.selectbox(f"Day {i+1} Crime", crime_options, key=f"crime_{i}")
        arrest = st.checkbox(f"Day {i+1} Arrest", key=f"arrest_{i}")
        location = st.text_input(f"Day {i+1} Location Description", value="UNKNOWN", key=f"location_{i}")
        try:
            location_code = location_encoder.transform([location])[0]
        except ValueError:
            location_code = location_encoder.transform(['UNKNOWN'])[0]
        crime_code = crime_encoder.transform([crime])[0]
        sequence.append([crime_code, int(arrest), location_code])
    
    description = st.text_area("Recent Crime Description", "Enter a brief description of recent crimes...")

    # Model selection
    model_type = st.selectbox("Select Model", ['rnn', 'lstm', 'gru', 'bidirectional_lstm', 'transformer'])

    if st.button("Predict"):
        predictor = Predictor(f'models/{model_type}_model.keras', crime_encoder, tfidf)
        prediction = predictor.predict(np.array(sequence), description)
        st.success(f"Predicted Crime Category for Community Area {community_area} Tomorrow: {prediction}")

if __name__ == "__main__":
    main()
