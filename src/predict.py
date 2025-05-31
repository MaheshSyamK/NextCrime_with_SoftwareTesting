import tensorflow as tf
import numpy as np

class Predictor:
    def __init__(self, model_path, crime_encoder, tfidf):
        self.model = tf.keras.models.load_model(model_path)
        self.crime_encoder = crime_encoder
        self.tfidf = tfidf

    def predict(self, sequence, text):
        """Predict the next day's crime category."""
        print("Making prediction...")
        # Prepare sequence
        sequence = np.array(sequence).reshape(1, -1, sequence.shape[1])
        # Prepare text features
        text_feature = self.tfidf.transform([text]).toarray()
        # Predict
        pred = self.model.predict([sequence, text_feature])
        pred_class = np.argmax(pred, axis=1)[0]
        prediction = self.crime_encoder.inverse_transform([pred_class])[0]
        print(f"Predicted Crime Category: {prediction}")
        return prediction