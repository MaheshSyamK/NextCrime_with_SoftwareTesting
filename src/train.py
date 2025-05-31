import os
from src.models import CrimeModel
from sklearn.metrics import classification_report
import numpy as np

class Trainer:
    def __init__(self, X_seq_train, X_text_train, y_train, X_seq_val, X_text_val, y_val, num_classes):
        self.X_seq_train = X_seq_train
        self.X_text_train = X_text_train
        self.y_train = y_train
        self.X_seq_val = X_seq_val
        self.X_text_val = X_text_val
        self.y_val = y_val
        self.num_classes = num_classes
        self.models = {}

    def train_models(self):
        model_types = ['rnn', 'lstm', 'gru', 'bidirectional_lstm', 'transformer']

        os.makedirs('models', exist_ok=True)

        for model_type in model_types:
            print(f"\n=== Training {model_type.upper()} Model ===")
            model = CrimeModel(
                seq_shape=(self.X_seq_train.shape[1], self.X_seq_train.shape[2]),
                text_shape=(self.X_text_train.shape[1],),
                num_classes=self.num_classes
            )

            if model_type == 'rnn':
                model.build_rnn()
            elif model_type == 'lstm':
                model.build_lstm()
            elif model_type == 'gru':
                model.build_gru()
            elif model_type == 'bidirectional_lstm':
                model.build_bidirectional_lstm()
            elif model_type == 'transformer':
                model.build_transformer()

            history = model.train(
                self.X_seq_train, self.X_text_train, self.y_train,
                self.X_seq_val, self.X_text_val, self.y_val
            )

            self.models[model_type] = model

            model_path = os.path.join('models', f'{model_type}_model.keras')
            model.save(model_path)

            y_pred = np.argmax(model.model.predict([self.X_seq_val, self.X_text_val]), axis=1)
            print(f"\nClassification Report for {model_type.upper()}:")
            print(classification_report(self.y_val, y_pred, zero_division=0))

    def get_models(self):
        return self.models
