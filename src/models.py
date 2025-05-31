import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, SimpleRNN, LSTM, GRU, Bidirectional, Dense, Dropout, Concatenate

class CrimeModel:
    def __init__(self, seq_shape, text_shape, num_classes):
        self.seq_shape = seq_shape
        self.text_shape = text_shape
        self.num_classes = num_classes
        self.model = None

    def build_rnn(self):
        """Build a Simple RNN model with time-series and NLP inputs."""
        seq_input = Input(shape=self.seq_shape, name='seq_input')
        text_input = Input(shape=self.text_shape, name='text_input')
        x = SimpleRNN(64)(seq_input)
        x = Concatenate()([x, text_input])
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(self.num_classes, activation='softmax')(x)
        self.model = Model([seq_input, text_input], output)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return self.model

    def build_lstm(self):
        """Build an LSTM model with time-series and NLP inputs."""
        seq_input = Input(shape=self.seq_shape, name='seq_input')
        text_input = Input(shape=self.text_shape, name='text_input')
        x = LSTM(64)(seq_input)
        x = Concatenate()([x, text_input])
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(self.num_classes, activation='softmax')(x)
        self.model = Model([seq_input, text_input], output)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return self.model

    def build_gru(self):
        """Build a GRU model with time-series and NLP inputs."""
        seq_input = Input(shape=self.seq_shape, name='seq_input')
        text_input = Input(shape=self.text_shape, name='text_input')
        x = GRU(64)(seq_input)
        x = Concatenate()([x, text_input])
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(self.num_classes, activation='softmax')(x)
        self.model = Model([seq_input, text_input], output)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return self.model

    def build_bidirectional_lstm(self):
        """Build a Bidirectional LSTM model with time-series and NLP inputs."""
        seq_input = Input(shape=self.seq_shape, name='seq_input')
        text_input = Input(shape=self.text_shape, name='text_input')
        x = Bidirectional(LSTM(64))(seq_input)
        x = Concatenate()([x, text_input])
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(self.num_classes, activation='softmax')(x)
        self.model = Model([seq_input, text_input], output)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return self.model

    def build_transformer(self):
        """Build a Transformer model with time-series and NLP inputs."""
        seq_input = Input(shape=self.seq_shape, name='seq_input')
        text_input = Input(shape=self.text_shape, name='text_input')
        projected = Dense(64)(seq_input)
        x = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(projected, projected)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = Concatenate()([x, text_input])
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(self.num_classes, activation='softmax')(x)
        self.model = Model([seq_input, text_input], output)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return self.model

    def train(self, X_seq_train, X_text_train, y_train, X_seq_val, X_text_val, y_val, epochs=10, batch_size=32):
        """Train the model."""
        print("Training model...")
        history = self.model.fit(
            [X_seq_train, X_text_train], y_train,
            validation_data=([X_seq_val, X_text_val], y_val),
            epochs=epochs, batch_size=batch_size
        )
        print("Training completed.")
        return history

    def save(self, path):
        """Save the model."""
        print(f"Saving model to {path}...")
        self.model.save(path)
        print("Model saved.")