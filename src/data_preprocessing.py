import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.location_encoder = LabelEncoder()
        self.crime_encoder = LabelEncoder()
        self.tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        self.sequences = None
        self.text_features = None
        self.labels = None

    def load_data(self):
        """Load and filter dataset."""
        print("Loading dataset...")
        self.data = pd.read_csv(self.file_path)
        print("Dataset loaded. Shape:", self.data.shape)
        # Select relevant columns
        self.data = self.data[['Date', 'Primary Type', 'Community Area', 'Description', 'Arrest', 'Location Description']]
        # Convert Date to datetime
        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
        # Filter top 3 crime types
        top_crimes = ['THEFT', 'BATTERY', 'ASSAULT']
        self.data = self.data[self.data['Primary Type'].isin(top_crimes)]
        # Handle missing values
        self.data['Description'] = self.data['Description'].fillna('')
        self.data['Location Description'] = self.data['Location Description'].fillna('UNKNOWN')
        self.data = self.data.dropna(subset=['Date', 'Community Area'])
        print("Data filtered for top crimes and cleaned. Shape:", self.data.shape)

    def preprocess(self):
        """Preprocess data for modeling."""
        print("Preprocessing data...")

        # Encode categorical variables
        self.data['Crime_Label'] = self.crime_encoder.fit_transform(self.data['Primary Type'])
        self.data['Arrest'] = self.data['Arrest'].astype(int)
        self.data['Location_Label'] = self.location_encoder.fit_transform(self.data['Location Description'])
        # Extract date features
        self.data['Year'] = self.data['Date'].dt.year
        self.data['Month'] = self.data['Date'].dt.month
        self.data['Day'] = self.data['Date'].dt.day
        # Extract TF-IDF features from Description
        print("Extracting TF-IDF features...")
        self.data['Text_Features'] = self.tfidf.fit_transform(self.data['Description']).toarray().tolist()
        # Aggregate by day and community area
        print("Aggregating data by day and community area...")
        self.data = self.data.groupby(['Year', 'Month', 'Day', 'Community Area']).agg({
            'Crime_Label': lambda x: x.mode()[0],
            'Arrest': 'mean',
            'Location_Label': lambda x: x.mode()[0],
            'Text_Features': lambda x: np.mean(np.stack(x.values), axis=0) if not x.empty else np.zeros(100)
        }).reset_index()
        print("Data aggregated. Shape:", self.data.shape)
        # Create sequences
        self.create_sequences()

    def create_sequences(self, seq_length=5):
        """Create sequences of 5 days for each community area."""
        print("Creating sequences...")
        sequences = []
        text_features = []
        labels = []
        for area in self.data['Community Area'].unique():
            area_data = self.data[self.data['Community Area'] == area].sort_values(['Year', 'Month', 'Day'])
            crime_labels = area_data['Crime_Label'].values
            arrests = area_data['Arrest'].values
            locations = area_data['Location_Label'].values
            texts = np.array(area_data['Text_Features'].tolist())
            for i in range(len(crime_labels) - seq_length):
                seq = np.column_stack((crime_labels[i:i + seq_length], 
                                       arrests[i:i + seq_length], 
                                       locations[i:i + seq_length]))
                sequences.append(seq)
                text_features.append(texts[i + seq_length - 1])
                labels.append(crime_labels[i + seq_length])
        self.sequences = np.array(sequences)
        self.text_features = np.array(text_features)
        self.labels = np.array(labels)
        print("Sequences created. Sequence shape:", self.sequences.shape, "Text features shape:", self.text_features.shape)

    def split_data(self, test_size=0.2):
        """Split data into train and test sets."""
        print("Splitting data...")
        X_seq_train, X_seq_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
            self.sequences, self.text_features, self.labels, test_size=test_size, random_state=42
        )
        print("Data split. Train sequence shape:", X_seq_train.shape, "Test sequence shape:", X_seq_test.shape)
        return X_seq_train, X_seq_test, X_text_train, X_text_test, y_train, y_test

    def get_crime_encoder(self):
        """Return the crime label encoder."""
        return self.crime_encoder

    def get_location_encoder(self):
        """Return the location label encoder."""
        return self.location_encoder

    def get_tfidf(self):
        """Return the TF-IDF vectorizer."""
        return self.tfidf