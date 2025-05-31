import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.data_preprocessing import DataPreprocessor
from src.eda import EDA
from src.train import Trainer

def main():
    print("Starting Chicago Crime Prediction Project...")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor('data/Crimes_2001_to_Present.csv')
    
    # Load and preprocess data
    preprocessor.load_data()
    
    # Perform EDA
    print("\nPerforming EDA...")
    eda = EDA(preprocessor.data)
    eda.analyze_data()
    eda.plot_crime_distribution()
    eda.plot_temporal_trends()
    eda.plot_crime_by_community()

    # Preprocess for model
    preprocessor.preprocess()
    
    # Prepare data for modeling
    X_seq_train, X_seq_test, X_text_train, X_text_test, y_train, y_test = preprocessor.split_data()
    num_classes = len(preprocessor.get_crime_encoder().classes_)
    
    # Train models
    print("\nTraining models...")
    trainer = Trainer(X_seq_train, X_text_train, y_train, X_seq_test, X_text_test, y_test, num_classes)
    trainer.train_models()

if __name__ == "__main__":
    main()