import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class EDA:
    def __init__(self, data):
        self.data = data
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

    def analyze_data(self):
        """Perform detailed data analysis and print results."""
        print("\n=== Exploratory Data Analysis ===")
        print("\nDataset Head:")
        print(self.data.head())
        print("\nDataset Info:")
        print(self.data.info())
        print("\nDataset Description:")
        print(self.data.describe())
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        print("\nCrime Type Distribution:")
        print(self.data['Primary Type'].value_counts())
        print("\nArrest Rate by Crime Type:")
        print(self.data.groupby('Primary Type')['Arrest'].mean())
        print("\nTop 5 Location Descriptions:")
        print(self.data['Location Description'].value_counts().head())

    def plot_crime_distribution(self):
        """Plot distribution of crime types."""
        print("Generating crime distribution plot...")
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.data, x='Primary Type')
        plt.title('Distribution of Crime Types')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(self.output_dir, 'crime_distribution.png'))
        plt.close()
        print("Crime distribution plot saved.")

    def plot_temporal_trends(self):
        """Plot crime trends over time."""
        print("Generating temporal trends plot...")
        self.data['Year'] = self.data['Date'].dt.year
        yearly_crimes = self.data.groupby('Year').size()
        plt.figure(figsize=(10, 6))
        yearly_crimes.plot(kind='line')
        plt.title('Crime Trends by Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Crimes')
        plt.savefig(os.path.join(self.output_dir, 'temporal_trends.png'))
        plt.close()
        print("Temporal trends plot saved.")

    def plot_crime_by_community(self):
        """Plot crimes by community area."""
        print("Generating community area plot...")
        plt.figure(figsize=(12, 6))
        sns.countplot(data=self.data, x='Community Area', order=self.data['Community Area'].value_counts().index[:10])
        plt.title('Top 10 Community Areas by Crime Count')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(self.output_dir, 'crime_by_community.png'))
        plt.close()
        print("Community area plot saved.")