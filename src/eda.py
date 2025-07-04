# src/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    def __init__(self, data):
        self.df = data

    #def load_data(filepath):
        #return pd.read_excel(filepath)

    def complaints_by_column(self, column_name):
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame.")

        value_counts = self.df[column_name].value_counts()

        plt.figure(figsize=(10, 6))
        sns.barplot(x=value_counts.values, y=value_counts.index, palette="viridis")
        plt.title(f"Number of Complaints by {column_name}")
        plt.xlabel("Count")
        plt.ylabel(column_name)
        plt.tight_layout()
        plt.show()

        return value_counts


    def narrative_length_distribution(self):
        self.df['narrative_length'] = self.df['Consumer complaint narrative'].dropna().apply(lambda x: len(str(x).split()))
        plt.figure(figsize=(10,6))
        sns.histplot(self.df['narrative_length'].dropna(), bins=50, kde=True)
        plt.title("Distribution of Consumer Complaint Narrative Lengths")
        plt.xlabel("Number of Words")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
        return self.df['narrative_length'].describe()

    def count_narrative_presence(self):
        with_narrative = self.df['Consumer complaint narrative'].notna().sum()
        without_narrative = self.df['Consumer complaint narrative'].isna().sum()
        return {
            "With Narrative": with_narrative,
            "Without Narrative": without_narrative
        }
