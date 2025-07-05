# src/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

class EDA:
    def __init__(self, data):
        self.df = data
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    #def load_data(filepath):
        #return pd.read_excel(filepath)

    def complaints_by_column(self, column_name):
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame.")

        value_counts = self.df[column_name].value_counts().reset_index()
        value_counts.columns = [column_name, 'Count']

        plt.figure(figsize=(10, 6))
        sns.barplot(data=value_counts, x='Count', y=column_name, palette="viridis", legend=False)
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

        result = {
            "With Narrative": with_narrative,
            "Without Narrative": without_narrative
        }

        #print(result)  # force output to show when function is called
        return result

    def filter_complaints(self):
        valid_products = [
            'Credit card',
            'payday loan, title loan, or Personal loan',
            'Buy Now, Pay Later (BNPL)',
            'Checking or savings account',
            'Money transfers'
        ]

        # Step 1: Filter for selected products
        filtered_df = self.df[self.df['Product'].isin(valid_products)]

        # Step 2: Remove records with empty narratives
        filtered_df = filtered_df[filtered_df['Consumer complaint narrative'].notna()]

        return filtered_df.reset_index(drop=True)


    def clean_text(self, text):
        if pd.isnull(text):
            return ""

        # Lowercase
        text = text.lower()

        # Remove boilerplate phrases
        boilerplate_phrases = [
            "i am writing to file a complaint",
            "this is in reference to",
            "i want to report",
            "i am writing about",
            "i would like to lodge a complaint"
        ]
        for phrase in boilerplate_phrases:
            text = text.replace(phrase, "")

        # Remove special characters, numbers, punctuation
        text = re.sub(r"[^a-z\s]", " ", text)

        # Tokenize
        words = text.split()

        # Remove stopwords & lemmatize
        cleaned_words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]

        # Rejoin
        cleaned_text = " ".join(cleaned_words)
        return cleaned_text

    def clean_consumer_narratives(self, column='Consumer complaint narrative'):
        self.df['cleaned_narrative'] = self.df[column].apply(self.clean_text)
        return self.df
