"""
key_analysis_nlp.py

Refactored from: Key_analysis_(NLP).ipynb
- No Colab dependencies
- NLP preprocessing for keyword frequency analysis
- Class-based, reusable design
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Tuple, Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK resources are downloaded
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


class KeyReportAnalyzer:
    def __init__(self, file_path: str, text_column: str = "text"):
        self.file_path = file_path
        self.text_column = text_column
        self.data = None
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.key_freq = Counter()

    def load_data(self):
        """
        Load CSV or Excel file into a DataFrame.
        """
        if self.file_path.lower().endswith(".csv"):
            self.data = pd.read_csv(self.file_path)
        elif self.file_path.lower().endswith((".xls", ".xlsx")):
            self.data = pd.read_excel(self.file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        print(f"Loaded {len(self.data)} rows from {self.file_path}")

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Remove non-alphanumeric characters and normalize spaces.
        """
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text))
        text = re.sub(r"\s+", " ", text)
        return text.strip().lower()

    def preprocess_text(self, text: str) -> List[str]:
        """
        Tokenize, remove stopwords, and lemmatize.
        """
        tokens = nltk.word_tokenize(self.clean_text(text))
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        lemmatized = [self.lemmatizer.lemmatize(t) for t in tokens]
        return lemmatized

    def analyze_keywords(self, top_n: Optional[int] = None):
        """
        Count keyword frequencies for the dataset.
        """
        all_tokens = []
        for text in self.data[self.text_column].dropna():
            all_tokens.extend(self.preprocess_text(text))
        self.key_freq = Counter(all_tokens)
        if top_n:
            return self.key_freq.most_common(top_n)
        return self.key_freq

    def filter_by_keyword(self, keyword: str) -> pd.DataFrame:
        """
        Filter rows containing a specific keyword.
        """
        keyword = keyword.lower()
        return self.data[self.data[self.text_column].str.contains(keyword, case=False, na=False)]

    def plot_top_keywords(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot the top N keywords as a bar chart.
        """
        if not self.key_freq:
            self.analyze_keywords()
        top_words = self.key_freq.most_common(top_n)
        words, counts = zip(*top_words)
        plt.figure(figsize=figsize)
        plt.bar(words, counts, color="skyblue")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Keyword")
        plt.ylabel("Frequency")
        plt.title(f"Top {top_n} Keywords")
        plt.tight_layout()
        plt.show()

    def export_keywords(self, output_file: str):
        """
        Save keyword frequencies to CSV.
        """
        pd.DataFrame(self.key_freq.items(), columns=["keyword", "frequency"])\
            .sort_values("frequency", ascending=False)\
            .to_csv(output_file, index=False)
        print(f"Keyword frequencies exported to {output_file}")


def main():
    # Example usage
    file_path = "key_report_1.csv"  # Change to your dataset path
    analyzer = KeyReportAnalyzer(file_path, text_column="report_text")

    analyzer.load_data()
    top_keywords = analyzer.analyze_keywords(top_n=30)
    print("Top Keywords:", top_keywords)

    analyzer.plot_top_keywords(top_n=15)
    analyzer.export_keywords("keyword_frequencies.csv")

    filtered_df = analyzer.filter_by_keyword("machine")
    print(f"Rows containing 'machine': {len(filtered_df)}")


if __name__ == "__main__":
    main()
