"""
fasttext_sentiment.py

Refactored from: fastText_(Sentiment_Analysis).ipynb
- Portable (no Colab dependencies)
- Uses fastText for text classification
- Supports training, prediction, and evaluation
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

try:
    import fasttext
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "fasttext"])
    import fasttext


class FastTextSentimentAnalyzer:
    def __init__(self, label_prefix: str = "__label__", model_path: str = "fasttext_model.bin"):
        self.label_prefix = label_prefix
        self.model_path = model_path
        self.model = None

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by removing non-alphanumeric characters and extra spaces.
        """
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()

    def prepare_fasttext_format(self, df: pd.DataFrame, text_col: str, label_col: str, output_file: str):
        """
        Convert DataFrame into fastText format and save to a file.
        """
        with open(output_file, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                label = f"{self.label_prefix}{row[label_col]}"
                text = self.clean_text(str(row[text_col]))
                f.write(f"{label} {text}\n")

    def train(self, train_file: str, lr: float = 1.0, epoch: int = 25, word_ngrams: int = 2, verbose: int = 2):
        """
        Train a fastText supervised model.
        """
        self.model = fasttext.train_supervised(
            input=train_file,
            lr=lr,
            epoch=epoch,
            wordNgrams=word_ngrams,
            verbose=verbose
        )
        self.model.save_model(self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """
        Load an existing fastText model.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model = fasttext.load_model(self.model_path)

    def predict(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Predict labels and probabilities for a list of texts.
        """
        if self.model is None:
            self.load_model()
        predictions = []
        for text in texts:
            cleaned = self.clean_text(text)
            labels, probs = self.model.predict(cleaned)
            predictions.append((labels[0].replace(self.label_prefix, ""), probs[0]))
        return predictions

    def evaluate(self, test_file: str) -> dict:
        """
        Evaluate the model on a fastText formatted test file.
        """
        if self.model is None:
            self.load_model()
        result = self.model.test(test_file)
        metrics = {
            "samples": result[0],
            "precision": result[1],
            "recall": result[2]
        }
        return metrics

    def evaluate_dataframe(self, df: pd.DataFrame, text_col: str, label_col: str):
        """
        Evaluate using a DataFrame and produce classification report & confusion matrix.
        """
        preds = [self.predict([t])[0][0] for t in df[text_col]]
        true_labels = df[label_col].astype(str).tolist()

        print(classification_report(true_labels, preds))

        cm = confusion_matrix(true_labels, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=sorted(set(true_labels)),
                    yticklabels=sorted(set(true_labels)))
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

    def plot_label_distribution(self, df: pd.DataFrame, label_col: str):
        """
        Plot distribution of sentiment labels.
        """
        df[label_col].value_counts().plot(kind='bar', color='skyblue')
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.title("Label Distribution")
        plt.tight_layout()
        plt.show()


def main():
    # Example usage
    data_file = "sentiment_dataset.csv"  # Change this to your dataset path
    text_column = "text"
    label_column = "label"

    # Load dataset
    df = pd.read_csv(data_file)

    # Split dataset
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    # Initialize analyzer
    analyzer = FastTextSentimentAnalyzer()

    # Prepare data files
    analyzer.prepare_fasttext_format(train_df, text_column, label_column, "train.txt")
    analyzer.prepare_fasttext_format(test_df, text_column, label_column, "test.txt")

    # Train model
    analyzer.train("train.txt")

    # Evaluate
    metrics = analyzer.evaluate("test.txt")
    print("Evaluation metrics:", metrics)

    # Detailed evaluation
    analyzer.evaluate_dataframe(test_df, text_column, label_column)

    # Plot label distribution
    analyzer.plot_label_distribution(df, label_column)


if __name__ == "__main__":
    main()
