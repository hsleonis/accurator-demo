"""
article_set_analysis.py

Refactored from: ArticleSets_(springer)_analysis.ipynb
- Portable, modular, reusable
- Flexible plotting (bar, pie, histogram)
"""

import os
import re
from typing import Optional, Sequence
import pandas as pd
import matplotlib.pyplot as plt


class ArticleSetAnalysis:
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        """
        Load dataset from CSV or Excel file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if file_path.lower().endswith(".csv"):
            self.data = pd.read_csv(file_path, encoding=encoding)
        elif file_path.lower().endswith((".xls", ".xlsx")):
            self.data = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")

    def clean_text(self, column: str, lowercase: bool = True, remove_punct: bool = True) -> pd.Series:
        """
        Clean text in a column using regex and simple normalization.
        """
        series = self.data[column].astype(str)
        if lowercase:
            series = series.str.lower()
        if remove_punct:
            series = series.apply(lambda x: re.sub(r"[^\w\s]", "", x))
        self.data[column] = series
        return self.data[column]

    def count_by_column(self, column: str, normalize: bool = False) -> pd.Series:
        """
        Count occurrences of each unique value in a column.
        """
        return self.data[column].value_counts(normalize=normalize, dropna=False)

    def filter_by_keyword(self, column: str, keyword: str, case: bool = False) -> pd.DataFrame:
        """
        Return rows where the column contains the keyword.
        """
        mask = self.data[column].astype(str).str.contains(keyword, case=case, na=False)
        return self.data[mask]

    def summarize_statistics(self) -> pd.DataFrame:
        """
        Return basic statistics about the dataset.
        """
        return self.data.describe(include="all")

    def plot_distribution(
        self,
        column: str,
        chart_type: str = "bar",
        top_n: Optional[int] = None,
        title: Optional[str] = None,
        figsize: tuple = (10, 6),
        savepath: Optional[str] = None
    ):
        """
        Plot distribution of values in a column.

        chart_type: "bar", "pie", or "hist"
        """
        counts = self.count_by_column(column)
        if top_n is not None:
            counts = counts.head(top_n)

        fig, ax = plt.subplots(figsize=figsize)

        if chart_type == "bar":
            counts.plot(kind="bar", ax=ax)
            ax.set_ylabel("Count")
            ax.set_xlabel(column)
        elif chart_type == "pie":
            counts.plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.axis("equal")
        elif chart_type == "hist":
            self.data[column].dropna().plot(kind="hist", ax=ax)
            ax.set_xlabel(column)
        else:
            raise ValueError("Invalid chart_type. Choose 'bar', 'pie', or 'hist'.")

        if title:
            ax.set_title(title)
        plt.tight_layout()

        if savepath:
            fig.savefig(savepath)
        else:
            plt.show()
        plt.close(fig)


def main():
    file_path = "articles.csv"  # Replace with your dataset path
    analysis = ArticleSetAnalysis(file_path)

    print("Dataset preview:")
    print(analysis.data.head())

    print("\nStatistics:")
    print(analysis.summarize_statistics())

    # Example plots
    analysis.plot_distribution(column="Category", chart_type="bar", top_n=10, title="Top 10 Categories")
    analysis.plot_distribution(column="Category", chart_type="pie", top_n=5, title="Top 5 Categories")
    analysis.plot_distribution(column="PublicationYear", chart_type="hist", title="Publication Year Distribution")


if __name__ == "__main__":
    main()
