"""
adverse_effects_analysis.py

Refactored from: Adverse_Effects.ipynb
- Class-based structured data analysis
- Flexible plotting for adverse effect frequency
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class AdverseEffectsAnalyzer:
    def __init__(self, file_path: str, effect_column: str = "effect", category_column: Optional[str] = None):
        """
        Initialize analyzer.
        :param file_path: Path to CSV or Excel file
        :param effect_column: Column containing adverse effect terms
        :param category_column: Optional column for grouping by category
        """
        self.file_path = file_path
        self.effect_column = effect_column
        self.category_column = category_column
        self.data = None

    def load_data(self):
        """
        Load CSV or Excel data into a DataFrame.
        """
        if self.file_path.lower().endswith(".csv"):
            self.data = pd.read_csv(self.file_path)
        elif self.file_path.lower().endswith((".xls", ".xlsx")):
            self.data = pd.read_excel(self.file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        print(f"Loaded {len(self.data)} rows from {self.file_path}")

    def clean_data(self):
        """
        Remove rows with missing adverse effect values.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Run load_data() first.")
        self.data = self.data.dropna(subset=[self.effect_column])
        print(f"Data cleaned. Remaining rows: {len(self.data)}")

    def analyze_effects(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Count frequency of each adverse effect.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Run load_data() first.")

        counts = self.data[self.effect_column].value_counts().reset_index()
        counts.columns = [self.effect_column, "count"]

        if top_n:
            counts = counts.head(top_n)

        return counts

    def analyze_by_category(self) -> pd.DataFrame:
        """
        Count frequency of adverse effects by category.
        """
        if self.category_column is None:
            raise ValueError("No category column specified.")
        if self.data is None:
            raise ValueError("Data not loaded. Run load_data() first.")

        counts = self.data.groupby(self.category_column)[self.effect_column].count().reset_index()
        counts.columns = [self.category_column, "count"]
        return counts.sort_values("count", ascending=False)

    def plot_effects(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot the most common adverse effects.
        """
        counts = self.analyze_effects(top_n)
        plt.figure(figsize=figsize)
        plt.bar(counts[self.effect_column], counts["count"], color="skyblue")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Adverse Effect")
        plt.ylabel("Frequency")
        plt.title(f"Top {top_n} Adverse Effects")
        plt.tight_layout()
        plt.show()

    def plot_by_category(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot adverse effects counts by category.
        """
        if self.category_column is None:
            raise ValueError("No category column specified.")
        counts = self.analyze_by_category()
        plt.figure(figsize=figsize)
        plt.bar(counts[self.category_column], counts["count"], color="orange")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.title("Adverse Effects by Category")
        plt.tight_layout()
        plt.show()


def main():
    # Example usage
    file_path = "adverse_effects.csv"  # Change to your file path
    analyzer = AdverseEffectsAnalyzer(file_path, effect_column="effect", category_column="category")

    analyzer.load_data()
    analyzer.clean_data()

    print(analyzer.analyze_effects(top_n=15))
    analyzer.plot_effects(top_n=15)

    if analyzer.category_column:
        print(analyzer.analyze_by_category())
        analyzer.plot_by_category()


if __name__ == "__main__":
    main()
