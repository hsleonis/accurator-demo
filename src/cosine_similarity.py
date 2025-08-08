import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class CosineSimilarityAnalyzer:
    def __init__(self, csv_path: str):
        """Initialize with a CSV file path."""
        self.df = pd.read_csv(csv_path)

    def filter_has_text(self):
        """Filter rows where HasText == 1."""
        self.df = self.df[self.df['HasText'] == 1]

    def filter_and_analyze(self, column_flag: str, column_score: str, threshold: float):
        """
        Filter by flag column, sort by score column, plot histogram,
        and calculate stats above threshold.
        """
        subset = self.df[self.df[column_flag] == 1].sort_values(column_score)

        # Histogram
        plt.hist(subset[column_score])
        plt.title(f"Distribution of {column_score}")
        plt.xlabel(column_score)
        plt.ylabel("Frequency")
        plt.show()

        # Stats
        count_total = len(subset)
        count_above = len(subset[subset[column_score] >= threshold])
        percentage_above = (count_above / count_total) * 100 if count_total > 0 else 0

        return {
            "count_total": count_total,
            "count_above_threshold": count_above,
            "percentage_above_threshold": percentage_above
        }


def main():
    csv_path = "compare_adsum.csv"
    analyzer = CosineSimilarityAnalyzer(csv_path)

    analyzer.filter_has_text()

    # Abstract analysis
    abstract_stats = analyzer.filter_and_analyze(
        column_flag='abstract',
        column_score='CsAbstract',
        threshold=0.7
    )
    print("Abstract Stats:", abstract_stats)

    # Summary analysis
    summary_stats = analyzer.filter_and_analyze(
        column_flag='summary',
        column_score='CsSummery',
        threshold=0.7
    )
    print("Summary Stats:", summary_stats)


if __name__ == "__main__":
    main()
