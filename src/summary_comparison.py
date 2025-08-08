"""
summary_comparison.py

Refactored from: Summary_comparison.ipynb
- Portable and reusable
- Supports ROUGE and BLEU scoring
- Flexible plotting
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


class SummaryComparison:
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        """
        Load dataset from a CSV or Excel file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.lower().endswith(".csv"):
            self.data = pd.read_csv(file_path, encoding=encoding)
        elif file_path.lower().endswith((".xls", ".xlsx")):
            self.data = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")

    def compare_columns(self, col1: str, col2: str) -> pd.DataFrame:
        """
        Return basic differences between two columns.
        """
        if col1 not in self.data.columns or col2 not in self.data.columns:
            raise ValueError(f"Columns {col1} or {col2} not found in dataset")

        comparison_df = pd.DataFrame({
            f"{col1}_len": self.data[col1].astype(str).apply(len),
            f"{col2}_len": self.data[col2].astype(str).apply(len)
        })
        comparison_df["length_diff"] = comparison_df[f"{col1}_len"] - comparison_df[f"{col2}_len"]
        return comparison_df

    def calculate_bleu_rouge(self, reference_col: str, generated_col: str) -> Tuple[pd.DataFrame, dict]:
        """
        Compute BLEU and ROUGE scores for each row and average scores.
        """
        if reference_col not in self.data.columns or generated_col not in self.data.columns:
            raise ValueError(f"Columns {reference_col} or {generated_col} not found in dataset")

        smoothie = SmoothingFunction().method4
        rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        scores_list = []
        for ref, gen in zip(self.data[reference_col], self.data[generated_col]):
            ref_str, gen_str = str(ref), str(gen)

            bleu = sentence_bleu([ref_str.split()], gen_str.split(), smoothing_function=smoothie)
            rouge_scores = rouge.score(ref_str, gen_str)

            scores_list.append({
                "BLEU": bleu,
                "ROUGE-1": rouge_scores["rouge1"].fmeasure,
                "ROUGE-2": rouge_scores["rouge2"].fmeasure,
                "ROUGE-L": rouge_scores["rougeL"].fmeasure
            })

        scores_df = pd.DataFrame(scores_list)
        avg_scores = scores_df.mean().to_dict()
        return scores_df, avg_scores

    def plot_length_distribution(self, columns: Sequence[str], bins: int = 20, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot length distribution for multiple text columns.
        """
        plt.figure(figsize=figsize)
        for col in columns:
            if col not in self.data.columns:
                raise ValueError(f"Column {col} not found in dataset")
            lengths = self.data[col].astype(str).apply(len)
            plt.hist(lengths, bins=bins, alpha=0.5, label=col)
        plt.xlabel("Length (characters)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.title("Length Distribution")
        plt.tight_layout()
        plt.show()

    def plot_score_comparison(self, score_df: pd.DataFrame, figsize: Tuple[int, int] = (8, 5)):
        """
        Plot average BLEU and ROUGE scores from a score DataFrame.
        """
        avg_scores = score_df.mean().to_dict()
        plt.figure(figsize=figsize)
        plt.bar(avg_scores.keys(), avg_scores.values())
        plt.ylabel("Score")
        plt.title("Average BLEU and ROUGE Scores")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

    def export_comparison(self, df: pd.DataFrame, path: str):
        """
        Save a comparison DataFrame to CSV.
        """
        df.to_csv(path, index=False)
        print(f"Comparison results saved to {path}")


def main():
    file_path = "key_report_1.csv"  # Change to your dataset
    ref_col = "ReferenceSummary"
    gen_col = "GeneratedSummary"

    comp = SummaryComparison(file_path)

    # Length comparison
    length_df = comp.compare_columns(ref_col, gen_col)
    print(length_df.head())

    # BLEU & ROUGE
    score_df, avg_scores = comp.calculate_bleu_rouge(ref_col, gen_col)
    print("\nAverage scores:", avg_scores)

    # Plots
    comp.plot_length_distribution([ref_col, gen_col])
    comp.plot_score_comparison(score_df)

    # Save results
    comp.export_comparison(score_df, "summary_scores.csv")


if __name__ == "__main__":
    main()
