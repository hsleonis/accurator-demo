"""
language_analysis_refactor.py

Refactored from: Language_analysis_(Springer).ipynb
- Organizes code into a LanguageAnalysis class and helper functions
- Preserves original plotting behavior but makes it configurable
- Usage:
    from language_analysis_refactor import LanguageAnalysis
    analysis = LanguageAnalysis("lang_report_1.csv")
    counts = analysis.count_languages()
    analysis.plot_histogram()
    analysis.plot_pie_chart(langs=["EN","AF","DE","RO","FR","CY"])
"""

from typing import Dict, Optional, Sequence
import pandas as pd
import matplotlib.pyplot as plt
import os

class LanguageAnalysis:
    def __init__(self, file_path: str, lang_column: str = "Lang", encoding: str = "utf-8"):
        """
        Initialize the LanguageAnalysis object and load data.

        :param file_path: Path to the CSV file containing language column.
        :param lang_column: Column name that contains language codes.
        :param encoding: File encoding for reading CSV.
        """
        self.file_path = file_path
        self.lang_column = lang_column
        self.encoding = encoding
        self.data = self._load_data(file_path)

    def _load_data(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        try:
            df = pd.read_csv(file_path, encoding=self.encoding)
        except Exception:
            df = pd.read_csv(file_path)
        if self.lang_column not in df.columns:
            raise KeyError(f"Language column '{self.lang_column}' not found in CSV columns: {df.columns.tolist()}")
        return df.copy()

    def count_languages(self, normalize: bool = False) -> Dict[str, int]:
        """
        Count occurrences of each language code.

        :param normalize: If True, return relative frequencies (sums to 1.0).
        :return: dict mapping language code -> count (or proportion).
        """
        counts = self.data[self.lang_column].fillna("").astype(str).str.strip().value_counts(dropna=False)
        if normalize:
            return counts.div(counts.sum()).to_dict()
        return counts.to_dict()

    def get_non_english_count(self, english_codes: Sequence[str] = ("EN", "en", "en-US", "en_GB")) -> int:
        """
        Return number of rows whose language is not English (any of the provided english_codes).
        """
        series = self.data[self.lang_column].fillna("").astype(str).str.strip()
        mask = ~series.isin([c for c in english_codes])
        return int(mask.sum())

    def plot_histogram(self, top_n: Optional[int] = None, figsize: tuple = (10, 6),
                       title: Optional[str] = None, savepath: Optional[str] = None):
        """
        Plot a bar chart of language counts.

        :param top_n: If provided, show only the top N languages by count.
        :param figsize: Figure size.
        :param title: Optional title.
        :param savepath: If provided, save figure to this path.
        """
        counts = pd.Series(self.count_languages()).sort_values(ascending=False)
        if top_n is not None:
            counts = counts.head(top_n)
        fig, ax = plt.subplots(figsize=figsize)
        counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Language")
        ax.set_ylabel("Count")
        if title:
            ax.set_title(title)
        plt.tight_layout()
        if savepath:
            fig.savefig(savepath)
        else:
            plt.show()
        plt.close(fig)

    def plot_pie_chart(self, langs: Optional[Sequence[str]] = None, explode_others: bool = True,
                       figsize: tuple = (8, 8), title: Optional[str] = None, savepath: Optional[str] = None):
        """
        Plot a pie chart for given languages. If langs is None, plot all languages.
        If langs is provided, languages not in the list can be lumped into 'Other'.

        :param langs: Sequence of language codes to show explicitly (case sensitive).
        :param explode_others: Whether to separate 'Other' slice in the pie chart.
        :param figsize: Figure size.
        :param title: Optional title.
        :param savepath: If provided, save figure to this path.
        """
        counts = pd.Series(self.count_languages()).sort_values(ascending=False)
        if langs is None:
            labels = counts.index.tolist()
            sizes = counts.values.tolist()
        else:
            langs = list(langs)
            selected = counts[counts.index.isin(langs)]
            other = counts[~counts.index.isin(langs)].sum()
            labels = selected.index.tolist()
            sizes = selected.values.tolist()
            if other > 0:
                labels.append("Other")
                sizes.append(int(other))
        fig, ax = plt.subplots(figsize=figsize)
        ax.pie(sizes, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        if title:
            ax.set_title(title)
        ax.legend(labels, loc="best")
        plt.tight_layout()
        if savepath:
            fig.savefig(savepath)
        else:
            plt.show()
        plt.close(fig)

    def filter_languages(self, include: Optional[Sequence[str]] = None,
                         exclude: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """
        Return a filtered DataFrame including or excluding certain language codes.
        """
        df = self.data.copy()
        if include is not None:
            include = set([str(i).strip() for i in include])
            df = df[df[self.lang_column].astype(str).str.strip().isin(include)]
        if exclude is not None:
            exclude = set([str(e).strip() for e in exclude])
            df = df[~df[self.lang_column].astype(str).str.strip().isin(exclude)]
        return df

def load_data(path: str, lang_column: str = "Lang") -> pd.DataFrame:
    """
    Utility function to load CSV and return DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path)
    if lang_column not in df.columns:
        raise KeyError(f"Language column '{lang_column}' not found in CSV")
    return df

def main(csv_path: str = "lang_report_1.csv"):
    analysis = LanguageAnalysis(csv_path)
    print("Total rows:", len(analysis.data))
    print("Language counts (top 10):")
    counts = analysis.count_languages()
    for i, (k, v) in enumerate(sorted(counts.items(), key=lambda x: -x[1])):
        if i >= 10:
            break
        print(f"  {k}: {v}")
    non_en = analysis.get_non_english_count()
    print(f"Non-English rows: {non_en}")

    # Example plots
    try:
        analysis.plot_histogram(top_n=20, title="Top 20 languages")
        analysis.plot_pie_chart(langs=["EN", "AF", "DE", "RO", "FR", "CY"],
                                title="Selected languages vs Others")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()
