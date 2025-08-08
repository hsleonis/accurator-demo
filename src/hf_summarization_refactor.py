"""
hf_summarization_refactor.py

Refactor of a HuggingFace-based summarization Notebook into:
- HuggingFaceSummarizer class (load, preprocess, tokenize, train, evaluate, infer)
- CLI entrypoint for common workflows

Example usage:
    python hf_summarization_refactor.py --csv data/title_text.csv --text_col text --summary_col summary \
        --model_checkpoint facebook/bart-large-cnn --do_train --output_dir ./models/bart_summarizer
"""

from typing import Optional, Dict, Any, Sequence, Callable
import os
import re
import argparse
import logging
import json

import numpy as np
import pandas as pd

import torch
from datasets import Dataset, load_metric, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Try imports with fallback notes
try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except Exception:
    _HAS_BS4 = False

try:
    import evaluate
except Exception:
    evaluate = None

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HuggingFaceSummarizer:
    def __init__(
        self,
        model_checkpoint: str = "facebook/bart-large-cnn",
        max_input_length: int = 1024,
        max_target_length: int = 128,
        prefix: Optional[str] = None,
        device: Optional[str] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize tokenizer and model.

        :param model_checkpoint: HF model checkpoint ID
        :param max_input_length: max tokens for encoder input
        :param max_target_length: max tokens for decoder target / summaries
        :param prefix: optional text prefix for models that use it (e.g. "summarize: " for T5)
        :param device: device string ('cuda'/'cpu') or None to auto-select
        """

        self.model_checkpoint = model_checkpoint
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.prefix = prefix if prefix is not None else ""
        self.tokenizer_kwargs = tokenizer_kwargs or {}

        # Load tokenizer and model
        logger.info(f"Loading tokenizer '{model_checkpoint}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, **self.tokenizer_kwargs)

        logger.info(f"Loading model '{model_checkpoint}'...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

        # Select device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")

        # Data collator for seq2seq
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        # Prepare rouge if available
        self.rouge = None
        if evaluate is not None:
            try:
                self.rouge = evaluate.load("rouge")
            except Exception:
                self.rouge = None
        else:
            # try fallback import
            try:
                self.rouge = load_metric("rouge")
            except Exception:
                self.rouge = None

    # -----------------------
    # Data loading / cleaning
    # -----------------------
    def load_csv(self, csv_path: str, text_col: str = "text", summary_col: Optional[str] = None, nrows: Optional[int] = None) -> Dataset:
        """
        Load CSV into an Arrow Dataset object (datasets.Dataset).
        Expects text column and optional summary column.

        :param csv_path: path to CSV
        :param text_col: column name for source text
        :param summary_col: column name for target summary (if training)
        :param nrows: optional subset rows to load (for quick experiments)
        :return: datasets.Dataset
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} not found")

        df = pd.read_csv(csv_path, nrows=nrows)
        # Ensure the columns exist
        if text_col not in df.columns:
            raise KeyError(f"text_col '{text_col}' not in CSV columns: {df.columns.tolist()}")
        if summary_col is not None and summary_col not in df.columns:
            raise KeyError(f"summary_col '{summary_col}' not in CSV columns: {df.columns.tolist()}")

        # Create dataset
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        return dataset

    def clean_html(self, raw: str) -> str:
        """Remove HTML tags and collapse whitespace."""
        if raw is None:
            return ""
        s = str(raw)
        if _HAS_BS4:
            # Use BeautifulSoup if available (cleaner)
            try:
                text = BeautifulSoup(s, "html.parser").get_text(separator=" ")
            except Exception:
                text = re.sub(r"<.*?>", " ", s)
        else:
            # Fallback simple regex
            text = re.sub(r"<[^>]+>", " ", s)
        # Normalize whitespace and remove escape sequences
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize_unicode(self, text: str) -> str:
        """Basic unicode normalization (remove odd escapes)."""
        if text is None:
            return ""
        # Remove control characters and redundant spaces
        text = re.sub(r"[\r\n\t]+", " ", str(text))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # -----------------------
    # Tokenization / mapping
    # -----------------------
    def preprocess_function(
        self,
        examples,
        text_column: str = "text",
        summary_column: Optional[str] = None,
        max_input_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
    ):
        """
        Tokenize input texts and target summaries for seq2seq training.
        This is intended to be used with dataset.map(preprocessing_function, batched=True).
        """
        max_input_length = max_input_length or self.max_input_length
        max_target_length = max_target_length or self.max_target_length

        inputs = [self.prefix + self.normalize_unicode(self.clean_html(t)) for t in examples[text_column]]
        model_inputs = self.tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")

        labels = None
        if summary_column is not None:
            targets = [self.normalize_unicode(self.clean_html(s)) if s is not None else "" for s in examples[summary_column]]
            # Tokenize targets with the tokenizer as labels
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")

            # Replace tokenizer.pad_token_id in labels with -100 so that they're ignored in loss
            if labels is not None:
                label_ids = labels["input_ids"]
                label_ids = [
                    [(l if l != self.tokenizer.pad_token_id else -100) for l in label_seq]
                    for label_seq in label_ids
                ]
                model_inputs["labels"] = label_ids

        return model_inputs

    def tokenize_dataset(
        self,
        dataset,
        text_col: str = "text",
        summary_col: Optional[str] = None,
        batched: bool = True,
        remove_columns: Optional[Sequence[str]] = None,
    ) -> Dataset:
        """
        Tokenize a datasets.Dataset object and return tokenized dataset (ready for Trainer).
        """
        remove_columns = remove_columns or []
        preprocess = lambda examples: self.preprocess_function(
            examples,
            text_column=text_col,
            summary_column=summary_col,
        )
        tokenized = dataset.map(preprocess, batched=batched, remove_columns=remove_columns or dataset.column_names)
        return tokenized

    # -----------------------
    # Training / Evaluation
    # -----------------------
    def train(
        self,
        train_dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: str = "./hf_summarizer",
        training_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Train model using HF Trainer API.

        :param train_dataset: tokenized training Dataset
        :param eval_dataset: tokenized evaluation Dataset (optional)
        :param output_dir: output directory for model / logs
        :param training_args: dict of training args (overrides defaults)
        :return: Trainer object
        """
        default_args = {
            "output_dir": output_dir,
            "evaluation_strategy": "steps" if eval_dataset is not None else "no",
            "save_strategy": "steps",
            "save_steps": 500,
            "eval_steps": 500 if eval_dataset is not None else None,
            "learning_rate": 5e-5,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "num_train_epochs": 3,
            "weight_decay": 0.01,
            "fp16": torch.cuda.is_available(),
            "push_to_hub": False,
            "logging_steps": 100,
            "report_to": "none",
            "predict_with_generate": True,
        }
        if training_args:
            default_args.update(training_args)

        # Create TrainingArguments object
        training_arguments = Seq2SeqTrainingArguments(**{k: v for k, v in default_args.items() if v is not None})

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        logger.info("Starting training...")
        trainer.train()
        logger.info("Training finished. Saving model...")
        trainer.save_model(output_dir)
        return trainer

    def postprocess_texts(self, preds: Sequence[str], labels: Optional[Sequence[str]] = None):
        """Clean predicted and label texts before metric calculation."""
        cleaned_preds = [re.sub(r"\s+", " ", p).strip() for p in preds]
        cleaned_labels = None
        if labels is not None:
            cleaned_labels = [re.sub(r"\s+", " ", l).strip() for l in labels]
        return cleaned_preds, cleaned_labels

    def evaluate(self, tokenized_dataset, text_col: str = "text", summary_col: str = "summary", batch_size: int = 8):
        """
        Evaluate model on tokenized dataset. Returns rouge scores.
        tokenized_dataset should be tokenized with tokenizer (and include raw columns if you preserved them).
        """
        # Use trainer for generation conveniently
        training_args = Seq2SeqTrainingArguments(
            output_dir="./tmp_eval",
            per_device_eval_batch_size=batch_size,
            predict_with_generate=True,
            logging_strategy="no",
            report_to="none",
        )
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        logger.info("Running prediction/generation for evaluation...")
        preds_output = trainer.predict(tokenized_dataset)
        # preds_output.predictions shape depends on model and settings: decode to text
        if isinstance(preds_output.predictions, tuple):
            pred_ids = preds_output.predictions[0]
        else:
            pred_ids = preds_output.predictions

        decoded_preds = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        # If the dataset includes labels in tokenized 'labels', decode them too
        # But trainer.predict may already have labels
        labels_ids = preds_output.label_ids
        if labels_ids is not None:
            # Replace -100 with pad_token_id for decoding
            labels_ids = np.where(labels_ids != -100, labels_ids, self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        else:
            decoded_labels = None

        decoded_preds, decoded_labels = self.postprocess_texts(decoded_preds, decoded_labels)

        # Compute rouge using evaluate (preferred) or datasets.load_metric fallback
        if self.rouge is not None:
            try:
                results = self.rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
                # rouge returns keys like rouge1, rouge2, rougeL
                # Convert values to percentages
                results = {k: float(v.mid.fmeasure * 100) if hasattr(v, "mid") else float(v * 100) for k, v in results.items()}
            except Exception as e:
                logger.warning(f"ROUGE computation failed: {e}")
                results = {}
        else:
            results = {}

        return {"preds": decoded_preds, "labels": decoded_labels, "rouge": results}

    # -----------------------
    # Inference
    # -----------------------
    def summarize(self, text: str, num_beams: int = 4, max_length: Optional[int] = None, min_length: Optional[int] = None, length_penalty: float = 2.0, no_repeat_ngram_size: int = 3):
        """
        Summarize a single string input (returns string).
        """
        max_length = max_length or self.max_target_length
        inputs = self.prefix + self.normalize_unicode(self.clean_html(text))
        tokenized = self.tokenizer([inputs], max_length=self.max_input_length, truncation=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized.get("attention_mask", None),
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True,
            )
        decoded = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return decoded.strip()


# -----------------------
# CLI / main
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="HuggingFace summarization refactor CLI")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV with texts")
    parser.add_argument("--text_col", type=str, default="text", help="Column name for input texts")
    parser.add_argument("--summary_col", type=str, default=None, help="Column name for target summaries (optional)")
    parser.add_argument("--model_checkpoint", type=str, default="facebook/bart-large-cnn", help="HF model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./hf_summarizer", help="Directory to save trained model")
    parser.add_argument("--do_train", action="store_true", help="Run training")
    parser.add_argument("--do_eval", action="store_true", help="Run evaluation (predict + rouge)")
    parser.add_argument("--do_predict", action="store_true", help="Run single-text predictions as demo")
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--nrows", type=int, default=None, help="Load only nrows from CSV (for quick tests)")
    parser.add_argument("--train_frac", type=float, default=0.9, help="Fraction to use for training if splitting")
    parser.add_argument("--prefix", type=str, default=None, help="Optional prefix (e.g. 'summarize: ')")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    summarizer = HuggingFaceSummarizer(
        model_checkpoint=args.model_checkpoint,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        prefix=args.prefix,
    )

    dataset = summarizer.load_csv(args.csv, text_col=args.text_col, summary_col=args.summary_col, nrows=args.nrows)

    # If summary_col is present and user wants to train, we need to split and tokenize
    if args.do_train or args.do_eval:
        # Split into train/validation if summary_col provided; otherwise can't train
        if args.summary_col is None:
            raise ValueError("Training/eval requires --summary_col to be specified (target summaries).")

        # Optionally split dataset into train/validation
        ds = dataset.train_test_split(test_size=1 - args.train_frac) if args.train_frac < 1.0 else DatasetDict({"train": dataset})
        train_ds = ds["train"] if "train" in ds else ds["train"]
        val_ds = ds["test"] if "test" in ds else None

        # Tokenize (remove original text columns to keep only tokenized ones for Trainer)
        remove_cols = dataset.column_names
        tokenized_train = summarizer.tokenize_dataset(train_ds, text_col=args.text_col, summary_col=args.summary_col, remove_columns=remove_cols)
        tokenized_val = summarizer.tokenize_dataset(val_ds, text_col=args.text_col, summary_col=args.summary_col, remove_columns=remove_cols) if val_ds is not None else None

        if args.do_train:
            training_args = {
                "output_dir": args.output_dir,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "per_device_eval_batch_size": args.per_device_eval_batch_size,
                "num_train_epochs": args.num_train_epochs,
                "evaluation_strategy": "steps" if tokenized_val is not None else "no",
                "logging_steps": 100,
                "save_steps": 500,
                "eval_steps": 500 if tokenized_val is not None else None,
            }
            trainer = summarizer.train(tokenized_train, eval_dataset=tokenized_val, output_dir=args.output_dir, training_args=training_args)
            logger.info(f"Model trained and saved to {args.output_dir}")

        if args.do_eval:
            eval_dataset_tokenized = tokenized_val if tokenized_val is not None else tokenized_train
            eval_result = summarizer.evaluate(eval_dataset_tokenized)
            logger.info("Evaluation results (ROUGE %):")
            logger.info(json.dumps(eval_result.get("rouge", {}), indent=2))
            # Optionally save predictions
            out_preds_path = os.path.join(args.output_dir, "eval_predictions.json")
            with open(out_preds_path, "w", encoding="utf-8") as wf:
                json.dump({"preds": eval_result.get("preds", []), "labels": eval_result.get("labels", [])}, wf, ensure_ascii=False, indent=2)
            logger.info(f"Saved eval preds/labels to {out_preds_path}")

    # Single-text predict demo
    if args.do_predict:
        # Use raw dataset first row as demo if available
        raw_first = dataset[0]
        text_example = raw_first.get(args.text_col) if raw_first is not None else ""
        logger.info("Running demo inference on first row of CSV...")
        summary = summarizer.summarize(text_example)
        logger.info("Input (first 300 chars):\n" + str(text_example)[:300])
        logger.info("Generated summary:\n" + summary)

    logger.info("Done.")


if __name__ == "__main__":
    main()
