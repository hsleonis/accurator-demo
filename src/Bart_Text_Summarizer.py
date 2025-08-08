import pandas as pd
from fastai.text.all import *
from transformers import *
from blurr.data.all import *
from blurr.modeling.all import *
from bleu import list_bleu
from bs4 import BeautifulSoup


class BartTextSummarizer:
    def __init__(self, csv_path: str, pretrained_model_name: str = "facebook/bart-large"):
        self.csv_path = csv_path
        self.pretrained_model_name = pretrained_model_name
        self.df = None
        self.hf_arch = None
        self.hf_config = None
        self.hf_tokenizer = None
        self.hf_model = None
        self.hf_batch_tfm = None
        self.dls = None
        self.learn = None

    def load_data(self):
        """Load the dataset from CSV."""
        self.df = pd.read_csv(self.csv_path)

    def setup_model(self):
        """Load BART model and tokenizer."""
        self.hf_arch, self.hf_config, self.hf_tokenizer, self.hf_model = BLURR.get_hf_objects(
            self.pretrained_model_name,
            model_cls=BartForConditionalGeneration
        )

    def prepare_transforms(self):
        """Prepare the before-batch transforms for summarization."""
        text_gen_kwargs = default_text_gen_kwargs(self.hf_config, self.hf_model, task='summarization')
        self.hf_batch_tfm = HF_Seq2SeqBeforeBatchTransform(
            self.hf_arch, self.hf_config, self.hf_tokenizer, self.hf_model,
            task='summarization',
            text_gen_kwargs=text_gen_kwargs
        )

    def create_dataloaders(self, text_col="Text", target_col="Abstract", bs=2):
        """Create DataLoaders for training."""
        blocks = (HF_Seq2SeqBlock(before_batch_tfm=self.hf_batch_tfm), noop)
        dblock = DataBlock(
            blocks=blocks,
            get_x=ColReader(text_col),
            get_y=ColReader(target_col),
            splitter=RandomSplitter()
        )
        self.dls = dblock.dataloaders(self.df, bs=bs)

    def build_learner(self):
        """Build the FastAI learner with custom metrics."""
        seq2seq_metrics = {
            'bertscore': {
                'compute_kwargs': {'lang': 'en'},
                'returns': ["precision", "recall", "f1"]
            }
        }
        model = HF_BaseModelWrapper(self.hf_model)
        learn_cbs = [HF_BaseModelCallback]
        fit_cbs = [HF_Seq2SeqMetricsCallback(custom_metrics=seq2seq_metrics)]
        self.learn = Learner(
            self.dls,
            model,
            opt_func=Adam,
            loss_func=CrossEntropyLossFlat(),
            cbs=learn_cbs + fit_cbs
        )

    def train(self, epochs=1, lr=2e-5):
        """Train the summarization model."""
        self.learn.fit_one_cycle(epochs, lr)

    def summarize(self, text: str):
        """Generate summary for a single text input."""
        return self.learn.blurr_generate(text)


def main():
    summarizer = BartTextSummarizer(csv_path="text_abstract.csv")
    summarizer.load_data()
    summarizer.setup_model()
    summarizer.prepare_transforms()
    summarizer.create_dataloaders()
    summarizer.build_learner()

    # Example training
    summarizer.train(epochs=1)

    # Example summarization
    example_text = summarizer.df["Text"].iloc[0]
    summary = summarizer.summarize(example_text)
    print("Generated Summary:", summary)


if __name__ == "__main__":
    main()
