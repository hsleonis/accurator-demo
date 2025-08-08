import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import warnings
from nltk import download
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Attention
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from gensim.models.fasttext import FastText

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", None)


class Seq2SeqAttentionLSTM:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.data = None
        self.all_stopwords = None
        self.tokenizer_x = None
        self.tokenizer_y = None
        self.encoder_input_data = None
        self.decoder_input_data = None
        self.decoder_target_data = None
        self.model = None

    def load_data(self):
        """Load and clean dataset."""
        self.data = pd.read_csv(self.csv_path)
        self.data.dropna(axis=0, inplace=True)

    def setup_stopwords(self):
        """Download and set stopwords."""
        download('stopwords')
        download('wordnet')
        self.all_stopwords = stopwords.words('english')

    def text_cleaner(self, text):
        """Clean text using regex and stopwords."""
        new_string = text.lower()
        new_string = re.sub(r'\([^)]*\)', '', new_string)
        new_string = re.sub('"', '', new_string)
        new_string = re.sub("[^a-zA-Z]", " ", new_string)
        new_string = re.sub('[m]{2,}', 'mm', new_string)
        new_string = re.sub(r"\'s\b", "", new_string)
        text_tokens = new_string.split()
        tokens_without_sw = [word for word in text_tokens if word not in self.all_stopwords]
        new_string = " ".join(tokens_without_sw)
        new_string = re.sub(r"\s+", " ", new_string)
        return new_string

    def clean_texts(self):
        """Apply cleaning to text and summary columns."""
        self.data['cleaned_text'] = [self.text_cleaner(t) for t in self.data['Text']]
        self.data['cleaned_summary'] = [self.text_cleaner(t) for t in self.data['Summary']]

    def visualize_length_distribution(self):
        """Plot length distribution of text and summary."""
        text_word_count = [len(t.split()) for t in self.data['cleaned_text']]
        summary_word_count = [len(s.split()) for s in self.data['cleaned_summary']]
        length_df = pd.DataFrame({'text': text_word_count, 'summary': summary_word_count})
        length_df.hist(bins=15)
        plt.show()

    def prepare_tokenizers(self, num_words_text=5000, num_words_summary=2000):
        """Fit tokenizers for text and summary."""
        self.tokenizer_x = Tokenizer(num_words=num_words_text)
        self.tokenizer_y = Tokenizer(num_words=num_words_summary)
        self.tokenizer_x.fit_on_texts(self.data['cleaned_text'])
        self.tokenizer_y.fit_on_texts(self.data['cleaned_summary'])

    def build_model(self, latent_dim=300):
        """Build the Seq2Seq model with Attention."""
        encoder_inputs = Input(shape=(None,))
        encoder_embedding = Embedding(input_dim=len(self.tokenizer_x.word_index)+1, output_dim=latent_dim)(encoder_inputs)
        encoder_lstm = LSTM(latent_dim, return_state=True, return_sequences=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

        decoder_inputs = Input(shape=(None,))
        decoder_embedding = Embedding(input_dim=len(self.tokenizer_y.word_index)+1, output_dim=latent_dim)(decoder_inputs)
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])

        attention_layer = Attention()
        attention_result = attention_layer([decoder_outputs, encoder_outputs])

        decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, attention_result])
        decoder_dense = TimeDistributed(Dense(len(self.tokenizer_y.word_index)+1, activation='softmax'))
        decoder_outputs = decoder_dense(decoder_concat_input)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy')

    def train(self, encoder_input, decoder_input, decoder_target, batch_size=64, epochs=10):
        """Train the model."""
        es = EarlyStopping(monitor='val_loss', mode='min', patience=2, verbose=1)
        self.model.fit(
            [encoder_input, decoder_input],
            decoder_target,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            callbacks=[es]
        )


def main():
    seq2seq = Seq2SeqAttentionLSTM(csv_path="abstract_summary.csv")
    seq2seq.load_data()
    seq2seq.setup_stopwords()
    seq2seq.clean_texts()
    seq2seq.visualize_length_distribution()
    seq2seq.prepare_tokenizers()
    seq2seq.build_model()
    # Training would require prepared input arrays:
    # seq2seq.train(encoder_input_data, decoder_input_data, decoder_target_data)


if __name__ == "__main__":
    main()
