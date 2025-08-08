#!/usr/bin/env python3
"""
seq2seq_encoder_word.py

Refactor of Seq2Seq_Encoder_(Word).ipynb into a portable script.

Features:
- Word-level seq2seq encoder-decoder (TensorFlow / Keras)
- Vocabulary building, tokenization, padding
- Training, model saving/loading
- Greedy inference (decode one token at a time)
- Configurable hyperparameters

Usage:
    python seq2seq_encoder_word.py --data_file abstract_mod.txt --epochs 10
"""

import os
import argparse
import json
from typing import List, Tuple, Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class Seq2SeqEncoder:
    def __init__(
        self,
        embedding_size: int = 50,
        latent_dim: int = 256,
        num_words: Optional[int] = None,
        start_token: str = "<START>",
        end_token: str = "<END>",
        oov_token: str = "<OOV>",
        pad_token: str = "<PAD>",
    ):
        """
        Initialize seq2seq parameters.

        :param embedding_size: embedding vector dimension
        :param latent_dim: LSTM latent dimension
        :param num_words: maximum vocabulary size (None = keep all)
        """
        self.embedding_size = embedding_size
        self.latent_dim = latent_dim
        self.num_words = num_words
        self.start_token = start_token
        self.end_token = end_token
        self.oov_token = oov_token
        self.pad_token = pad_token

        # Tokenizers for encoder (source) and decoder (target)
        self.src_tokenizer: Optional[Tokenizer] = None
        self.tgt_tokenizer: Optional[Tokenizer] = None

        # Vocabulary sizes (filled after fitting tokenizers)
        self.src_vocab_size: Optional[int] = None
        self.tgt_vocab_size: Optional[int] = None

        # Model components
        self.model: Optional[Model] = None
        self.encoder_model: Optional[Model] = None
        self.decoder_model: Optional[Model] = None

    # -----------------------
    # Data loading / parsing
    # -----------------------
    def load_parallel_sentences(self, file_path: str, sep: str = ",", nrows: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """
        Load a simple text file with pairs of sentences separated by `sep`.
        The original notebook read a file and split by commas. This function
        expects each pair like: "source<sep>target" per entry OR file that
        contains alternating sentences depending on your file format.

        If your `abstract_mod.txt` contains a single long line with commas,
        you can instead read and chunk appropriately before calling this.

        Returns (sources, targets)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()

        # Basic heuristic: if file contains the separator many times, split into pairs
        parts = raw.split(sep)
        # If file is structured as alternating source,target,source,target...
        if len(parts) % 2 == 0 and len(parts) > 2:
            sources = [parts[i].strip() for i in range(0, len(parts), 2)]
            targets = [parts[i].strip() for i in range(1, len(parts), 2)]
        else:
            # Fallback: assume each line contains "src \t tgt"
            lines = raw.splitlines()
            sources = []
            targets = []
            for ln in lines:
                if "\t" in ln:
                    s, t = ln.split("\t", 1)
                    sources.append(s.strip())
                    targets.append(t.strip())
                else:
                    # If the file is simply a list of sentences, pair each sentence with a short target (echo)
                    # This is a fallback and should be replaced by correct parsing for your dataset
                    sources.append(ln.strip())
                    targets.append(ln.strip())

        if nrows is not None:
            sources = sources[:nrows]
            targets = targets[:nrows]
        return sources, targets

    # -----------------------
    # Tokenization / vocab
    # -----------------------
    def build_tokenizers(self, sources: List[str], targets: List[str]):
        """
        Build and fit word-level tokenizers for the source and target languages.
        Adds start/end tokens to the target sequences when tokenizing.
        """
        # Source tokenizer
        self.src_tokenizer = Tokenizer(
            num_words=self.num_words,
            oov_token=self.oov_token,
            filters=''  # we will not remove punctuation automatically; rely on input cleaning
        )
        self.src_tokenizer.fit_on_texts(sources)

        # Target tokenizer: include start/end tokens explicitly in texts before fitting
        targets_with_tokens = [f"{self.start_token} {t} {self.end_token}" for t in targets]
        self.tgt_tokenizer = Tokenizer(
            num_words=self.num_words,
            oov_token=self.oov_token,
            filters=''
        )
        self.tgt_tokenizer.fit_on_texts(targets_with_tokens)

        # Reserve an index for pad token if needed (Keras uses 0 for padding by default)
        # Ensure pad token not in tokenizer word_index; we'll treat 0 as pad.
        self.src_vocab_size = min(self.num_words or (len(self.src_tokenizer.word_index) + 1),
                                  len(self.src_tokenizer.word_index) + 1)
        self.tgt_vocab_size = min(self.num_words or (len(self.tgt_tokenizer.word_index) + 1),
                                  len(self.tgt_tokenizer.word_index) + 1)

        # +1 because tokenizers index start at 1; 0 is reserved for padding
        # We'll use vocab_size values directly with Embedding input_dim
        # For convenience expose them
        return self.src_vocab_size, self.tgt_vocab_size

    def texts_to_sequences(self, sources: List[str], targets: List[str], max_encoder_len: Optional[int] = None, max_decoder_len: Optional[int] = None
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        """
        Convert raw texts to padded sequences for encoder_input_data, decoder_input_data, decoder_target_data (one-hot).
        Returns (encoder_input_data, decoder_input_data, decoder_target_data, max_encoder_len, max_decoder_len)
        """
        if self.src_tokenizer is None or self.tgt_tokenizer is None:
            raise ValueError("Tokenizers not built. Call build_tokenizers() first.")

        # Add start/end tokens to targets for decoding
        targets_with_tokens = [f"{self.start_token} {t} {self.end_token}" for t in targets]

        encoder_sequences = self.src_tokenizer.texts_to_sequences(sources)
        decoder_sequences = self.tgt_tokenizer.texts_to_sequences(targets_with_tokens)

        # Determine maximum lengths
        max_encoder_len = max(len(seq) for seq in encoder_sequences) if max_encoder_len is None else max_encoder_len
        max_decoder_len = max(len(seq) for seq in decoder_sequences) if max_decoder_len is None else max_decoder_len

        encoder_input_data = pad_sequences(encoder_sequences, maxlen=max_encoder_len, padding='post')
        decoder_input_data = pad_sequences([seq[:-1] for seq in decoder_sequences], maxlen=max_decoder_len - 1, padding='post')
        # decoder_target_data is decoder_sequences shifted left (the next token), we will use sparse categorical crossentropy
        decoder_target_data = pad_sequences([seq[1:] for seq in decoder_sequences], maxlen=max_decoder_len - 1, padding='post')

        return encoder_input_data, decoder_input_data, decoder_target_data, max_encoder_len, max_decoder_len

    # -----------------------
    # Model building
    # -----------------------
    def build_model(self, src_vocab_size: int, tgt_vocab_size: int, max_encoder_len: int, max_decoder_len: int):
        """
        Build Keras encoder-decoder model for training with teacher forcing.
        We'll use Embedding -> LSTM encoder and Embedding -> LSTM decoder with Dense softmax.
        The training model takes encoder_input and decoder_input and outputs decoder tokens.
        """
        # +1 not necessary if vocab sizes already include the padding index handling
        encoder_inputs = Input(shape=(None,), name="encoder_inputs")
        enc_emb = Embedding(input_dim=src_vocab_size, output_dim=self.embedding_size, mask_zero=True, name="encoder_embedding")(encoder_inputs)
        encoder_lstm = LSTM(self.latent_dim, return_state=True, name="encoder_lstm")
        _, state_h, state_c = encoder_lstm(enc_emb)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None,), name="decoder_inputs")
        dec_emb_layer = Embedding(input_dim=tgt_vocab_size, output_dim=self.embedding_size, mask_zero=True, name="decoder_embedding")
        dec_emb = dec_emb_layer(decoder_inputs)
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
        decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
        decoder_dense = Dense(tgt_vocab_size, activation='softmax', name="decoder_dense")
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model

        # Build inference encoder model (to get states)
        encoder_model = Model(encoder_inputs, encoder_states)
        self.encoder_model = encoder_model

        # Build inference decoder model
        # Inputs for states
        decoder_state_input_h = Input(shape=(self.latent_dim,), name="decoder_state_input_h")
        decoder_state_input_c = Input(shape=(self.latent_dim,), name="decoder_state_input_c")
        dec_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        dec_emb2 = dec_emb_layer(decoder_inputs)
        decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=dec_states_inputs)
        decoder_outputs2 = decoder_dense(decoder_outputs2)
        decoder_model = Model([decoder_inputs] + dec_states_inputs, [decoder_outputs2, state_h2, state_c2])
        self.decoder_model = decoder_model

        return model

    # -----------------------
    # Training / Save / Load
    # -----------------------
    def train(
        self,
        encoder_input_data: np.ndarray,
        decoder_input_data: np.ndarray,
        decoder_target_data: np.ndarray,
        epochs: int = 20,
        batch_size: int = 64,
        validation_split: float = 0.1,
        save_dir: str = "./seq2seq_model"
    ):
        """
        Train the seq2seq model. decoder_target_data should be integer token ids (not one-hot).
        Keras expects targets shaped (batch, timesteps) for sparse_categorical_crossentropy, but
        the model outputs (batch, timesteps, vocab). We'll expand dims of target to (batch, timesteps, 1).
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Prepare targets for sparse categorical crossentropy: expand the last axis
        decoder_target_data_exp = np.expand_dims(decoder_target_data, -1)  # shape (samples, timesteps, 1)

        history = self.model.fit(
            [encoder_input_data, decoder_input_data],
            decoder_target_data_exp,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            verbose=1
        )

        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, "seq2seq_training_model.h5")
        self.model.save(model_path)
        # Also save tokenizers and params
        params = {
            "embedding_size": self.embedding_size,
            "latent_dim": self.latent_dim,
            "num_words": self.num_words,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "oov_token": self.oov_token,
            "pad_token": self.pad_token
        }
        with open(os.path.join(save_dir, "params.json"), "w", encoding="utf-8") as fp:
            json.dump(params, fp)
        # Save tokenizers
        if self.src_tokenizer:
            with open(os.path.join(save_dir, "src_tokenizer.json"), "w", encoding="utf-8") as tfp:
                tfp.write(self.src_tokenizer.to_json())
        if self.tgt_tokenizer:
            with open(os.path.join(save_dir, "tgt_tokenizer.json"), "w", encoding="utf-8") as tfp:
                tfp.write(self.tgt_tokenizer.to_json())

        print(f"Model and tokenizers saved to {save_dir}")
        return history

    def load(self, save_dir: str = "./seq2seq_model"):
        """
        Load model and tokenizers from disk and rebuild inference models.
        """
        model_path = os.path.join(save_dir, "seq2seq_training_model.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        # Load training model to get weights and layer objects
        self.model = load_model(model_path, compile=False)
        # Load tokenizers
        src_tokenizer_path = os.path.join(save_dir, "src_tokenizer.json")
        tgt_tokenizer_path = os.path.join(save_dir, "tgt_tokenizer.json")
        if os.path.exists(src_tokenizer_path):
            with open(src_tokenizer_path, "r", encoding="utf-8") as f:
                self.src_tokenizer = Tokenizer.from_json(f.read())
        if os.path.exists(tgt_tokenizer_path):
            with open(tgt_tokenizer_path, "r", encoding="utf-8") as f:
                self.tgt_tokenizer = Tokenizer.from_json(f.read())

        # Reconstruct encoder and decoder inference models from the loaded training model
        # Assumes the layer names are the same as in build_model
        # Find relevant layers
        encoder_inputs = self.model.get_layer("encoder_inputs").input if "encoder_inputs" in [l.name for l in self.model.layers] else self.model.input[0]
        # Try to reconstruct encoder states output
        # For safety, rebuild architecture using saved params if tokenizers exist
        # We will rebuild inference models by re-calling build_model with saved sizes if possible
        if self.src_tokenizer and self.tgt_tokenizer:
            src_vocab_size = len(self.src_tokenizer.word_index) + 1
            tgt_vocab_size = len(self.tgt_tokenizer.word_index) + 1
            # Need some max lengths; we'll set arbitrary but decoder/encoder layers are shape-agnostic
            self.build_model(src_vocab_size, tgt_vocab_size, max_encoder_len=10, max_decoder_len=10)
            # Now load weights from training model into self.model's layers
            try:
                self.model.load_weights(model_path)
            except Exception:
                # If direct load fails, just keep loaded training model for inference via different API
                pass

        print(f"Loaded model from {save_dir}")

    # -----------------------
    # Inference (greedy decode)
    # -----------------------
    def decode_sequence(self, input_seq: np.ndarray, max_decoder_len: int = 20) -> str:
        """
        Given encoder input sequence, produce decoded target sentence (greedy decoding).
        """
        if self.encoder_model is None or self.decoder_model is None:
            raise ValueError("Inference models not built. Ensure build_model() was called and inference models set.")

        # Encode the input as state vectors
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1 with only the start token
        # get start token id
        if self.tgt_tokenizer is None:
            raise ValueError("Target tokenizer not available.")
        start_id = self.tgt_tokenizer.word_index.get(self.start_token)
        end_id = self.tgt_tokenizer.word_index.get(self.end_token)

        target_seq = np.array([[start_id]])  # shape (1,1)

        stop_condition = False
        decoded_tokens = []
        for _ in range(max_decoder_len):
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
            # output_tokens shape: (1, 1, vocab)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            if sampled_token_index == 0:
                # padding or unknown; break if cannot proceed
                break
            sampled_word = None
            # find word from index
            for w, idx in self.tgt_tokenizer.word_index.items():
                if idx == sampled_token_index:
                    sampled_word = w
                    break
            if sampled_word is None:
                break
            if sampled_word == self.end_token:
                break
            decoded_tokens.append(sampled_word)
            # Update target_seq (of length 1)
            target_seq = np.array([[sampled_token_index]])
            # Update states
            states_value = [h, c]

        return " ".join(decoded_tokens)

    # -----------------------
    # Utility
    # -----------------------
    @staticmethod
    def sequence_to_text(sequence: List[int], tokenizer: Tokenizer) -> str:
        inv_map = {v: k for k, v in tokenizer.word_index.items()}
        words = [inv_map.get(idx, "") for idx in sequence if idx != 0]
        return " ".join(w for w in words if w)

# -----------------------
# Example main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Seq2Seq word-level encoder/decoder")
    parser.add_argument("--data_file", type=str, default="abstract_mod.txt", help="Path to raw data file")
    parser.add_argument("--sep", type=str, default=",", help="Separator used in raw file to split pairs")
    parser.add_argument("--nrows", type=int, default=4000, help="Number of pairs to use (if applicable)")
    parser.add_argument("--embedding_size", type=int, default=50)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_dir", type=str, default="./seq2seq_model")
    args = parser.parse_args()

    # Instantiate
    s2s = Seq2SeqEncoder(embedding_size=args.embedding_size, latent_dim=args.latent_dim)

    # Load parallel sentences (attempt common formats)
    sources, targets = s2s.load_parallel_sentences(args.data_file, sep=args.sep, nrows=args.nrows)
    print(f"Loaded {len(sources)} pairs")

    # Optionally do simple cleaning (lowercasing, strip)
    sources = [s.strip().lower() for s in sources]
    targets = [t.strip().lower() for t in targets]

    # Build tokenizers/vocab
    src_vocab_size, tgt_vocab_size = s2s.build_tokenizers(sources, targets)
    print(f"Source vocab size: {src_vocab_size}, Target vocab size: {tgt_vocab_size}")

    # Convert texts to sequences & pad
    encoder_input_data, decoder_input_data, decoder_target_data, max_enc_len, max_dec_len = s2s.texts_to_sequences(
        sources, targets)
    print(f"Max encoder len: {max_enc_len}, Max decoder len: {max_dec_len}")

    # Build model
    model = s2s.build_model(src_vocab_size, tgt_vocab_size, max_enc_len, max_dec_len)
    model.summary()

    # Train
    s2s.train(
        encoder_input_data,
        decoder_input_data,
        decoder_target_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )

    # Demo inference on first example
    if len(encoder_input_data) > 0:
        input_seq = encoder_input_data[0:1]
        decoded = s2s.decode_sequence(input_seq, max_decoder_len=max_dec_len)
        print("Source:", sources[0][:200])
        print("Target:", targets[0][:200])
        print("Decoded:", decoded)

if __name__ == "__main__":
    main()
