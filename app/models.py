import pickle
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import sentencepiece


class SummarizerModels:
    """
    Contains all summarization models
    """
    def __init__(self, model_name, length=250, max_length=100, min_length=30):
        self.model_name = model_name.lower()
        self.text_length = length
        self.max_length = max_length
        self.min_length = min_length
        self.beams = 4
        self.early_stop = True
        self.truncate = True
        self.ngrams = 2

    def run(self, text):
        # get model
        func = getattr(SummarizerModels, 'model_{}'.format(self.model_name), None)

        if func is None:
            raise Exception(f"{self.model_name} model not found.")

        # run model
        return func(self, text)

    def model_demo1(self, text):
        data = text.split()
        if len(data) > 20:
            return " ".join(data[:20])

        return " ".join(data)

    def model_demo2(self, text):
        data = text.split()
        if len(data) > 10:
            return " ".join(data[:10])

        return " ".join(data)

    def model_bart(self, text):
        # loading
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-base", forced_bos_token_id=True)
        tok = BartTokenizer.from_pretrained("facebook/bart-base")
        device = torch.device('cpu')

        # remove newlines
        prepared_text = "summarize: " + text.strip().replace("\n", "")

        # tokenize
        tokenized_text = tok.encode(prepared_text, return_tensors="pt").to(device)

        # summmarize
        summary_ids = model.generate(tokenized_text,
                                     num_beams=self.beams,
                                     no_repeat_ngram_size=self.ngrams,
                                     min_length=self.min_length,
                                     max_length=self.max_length,
                                     early_stopping=True)

        # decode
        output = tok.decode(summary_ids[0], skip_special_tokens=True)

        return output

    def model_t5(self, text):
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        device = torch.device('cpu')

        preprocess_text = text.strip().replace("\n", "")
        t5_prepared_Text = "summarize: " + preprocess_text

        tokenized_text = tokenizer.encode(t5_prepared_Text,
                                          max_length=self.text_length,
                                          truncation=self.truncate,
                                          return_tensors="pt").to(device)

        # summmarize
        summary_ids = model.generate(tokenized_text,
                                     num_beams=self.beams,
                                     no_repeat_ngram_size=self.ngrams,
                                     min_length=self.min_length,
                                     max_length=self.max_length,
                                     early_stopping=True)

        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return output

"""
if __name__ == "__main__":
    tester = SummarizerModels('t5')
    tmp = 
        The US has "passed the peak" on new coronavirus cases, President Donald Trump said and predicted that some states would reopen this month.
        The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any country in the world.
        At the daily White House coronavirus briefing on Wednesday, Trump said new guidelines to reopen the country would be announced on Thursday after he speaks to governors.
        "We'll be the comeback kids, all of us," he said. "We want to get our country back."
        The Trump administration has previously fixed May 1 as a possible date to reopen the world's largest economy, but the president said some states may be able to return to normalcy earlier than that.
        
    out = tester.run(tmp)

    print(out)
"""

"""if __name__ == "__main__":
    tester = SummarizerModels('t5')
    out = tester.run("Lorem ipsum dolor immett ipsum dolor immett ipsum dolor immett ipsum dolor immett ipsum dolor immett"
                     "ipsum dolor immett ipsum dolor immett ipsum dolor immett ipsum dolor immett ipsum dolor immett")

    print(out)"""

"""
1. run model
2. summarize & save
3. concat
4. score
5. display
6. edit option
"""
