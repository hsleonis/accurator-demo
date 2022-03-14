import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import re
from rouge import Rouge
import json


def preprocess_helper(txt):
    """
    Preprocess html text
    :param txt: string
    :return: string
    """

    # Handle NaN
    if pd.isna(txt):
        return txt

    txt = re.sub(r"\[[\d,\s]*\]", '', str(txt))  # Remove citations
    txt = txt.encode('ascii', errors='ignore').strip().decode('ascii')  # decode ASCII to remove \u***** characters
    txt = re.sub(r'http.*?(?=\s)', "", txt)  # remove urls
    txt = re.sub(r'\s+[\.]', ' ', txt)  # remove space before full stop
    txt = re.sub(r'\s+', ' ', txt)  # remove extra whitespace

    return txt


def percentage(val, decimal='.2f'):
    """
    Convert number to percentage string.
    e.g. percentage(0.01111) => 1.11%
    :param val: float
    :param decimal: string
    :return: string
    """
    if val == '--':
        return val

    return str(format(val*100, decimal)) + '%'


def table_html_generator(data):
    """
    Generates HTML from list
    :param data: list of tuples in format (sumid,title,abstract,summary,model,url,entry_date)
    :return: html string
    """

    html = "<table id='table_summarization' class='table table-bordered table-responsive'>" \
           "<thead><tr><th>Info</th><th>Abstract</th><th>Summary</th><th>Metrics</th></tr></thead><tbody>"
    for row in data:
        rd = dict(zip(('uid', 'sumid', 'title', 'abstract', 'summary', 'model', 'url', 'bleu', 'f1',
                       'precision', 'recall', 'rating', 'feedback', 'entry_date'), row))

        html += f"<tr class='{rd['uid']}'><td>Title: <a target='_blank' href='{rd['url']}'>{rd['title']}</a><br/>" \
                f"<br/>Model: {rd['model']}</td>" + \
                f"<td><p>{rd['abstract']}</p></td>" \
                f"<td><p>{rd['summary']}</p></td>" \
                f"<td>BLEU Score: {percentage(rd['bleu'])}<br/>" \
                f" F1 Score: {percentage(rd['f1'])}<br/> Precision: {percentage(rd['precision'])}," \
                f" Recall: {percentage(rd['recall'])}<br />" \
                f"<br/>Rating: <p class='db-rating' data-uid='{rd['uid']}' data-rating='{rd['rating']}'></p>" \
                f"<br />Feedback: <div class='feedback-wrapper'><p class='feedback-text'>{rd['feedback']}</p>" \
                f"<button class='btn btn-primary edit_btn'>Edit</button></div>" \
                f"<form><textarea>{rd['feedback']}</textarea>" \
                f"<button data-uid='{rd['uid']}' class='btn btn-warning fd_btn'>Save Feedback</button></form></td>" \
                f"</tr>"
    html += "</tbody></table>"

    return html


def score_calculator(txt1, txt2):
    """
    Calculates Bleu score between two texts
    :param txt1: string - hypothesis
    :param txt2: string - reference
    :return: dict
    """
    decimal = '.2f'

    # bleu score
    try:
        chencherry = SmoothingFunction()
        bleu = sentence_bleu([str(txt1).split()], str(txt2).split(), smoothing_function=chencherry.method1)
    except Exception as error:
        raise 'Error in BLEU score calculation: {}'.format(error)

    # rouge score
    try:
        rouge = Rouge()
        rouge_score = rouge.get_scores(str(txt1), str(txt2))[0]['rouge-l']
    except Exception as error:
        raise 'Error in ROGUE score calculation: {}'.format(error)

    return {
        'bleu': format(bleu, decimal),
        'precision': format(rouge_score['p'], decimal),
        'recall': format(rouge_score['r'], decimal),
        'f1': format(rouge_score['f'], decimal)
    }
