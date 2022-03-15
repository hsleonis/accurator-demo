from app import app
from app import db
from flask import request, render_template
from flask import current_app
from markupsafe import escape
import requests, os, json, re
from bs4 import BeautifulSoup
from app.helpers import *
from app.models import SummarizerModels
import uuid
from datetime import datetime


@app.route("/")
def home():
    # create table if not exists
    database = db.Dataset()
    database.init_db()

    # model names to display
    data = {
        'models': current_app.config['MODELS']
    }
    return render_template("public/index.html", value=data)


@app.route('/parse', methods=['POST'])
def parse():
    url = request.values.get("url")
    uid = request.values.get("uid")

    # scrap articles with http-parser
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600',
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
    }
    req = requests.get(url, headers)

    # write html (in txt) file
    with open(f"data/{uid}.txt", "w") as outfile:
        outfile.write(req.text)

    return json.dumps({'success': True, 'url': url}), 200, {'ContentType': 'application/json'}


@app.route('/save_text', methods=['POST'])
def save_text():
    txt = request.values.get("txt")
    uid = 6

    # write html (in txt) file
    with open(f"data/{uid}.txt", "w") as outfile:
        txt = "<html><title>Custom Text</title><body>" + txt + "</body></html>"
        outfile.write(txt)

    return json.dumps({'success': True, 'url': '#'}), 200, {'ContentType': 'application/json'}


@app.route('/preprocess', methods=['POST'])
def preprocess():
    uid = request.values.get("uid")
    url = request.values.get("url")

    with open(f"data/{uid}.txt", "r") as html_file:

        # parse html
        soup = BeautifulSoup(html_file, 'html.parser')

        # title of the article
        title = soup.title.get_text().split('|')[0]

        # content
        body = soup.body.get_text()
        body = re.sub(r'[\n]+', "\n", body)

        # abstract
        match = re.search(r"(?<=Abstract)[\n\w\r]*.*", body)        # search for abstract
        abt = preprocess_helper(match.group()) if match is not None else "Not found."   # see if abstract found
        end = match.end() if match is not None else 0               # remove abstract from body text if found

        # split body content
        body = preprocess_helper(body[end:len(body)-1])
        tmp = body.split()          # split words with whitespaces
        split_size = 250            # size of each text chunk (later used in models separately)
        n = len(tmp) // split_size  # number of chunks
        body = [" ".join(tmp[i*split_size:i*split_size + split_size]) for i in range(n)]    # create chunks

        # build response
        sec_dict = {
            'success': True,
            'title': title,
            'body': body,
            'abstract': abt,
            'splits': n,
            'url': url
        }

        # write on json file
        with open(f"data/{uid}.json", "w") as outfile:
            json.dump(sec_dict, outfile)

    return json.dumps(sec_dict), 200, {'ContentType': 'application/json'}


@app.route("/model", methods=['POST'])
def model():
    model_name = str(escape(request.values.get("name")))
    sumid = request.values.get("sumid")

    success = f"Model: {model_name} passed."
    try_again = f"Model {model_name} not passed through."
    not_found = f"{model_name} not found, please contact admin."
    summary_length = 250

    if model_name not in current_app.config['MODELS']:
        return json.dumps({'success': False, 'msg': not_found}), 200, {'ContentType': 'application/json'}

    try:
        # load model
        model_to_use = SummarizerModels(model_name, summary_length, max_length=30)

        gen_summary = ""

        # open txt file
        with open(f"data/{sumid}.json", "r") as infile:
            # load data
            data = json.loads(infile.read())

            # generate summaries
            for idx, part in enumerate(data['body']):
                gen_summary += model_to_use.run(part)

            # insert into database
            database = db.Dataset()
            sql = "INSERT INTO summary (uid, sumid, title, abstract, summary, model, " \
                  "url, bleu, precision, recall, fscore, rating, feedback, entry_date) " \
                  "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            uid = str(uuid.uuid4())
            now = datetime.now()
            dt = now.strftime("%d-%m-%Y %H:%M:%S")
            rv = database.insert_db(sql,
                                    (uid, sumid, data['title'],
                                     data['abstract'], gen_summary,
                                     model_name, data['url'], "--", "--", "--", "--", 0.0, "-", dt))

            return json.dumps({'success': True, 'msg': success}), 200, {'ContentType': 'application/json'}
    except:
        return json.dumps({'success': False, 'msg': try_again}), 200, {'ContentType': 'application/json'}


@app.route("/table_viewer", methods=['POST'])
def table_viewer():
    success = f"Table generated."
    not_found = f"404 not found, please contact admin."

    try:
        # connect db
        database = db.Dataset()

        # get data
        data = database.query_db("SELECT * FROM summary")

        # process data
        html = table_html_generator(data)

        # close connection
        database.close_db()

        return json.dumps({'success': True, 'msg': success, 'html': html}), 200, {'ContentType': 'application/json'}
    except:
        return json.dumps({'success': False, 'msg': not_found, 'html': ''}), 200, {'ContentType': 'application/json'}


@app.route("/calc_metrics", methods=['POST'])
def calc_metrics():
    success = f"Table generated."
    not_found = f"404 not found, please contact admin."

    # connect db
    database = db.Dataset()

    # get data
    update_data = database.query_db("SELECT * FROM summary")

    for row in update_data:
        # calculate
        met = score_calculator(row[3], row[4])

        # update database
        sql = f"UPDATE summary SET bleu = {met['bleu']}, precision = {met['precision']}, "\
              f"recall = {met['recall']}, fscore = {met['f1']} WHERE uid = '{row[0]}'"

        database.update_db(sql)

    # get summaries
    data = database.query_db("SELECT * FROM summary")
    html = table_html_generator(data)

    # close connection
    database.close_db()

    return json.dumps({'success': True, 'msg': success, 'html': html}), 200, {'ContentType': 'application/json'}
    # return json.dumps({'success': False, 'msg': not_found, 'html': ''}), 200, {'ContentType': 'application/json'}


@app.route("/star_rating", methods=['POST'])
def star_rating():
    success = f"Rating saved."
    not_found = f"404 not found, please contact admin."
    val = float(request.values.get("val"))
    uid = request.values.get("uid")

    # connect db
    database = db.Dataset()

    # update database
    sql = f"UPDATE summary SET rating={val} WHERE uid = '{uid}'"
    database.update_db(sql)

    # close connection
    database.close_db()

    return json.dumps({'success': True, 'msg': success}), 200, {'ContentType': 'application/json'}


@app.route("/save_feedback", methods=['POST'])
def save_feedback():
    success = f"Feedback saved."
    not_found = f"404 not found, please contact admin."
    val = request.values.get("val")
    uid = request.values.get("uid")

    # connect db
    database = db.Dataset()

    # update database
    sql = f"UPDATE summary SET feedback='{val}' WHERE uid = '{uid}'"
    database.update_db(sql)

    # close connection
    database.close_db()

    return json.dumps({'success': True, 'msg': success}), 200, {'ContentType': 'application/json'}


@app.route("/clear_table", methods=['POST'])
def clear_table():
    success = f"All data cleared."
    not_found = f"Error occurred, please contact admin."

    try:
        # connect db
        database = db.Dataset()

        # update database
        sql = f"DELETE FROM summary"
        database.update_db(sql)

        # close connection
        database.close_db()
    except:
        return json.dumps({'success': False, 'msg': not_found}), 200, {'ContentType': 'application/json'}

    return json.dumps({'success': True, 'msg': success}), 200, {'ContentType': 'application/json'}


@app.errorhandler(404)
def page_not_found(error):
    return "404 not found."
