from pathlib import Path
from flask import current_app
import sqlite3
import subprocess
from app import app


class Dataset(object):
    """A simple class to perform operations on SQLite database.
    """

    def __init__(self):
        self.db = None
        self.cur = self.get_db().cursor()

    def get_db(self):
        """
        Returns a connection to the SQLite database file defined in config.py
        :return: database object
        """
        with app.app_context():
            if self.db is None:
                self.db = sqlite3.connect(current_app.config['DATABASE'])
            return self.db

    def init_db(self):
        with app.app_context():
            conn = self.get_db()
            sql = "CREATE TABLE IF NOT EXISTS summary (" \
                  "uid TEXT NOT NULL, " \
                  "sumid integer NOT NULL, " \
                  "title TEXT, " \
                  "abstract TEXT, " \
                  "summary TEXT, " \
                  "model TEXT, " \
                  "url TEXT, " \
                  "bleu REAL, " \
                  "precision REAL, "\
                  "recall REAL, " \
                  "fscore REAL, " \
                  "rating REAL, " \
                  "feedback TEXT, " \
                  "entry_date text NOT NULL)"
            rv = conn.execute(sql)
            conn.commit()

            return rv

    def insert_db(self, query, args=()):
        """
        Query the SQLite database
        :param query: string
        :param args: parameters when query string has placeholders (?)
        :return: one / all results from query
        """
        with app.app_context():
            conn = self.get_db()
            rv = conn.execute(query, args)
            conn.commit()

            return rv

    def query_db(self, query, args=(), one=False):
        """
        Query the SQLite database
        :param query: string
        :param args: parameters when query string has placeholders (?)
        :param one: boolean (returns one result)
        :return: one / all results from query
        """
        with app.app_context():
            conn = self.get_db()
            rv = conn.execute(query, args).fetchall()
            return (rv[0] if rv else None) if one else rv

    def update_db(self, query):
        """
        Update the SQLite database
        :param args: tuple
        :param query: string
        :return: cursor object
        """
        with app.app_context():
            conn = self.get_db()
            conn.execute(query)
            conn.commit()

            return True

    @staticmethod
    def import_db():
        """
        Import CSV files defined in config.py to SQLite database
        """
        with app.app_context():
            db_name = Path(current_app.config['DATABASE']).resolve()

            for file_name in current_app.config['CSV_FILES']:
                csv_file = Path(file_name).resolve()
                table_name = file_name.split('.')[0]

                result = subprocess.run(['sqlite3',
                                         str(db_name),
                                         '-cmd',
                                         '.mode csv',
                                         '.import ' + str(csv_file).replace('\\', '\\\\')
                                         + ' ' + table_name],
                                        capture_output=True)

                print(result)

    def close_db(self):
        """
        Close the SQLite database connection
        :return:
        """
        with app.app_context():
            self.get_db().close()
