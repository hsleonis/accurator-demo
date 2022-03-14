class Config(object):
    TESTING = False


class ProductionConfig(Config):
    DATABASE = 'sqlite.db'
    CSV_FILES = []
    MODELS = ['Demo1', 'Demo2', 'Bart', 't5']
