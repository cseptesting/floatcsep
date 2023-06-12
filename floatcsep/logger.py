import logging.config

LOG_NAME = 'experiment.log'
LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "default": {
            "format": '%(asctime)s %(levelname)s - %(message)s',
            "datefmt": '%Y-%m-%d %H:%M:%S'},
    },
    "handlers": {
        "console": {
            "formatter": "default",
            "level": "INFO",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        }
    },
    'loggers': {
        'floatLogger': {
            'level': 'DEBUG',
            'handlers': ['console'],
            'propagate': False
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console']}
}


def add_fhandler(filename):
    formatter = logging.Formatter(
        fmt=LOGGING_CONFIG['formatters']['default']['format'],
        datefmt=LOGGING_CONFIG['formatters']['default']['datefmt']
    )
    fhandler = logging.FileHandler(filename)
    fhandler.setFormatter(formatter)
    fhandler.setLevel(logging.DEBUG)

    logging.getLogger('floatLogger').addHandler(fhandler)


logging.config.dictConfig(LOGGING_CONFIG)
logging.getLogger('numexpr').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
