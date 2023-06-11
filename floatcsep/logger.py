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
        "logfile": {
            "formatter": "default",
            "level": "INFO",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_NAME,
            "backupCount": 2,
            "delay": True
        },
        "console": {
            "formatter": "default",
            "level": "INFO",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        }
    },
    'loggers': {
        'fileLogger':
            {'level': 'INFO',
             'handlers': ['logfile'],
             'propagate': False}},
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'logfile']}
}

logging.config.dictConfig(LOGGING_CONFIG)
logging.getLogger('numexpr').setLevel(logging.WARNING)
logger = logging.getLogger()
# if not os.path.exists(LOG_NAME):
#     for i in logger.handlers:
#         i.doRollover()
