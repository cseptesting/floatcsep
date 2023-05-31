import logging
import logging.config
from floatcsep.registry import LOGGING_CONFIG

__version__ = '0.1.2'
logging.config.dictConfig(LOGGING_CONFIG)
a = logging.getLogger(__name__)
a.info(f'Running floatCSEP v{__version__}')
