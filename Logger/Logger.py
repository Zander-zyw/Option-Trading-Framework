import logging
import os

class Logger:
    def __init__(self, filename="trading.log", folder="Logger"):
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, filename)

        self.logger = logging.getLogger("DeribitTradingLogger")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            file_handler = logging.FileHandler(filename)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

logger = Logger()