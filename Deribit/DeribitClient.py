### import Logger ###
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
sys.path.append(BASE_DIR)

from Logger.Logger import logger
### import Logger ###

class DeribitClient:
    def __init__(self):
        pass