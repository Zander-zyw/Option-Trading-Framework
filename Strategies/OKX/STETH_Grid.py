### import OKXClient ###
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from OKX.OKXClient import OKXClient
from Logger.Logger import logger
### import OKXClient ###

class STETHGridClient(OKXClient):
    def __init__(self, symbol: str, side: str, leverage: float):
        super().__init__()
        self.symbol = symbol.upper()
        self.side = side.lower()
        self.leverage = leverage
