### import Logger ###
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
sys.path.append(BASE_DIR)

from Logger.Logger import logger
### import Logger ###

from cryptography.hazmat.primitives import serialization
from datetime import datetime
import base64


class DeribitClient:
    def __init__(self):
        self.client_id = '7WPAssam'
        with open('../password.txt', 'r') as pswd:
            self.password = pswd.read()
        with open('../private.pem', 'rb') as private_pem:
            self.private_key = serialization.load_pem_private_key(private_pem.read(), password=f'{self.password}'.encode())
    
    def _generate_request_id(self):
        # Generate a unique request ID
        req_id = int(datetime.now().timestamp() * 1000)
        return req_id

    #Generate Authentication Message
    def _generate_auth_msg(self):
        timestamp = round(datetime.now().timestamp() * 1000)
        nonce = "abcd"
        data = ""

        data_to_sign = bytes('{}\n{}\n{}'.format(timestamp, nonce, data), "latin-1")
        signature = base64.urlsafe_b64encode(self.private_key.sign(data_to_sign)).decode('utf-8').rstrip('=')

        #Authentication Request
        auth_msg = {
            "jsonrpc": "2.0",
            "id": self._generate_request_id(),
            "method": "public/auth",
            "params": {
                "grant_type": "client_signature",
                "client_id": self.client_id,
                "timestamp": timestamp,
                "signature": signature,
                "nonce": nonce,
                "data": data
            }
        }
        
        return auth_msg