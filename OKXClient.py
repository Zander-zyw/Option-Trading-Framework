import okx.Trade as Trade
import hmac
import hashlib
import requests
import base64
from datetime import datetime, timezone

class OKXClient:
    def __init__(self, api_key, api_secret, passphrase):
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self.passphrase = passphrase.strip()

    def _generate_auth_headers(self, method, request_path, body=""):
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        message = f"{timestamp}{method.upper()}{request_path}{body}"
        signature = base64.b64encode(hmac.new(self.api_secret.encode(), message.encode(), hashlib.sha256).digest()).decode()
        headers = {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json"
        }
        return headers

    def authenticate(self, request_path="/api/v5/account/balance", method="GET"):
        body = ""
        headers = self._generate_auth_headers(method, request_path, body)
        url = "https://www.okx.com" + request_path
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": f"请求异常: {e}"}
        
    def subscribe(self, channel):
        pass

if __name__ == '__main__':
    api_key = open('Account_Info/OKX/API_Key.txt').read()
    api_secret = open('Account_Info/OKX/Secret_Key.txt').read()
    passphrase = open('Account_Info/OKX/Passphrase.txt').read()
    
    client = OKXClient(api_key, api_secret, passphrase)
    auth_response = client.authenticate()
    print("认证接口响应:")
    print(auth_response)
