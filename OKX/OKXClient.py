### import DeribitClient ###
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from Logger.Logger import logger
### import Logger ###

from datetime import datetime, timezone
import hmac
import hashlib
import base64
import json
import requests
import urllib.parse
import websockets
import asyncio
import time

class OKXClient:
    def __init__(self):
        """
        初始化 OKXClient 实例
        :param api_key: 用户的 API Key
        :param secret_key: 用户的 API Secret
        :param passphrase: 用户的 API Passphrase
        :param BASE_REST_URL: 请求的基础 URL（默认为生产环境地址）
        :param BASE_WS_URL: 请求的基础 WebSocket URL（默认为生产环境地址）
        """
        with open("Account_Info/OKX/API_Key.txt", "r") as f:
            self.api_key = f.read()
        with open("Account_Info/OKX/Secret_Key.txt", "r") as f:
            self.secret_key = f.read()
        with open("Account_Info/OKX/Passphrase.txt", "r") as f:
            self.passphrase = f.read()
        
        # REST API
        self.BASE_REST_URL = "https://www.okx.com"
        # WebSocket
        self.BASE_WS_PUBLIC = "wss://ws.okx.com:8443/ws/v5/public"
        self.BASE_WS_PRIVATE = "wss://ws.okx.com:8443/ws/v5/private"
        self.websocket_public = None
        self.websocket_private = None

        # == Initialize Variables ==
        self.is_running = True

    # == Get Signature ==
    @staticmethod
    def _get_sign(message, secret):
        signature = hmac.new(bytes(secret, encoding='utf8'), bytes(message, encoding='utf-8'), digestmod='sha256')
        return base64.b64encode(signature.digest()).decode('utf-8')
    
    # == Pre-hash ==
    @staticmethod
    def _pre_hash(timestamp, method, request_path, body):
        return str(timestamp) + str.upper(method) + request_path + body

    # == Get Request Headers ==
    @staticmethod
    def _get_header(key, sign, passphrase, timestamp):
        header = {}
        header['Content-Type'] = 'application/json'
        header['OK-ACCESS-KEY'] = key
        header['OK-ACCESS-SIGN'] = sign
        header['OK-ACCESS-TIMESTAMP'] = str(timestamp)
        header['OK-ACCESS-PASSPHRASE'] = passphrase
        return header
    
    def send_request(self, method, path, params):
        clean = {k: v for k, v in params.items()
                 if v not in (None, "", [], {})}
        
        if method == "GET" and clean:
            qs = urllib.parse.urlencode(sorted(clean.items()))
            path = f"{path}?{qs}"

        body = ''
        if method.upper() == 'POST':
            body = json.dumps(clean)

        url = self.BASE_REST_URL + path

        # 5. 签名所需的 timestamp
        ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + "Z"
        sign = self._get_sign(
            self._pre_hash(ts, method, path, body),
            self.secret_key
        )

        header = self._get_header(
            self.api_key,
            sign,
            self.passphrase,
            ts
        )

        # 7. 发请求（自动用实例 proxies）
        response = requests.request(
            method=method,
            url=url,
            headers=header,
            data=body if method.upper() == 'POST' else None,
        )
        
        if not str(response.status_code).startswith('2'):
            logger.error(f"HTTP {response.status_code}: {response.text}")
            return None
        
        return response.json()

    # == Get Instruments ==
    def get_instruments(self, instType: str, uly: str = None, instFamily: str = None):
        """
        获取所有交易对
        :param instType: 交易对类型，可选值为 "SPOT" 或 "MARGIN"或"SWAP"或"FUTURES"或"OPTION"
        :param uly: 标的指数，当 instType 为 OPTION 时，uly或instFamily必填
        :param instFamily: 交易对所属的家族，当 instType 为 OPTION 时，uly或instFamily必填
        :return: 包含所有交易对的 JSON 数据，如果请求失败返回 None
        """

        params = {
            "instType": instType
        }
        if instType in ["OPTION", "FUTURES"]:
            if uly:
                params["uly"] = uly
            if instFamily:
                params["instFamily"] = instFamily
            if not uly and not instFamily:
                logger.error("uly or instFamily is required when instType is OPTION")
                return None

        request_path = "/api/v5/account/instruments"
        response = self.send_request("GET", request_path, params)
        return response
    
    # == Get Balance ==
    def get_balance(self, ccy: str = None):
        """
        获取交易账户中资金余额信息。

        Rate Limit: 20 requests per 2 seconds

        HTTP Request Header: GET /api/v5/account/balance

        请求参数：
        Parameter	    Type	    Required	    Description
        ccy	            str	        No	            币种，如 BTC 或 BTC,ETH
        """

        params = {}
        if ccy is not None:
            params["ccy"] = ccy

        request_path = "/api/v5/account/balance"

        response = self.send_request("GET", request_path, params)
        return response

    # == Tickers ==
    def tickers(self, instType: str):
        """
        获取所有交易对的最新价格信息。

        Rate Limit: 20 requests per 2 seconds

        HTTP Request Header: GET /api/v5/market/tickers

        请求参数：
        Parameter	    Type	    Required	    Description
        instType	    str	        Yes	            Instrument type, e.g. SPOT
        """

        params = {
            "instType": instType
        }

        request_path = "/api/v5/market/tickers"
        response = self.send_request("GET", request_path, params)
        return response
    
    # == Ticker ==
    def ticker(self, instId: str):
        """
        获取指定交易对的最新价格信息。

        Rate Limit: 20 requests per 2 seconds

        HTTP Request Header: GET /api/v5/market/ticker

        请求参数：
        Parameter	    Type	    Required	    Description
        instId	        str	        Yes	            Instrument ID, e.g. BTC-USDT
        """

        params = {
            "instId": instId
        }

        request_path = "/api/v5/market/ticker"
        response = self.send_request("GET", request_path, params)
        return response
    
    # == Orderbook ==
    def orderbook(self, instId: str, sz: str = 1):
        """
        获取指定交易对的最新价格信息。

        Rate Limit: 10 requests per 2 seconds

        HTTP Request Header: GET /api/v5/market/books-full

        请求参数：
        Parameter	    Type	    Required	    Description
        instId	        str	        Yes	            Instrument ID, e.g. BTC-USDT
        sz	            str	        No	            Order book depth per side. Default 1, max 5000
        """

        params = {
            "instId": instId,
            "sz": sz
        }

        request_path = "/api/v5/market/books-full"
        response = self.send_request("GET", request_path, params)
        return response
    
    # == Place Order ==
    def place_order(self, instId: str, tdMode: str, side: str, ordType: str, sz: str, px: str = None):
        """
        下单

        Rate Limit: 60 requests per 2 seconds

        HTTP Request Header: POST /api/v5/trade/order

        请求参数：
        Parameter	    Type	    Required	    Description
        instId	        str	        Yes	            Instrument ID, e.g. BTC-USDT
        tdMode	        str	        Yes	            Trading mode, e.g. cross, isolated
        side	        str	        Yes	            Side, e.g. buy, sell
        ordType	        str	        Yes	            Order type, e.g. limit, market
        sz	            str	        Yes	            Order size
        px	            str	        No	            Order price
        """
        params = {
            "instId": instId,
            "tdMode": tdMode,
            "side": side,
            "ordType": ordType,
            "sz": sz,
        }
        if ordType != "market":
            params["px"] = px

        request_path = "/api/v5/trade/order"
        response = self.send_request("POST", request_path, params)
        return response
    
    # == Place Multiple Orders ==
    def place_multiple_orders(self, batch_orders: list):
        """
        批量下单
        """
        pass

    # == Cancel Order ==
    def cancel_order(self, instId: str, ordId: str = None, clOrdId: str = None):
        """
        取消订单
        """
        if not ordId and not clOrdId:
            logger.error("ordId or clOrdId is required")
            return None
        
        params = {
            "instId": instId
        }
        if ordId:
            params["ordId"] = ordId
        if clOrdId:
            params["clOrdId"] = clOrdId

        request_path = "/api/v5/trade/cancel-order"
        response = self.send_request("POST", request_path, params)
        return response
    
    # == Cancel Multiple Orders ==
    def cancel_multiple_orders(self, batch_orders: list):
        """
        批量取消订单
        """
        pass

    # == Get Order Details ==
    def get_order_details(self, instId: str, ordId: str = None, clOrdId: str = None):
        """
        获取订单详情

        Rate Limit: 60 requests per 2 seconds

        HTTP Request Header: GET /api/v5/trade/orders-pending

        请求参数：
        Parameter	    Type	    Required	    Description
        instId	        str	        Yes	            Instrument ID, e.g. BTC-USDT
        ordId	        str	        No	            Order ID
        clOrdId	        str	        No	            Client Order ID
        """
        if not ordId and not clOrdId:
            logger.error("ordId or clOrdId is required")
            return None
        
        params = {
            "instId": instId
        }
        if ordId:
            params["ordId"] = ordId
        if clOrdId:
            params["clOrdId"] = clOrdId

        request_path = "/api/v5/trade/orders-pending"
        response = self.send_request("GET", request_path, params)
        return response
    
    # == Get Transaction Details (last 3 days) ==
    def get_transaction_details(self, instType: str = None, instId: str = None, ordId: str = None, after: str = None, before: str = None, begin: str = None, end: str = None, limit: str = None):
        """
        获取指定交易对的最新价格信息。
        """
        params = {
            "instType": instType,
            "instId": instId,
            "ordId": ordId,
            "after": after,
            "before": before,
            "begin": begin,
            "end": end,
            "limit": limit
        }

        request_path = "/api/v5/trade/fills"
        response = self.send_request("GET", request_path, params)
        return response
    
    # ==========================
    # == WebSocket Endpoints ===
    # ==========================

    # == WebSocket Connection ==
    async def connect(self, private: bool = False):
        """
        建立 WebSocket 连接，失败则重试
        private=False: 公共通道；True: 私有通道
        """
        try:
            if private:
                self.websocket_private = await websockets.connect(self.BASE_WS_PRIVATE)
                logger.info("Private WS connected.")
            else:
                self.websocket_public = await websockets.connect(self.BASE_WS_PUBLIC)
                logger.info("Public WS connected.")
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")

    # == WebSocket Login ==
    async def login(self):
        if not self.websocket_private:
            logger.error("Private WS not connected. Call connect(private=True) first.")
            return
        
        ts = str(int(datetime.now().timestamp()))
        sign = self._get_sign(ts + 'GET' + '/users/self/verify' + '', self.secret_key)
        msg = {
            'op': 'login',
            'args': [{
                'apiKey': self.api_key,
                'passphrase': self.passphrase,
                'timestamp': ts,
                'sign': sign
            }]
        }
        await self.websocket_private.send(json.dumps(msg))
        response = await self.websocket_private.recv()
        print(response)

    # == WebSocket Keep Alive ==
    async def heartbeat(self):
        """
        对已连接的 public/private WS 每 25s 发一次 'ping' 保持活跃。
        """
        try:
            while self.is_running:
                await asyncio.sleep(25)
                if self.websocket_public:
                    await self.websocket_public.send("ping")
                if self.websocket_private:
                    await self.websocket_private.send("ping")
        except Exception as e:
            logger.error(f"WebSocket heartbeat error: {e}")

    # == WebSocket Subscribe ==
    async def subscribe(self):
        pass

    # == WebSocket Unsubscribe ==
    async def unsubscribe(self):
        pass

    # == WebSocket Disconnect ==
    async def disconnect(self):
        if self.websocket_private:
            await self.websocket_private.close()
            self.websocket_private = None
            logger.info("Private WS disconnected.")
        if self.websocket_public:
            await self.websocket_public.close()
            self.websocket_public = None
            logger.info("Public WS disconnected.")

    # ==========================
    # == WebSocket Endpoints ===
    # ==========================

# if __name__ == "__main__":
#     client = OKXClient()
    
#     # 验证 API Key 是否有效
#     response = client.get_instruments(instType="OPTION", uly="ETH-USD")
#     if response:
#         print(json.dumps(response, indent=2))

#     # async def main():
#     #     await client.connect(private=True)
#     #     await client.login()
#     #     await client.disconnect()
#     # asyncio.run(main())