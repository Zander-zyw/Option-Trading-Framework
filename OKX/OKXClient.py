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
        # === API Keys ===
        with open("Account_Info/OKX/API_Key.txt", "r") as f:
            self.api_key = f.read().strip()
        with open("Account_Info/OKX/Secret_Key.txt", "r") as f:
            self.secret_key = f.read().strip()
        with open("Account_Info/OKX/Passphrase.txt", "r") as f:
            self.passphrase = f.read().strip()

        # === REST endpoints ===
        self.BASE_REST_URL = "https://www.okx.com"

        # === WS endpoints ===
        self.BASE_WS_PUBLIC  = "wss://ws.okx.com:8443/ws/v5/public"
        self.BASE_WS_PRIVATE = "wss://ws.okx.com:8443/ws/v5/private"
        self.ws_public  = None
        self.ws_private = None

        # === Shared state for WS ===
        self.is_running       = True
        self.pending_requests = {}   # Map key→Future
        self._req_counter     = 0
        # Track active subscriptions
        self.subscribed_public = set()
        self.subscribed_private = set()

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
    def get_instruments(self, instType: str, uly: str = None, instFamily: str = None, instId: str = None):
        """
        获取所有交易对
        
        Rate Limit: 20 requests per 2 seconds

        HTTP Request Header: GET /api/v5/account/instruments

        请求参数：
        Parameter	    Type	    Required	    Description
        instType	    str	        Yes	            Instrument type: SPOT, MARGIN, SWAP, FUTURES, OPTION
        uly	            str	        No	            Underlying (Only for FUTURES/SWAP/OPTION). If instType is OPTION, either uly or instFamily is required.
        instFamily	    str	        No	            Instrument family (Only for FUTURES/SWAP/OPTION). If instType is OPTION, either uly or instFamily is required.
        instId	        str	        No	            Instrument ID
        """

        if instType == "OPTION":
            if not uly and not instFamily:
                logger.error("uly or instFamily is required when instType is OPTION")
                return None
            
        params = {
            "instType": instType,
            "uly": uly,
            "instFamily": instFamily,
            "instId": instId
        }
            
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

        params = {
            "ccy": ccy
        }

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
            "instId": instId,
            "ordId": ordId,
            "clOrdId": clOrdId
        }

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

    # == WebSocket Read Loop ==
    async def _read_loop(self, ws):
        """
        Automatically read responses from OKX WS and handle them.
        """
        try:
            while self.is_running:
                raw = await ws.recv()
                msg = json.loads(raw)

                event = msg.get('event')
                arg   = msg.get('arg', {}) or {}
                channel = arg.get('channel')
                instId  = arg.get('instId')

                if event == "error":
                    code    = msg.get("code")
                    err_msg = msg.get("msg")

                    # **取消pending request的第一个请求，跳出**
                    for key in list(self.pending_requests):
                        if key.startswith("subscribe:") or key.startswith("unsubscribe:"):
                            fut = self.pending_requests.pop(key)
                            if not fut.done():
                                fut.set_result(msg)
                            break

                    logger.error(
                        f"WS subscription error [{code}]: {err_msg}"
                    )
                    continue

                # ——————————————————————————————
                # Existing ACK logic
                # ——————————————————————————————
                ack_key = f"{event}:{channel}:{instId}"
                if ack_key in self.pending_requests:
                    fut = self.pending_requests.pop(ack_key)
                    if not fut.done():
                        fut.set_result(msg)
                    continue

                logger.info(f"WS message: {msg}")

        except websockets.ConnectionClosed:
            logger.info("WebSocket closed.")
        except Exception as e:
            logger.error(f"Read loop error: {e}")


    # == WebSocket Send Request ==
    async def _send_ws(self, websocket, payload, ack_event):
        """
        Send a WS JSON payload and wait until a message with event==ack_event arrives.
        """
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self.pending_requests[ack_event] = fut

        await websocket.send(json.dumps(payload))
        return await fut


    # == WebSocket Connection ==
    async def connect(self, private: bool = False):
        """
        Connects to OKX WS (public or private) and starts the read loop.
        """
        url = self.BASE_WS_PRIVATE if private else self.BASE_WS_PUBLIC
        try:
            ws = await websockets.connect(url)
            if private:
                self.ws_private = ws
                logger.info("OKX private WS connected.")
            else:
                self.ws_public = ws
                logger.info("OKX public WS connected.")

            # start read loop
            asyncio.create_task(self._read_loop(ws))
            asyncio.create_task(self.heartbeat())
        except Exception as e:
            logger.error(f"Failed to connect WS ({'private' if private else 'public'}): {e}")


    # == WebSocket Login ==
    async def login(self):
        if not self.ws_private:
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
        response = await self._send_ws(self.ws_private, msg, "login")
        logger.info(f"Login response: {response}")

    # == WebSocket Keep Alive ==
    async def heartbeat(self):
        """
        对已连接的 public/private WS 每 25s 发一次 'ping' 保持活跃。
        """
        try:
            while self.is_running:
                await asyncio.sleep(20)

                if self.ws_public:
                    await self.ws_public.ping()
                    logger.info(f"WS public ping!")
                if self.ws_private:
                    await self.ws_private.ping()
                    logger.info(f"WS private ping!")
        except Exception as e:
            logger.error(f"WebSocket heartbeat error: {e}")

    # == WebSocket Subscribe ==
    async def subscribe(self, args: list, private: bool=False):
        ws = self.ws_private if private else self.ws_public
        if not ws:
            raise RuntimeError("Connect WS before subscribing")

        loop = asyncio.get_running_loop()
        ack_futures = {}
        for arg in args:
            channel = arg.get('channel')
            instId  = arg.get('instId')
            ccy     = arg.get('ccy')
            key = f"subscribe:{channel}:{instId or ccy}"
            fut = loop.create_future()
            self.pending_requests[key] = fut
            ack_futures[key] = fut

        await ws.send(json.dumps({'op': 'subscribe', 'args': args}))

        results = []
        for key, fut in ack_futures.items():
            res = await fut
            if res.get('event') == 'error':
                code = res.get('code')
                msg  = res.get('msg')
                logger.error(f"[subscribe skipped] {key} → [{code}] {msg}")
                # simply skip adding to subscribed_public/subscribed_private
                continue

            # normal ACK
            results.append(res)
            _, channel, instId = key.split(':')
            (self.subscribed_private if private else self.subscribed_public).add((channel, instId))
            logger.info(f"Subscribed to {channel} {instId}")

        return results


    # == WebSocket Unsubscribe ==
    async def unsubscribe(self, args: list, private: bool=False):
        ws = self.ws_private if private else self.ws_public
        if not ws:
            raise RuntimeError("Connect WS before unsubscribing")

        loop = asyncio.get_running_loop()
        ack_futures = {}
        for arg in args:
            channel = arg.get('channel')
            instId  = arg.get('instId')
            ccy     = arg.get('ccy')
            key = f"unsubscribe:{channel}:{instId or ccy}"
            fut = loop.create_future()
            self.pending_requests[key] = fut
            ack_futures[key] = fut

        await ws.send(json.dumps({'op': 'unsubscribe', 'args': args}))

        results = []
        for key, fut in ack_futures.items():
            res = await fut
            if res.get('event') == 'error':
                code = res.get('code')
                msg  = res.get('msg')
                logger.error(f"[unsubscribe skipped] {key} → [{code}] {msg}")
                continue

            # normal ACK
            results.append(res)
            _, channel, instId = key.split(':')
            (self.subscribed_private if private else self.subscribed_public).discard((channel, instId))
            logger.info(f"Unsubscribed from {channel} {instId}")

        return results


    # == WebSocket Disconnect ==
    async def disconnect(self):
        if self.ws_private:
            await self.ws_private.close()
            self.ws_private = None
            logger.info("Private WS disconnected.")
        if self.ws_public:
            await self.ws_public.close()
            self.ws_public = None
            logger.info("Public WS disconnected.")

    # ==========================
    # == WebSocket Endpoints ===
    # ==========================


if __name__ == "__main__":
    async def main():
        client = OKXClient()

        # 1. 连接到公共 WebSocket 并启动心跳与读循环
        await client.connect(private=False)

        # 2. 测试订阅——这里以 BTC-USDT 的 ticker channel 为例
        subscribe_args = [
            {"channel": "tickers", "instId": "BTC-USDT"},
            {"channel": "account", "ccy": "BTC"},
            {"channel": "account", "ccy": "ETH"},
            {"channel": "trades", "instId": "BTC-USDT"},
            {"channel": "trades", "instId": "ETH-USDT"},
        ]
        resp = await client.subscribe(subscribe_args, private=False)
        print("Subscribe response:", resp)

        # 3. 等待一段时间接收推送（读循环会打印推送消息）
        await asyncio.sleep(30)

        # 4. 测试取消订阅
        unsubscribe_args = [
            {"channel": "tickers", "instId": "BTC-USDT"},
            {"channel": "account", "ccy": "BTC"},
            {"channel": "account", "ccy": "ETH"},
            {"channel": "trades", "instId": "BTC-USDT"},
            {"channel": "trades", "instId": "ETH-USDT"},
        ]
        resp = await client.unsubscribe(unsubscribe_args, private=False)
        print("Unsubscribe response:", resp)

        # 5. 等待一段时间接收推送（读循环会打印推送消息）
        await asyncio.sleep(30)

        # 4. 断开连接
        await client.disconnect()
        print("Disconnected.")

    asyncio.run(main())
