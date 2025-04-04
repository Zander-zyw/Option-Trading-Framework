### import DeribitClient ###
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from Logger.Logger import logger
### import Logger ###

import websockets
import time
import hashlib
import hmac
import json
import asyncio
from urllib.parse import urlencode

class BinanceClient:
    def __init__(self):
        ### API Info ###
        self.api_key = open('Account_Info/Binance/api_key.txt', 'r').read()
        self.api_secret = open('Account_Info/Binance/api_secret.txt', 'r').read()
        ### API Info ###
        
        self.websocket = None
        self.is_running = True
        self.pending_requests = {}
        
    # == Generate Request ID ==
    def _generate_request_id(self):
        return str(int(time.time() * 1000))
    
    # == Generate Signature ==
    def _generate_signature(self, method, endpoint, params):
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    # == Check Websocket Connection ==
    def _is_ws_connected(self, websocket):
        if websocket.state == websockets.protocol.State.OPEN:
            return True
        elif websocket.state == websockets.protocol.State.CLOSED:
            logger.error("WebSocket connection is closed.")
            return False
        elif websocket.state == websockets.protocol.State.CLOSING:
            logger.error("WebSocket connection is closing.")
            return False
        else:
            logger.error("WebSocket connection is not established.")
            return False
    
    # == Read Response Loop ==
    async def _read_loop(self):
        """
        统一读取循环：只在这里调用 websocket.recv()，并将响应分发到对应的 Future 中。
        """
        while self.is_running and self._is_ws_connected(self.websocket):
            try:
                message_str = await self.websocket.recv()
                message = json.loads(message_str)
                
                if "id" in message:
                    req_id = message.get("id")
                    future = self.pending_requests.pop(req_id, None)
                    if future and not future.done():
                        future.set_result(message)
                else:
                    # 如果是订阅推送或其他消息，则直接记录日志
                    logger.info(f"Received message: {message}")

            except websockets.exceptions.ConnectionClosed:
                logger.info("WebSocket connection closed. Exiting read loop.")
                break
            except Exception as e:
                logger.error(f"Error in read loop: {e}")
                break

    # == Connect to Binance WebSocket ==
    async def connect(self):
        try:
            self.websocket = await websockets.connect(
                "wss://fstream.binance.com/ws"
            )
            logger.info("WebSocket connection established.")
            
            # Start the read loop
            asyncio.create_task(self._read_loop())
            logger.info("Read loop started.")
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
        
    # == Send Request ==
    async def send_request(self, request_msg):
        req_id = request_msg["id"]
        future = asyncio.get_event_loop().create_future()
        self.pending_requests[req_id] = future

        await self.websocket.send(json.dumps(request_msg))
        response = await future
        return response
    
    # == Close WebSocket Connection ==
    async def disconnect(self):
        self.is_running = False
        if self.websocket and self._is_ws_connected(self.websocket):
            await self.websocket.close()
            logger.info("WebSocket connection closed manually.")
        else: 
            logger.error("No WebSocket connection to close.")

    # == Get Ticker Information ==
    async def ticker(self, symbol: str):
        if not self.websocket or not self._is_ws_connected(self.websocket):
            logger.error("WebSocket is not connected. Cannot get ticker.")
            return

        request_id = self._generate_request_id()
        ticker_msg = {
            "method": "SUBSCRIBE",
            "params": [f"{symbol.lower()}@ticker"],
            "id": request_id
        }
        try:
            response = await self.send_request(ticker_msg)
            logger.info(f"Ticker response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error getting ticker: {e}")
            return None

# async def main():
#     binance_client = BinanceClient()
#     await binance_client.connect()
#     await asyncio.sleep(3)

#     # Get Ticker Information
#     ticker_response = await binance_client.ticker("BTCUSDT")
#     await asyncio.sleep(10)

#     await binance_client.disconnect()

# if __name__ == "__main__":
#     asyncio.run(main())