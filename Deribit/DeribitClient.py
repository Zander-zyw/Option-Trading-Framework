### import DeribitClient ###
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from Logger.Logger import logger
### import Logger ###

from cryptography.hazmat.primitives import serialization
from datetime import datetime
import base64
import websockets
import asyncio
import json

class DeribitClient:
    def __init__(self):
        ### API Info ###
        self.client_id = '7WPAssam'
        with open('Account_Info/Deribit/password.txt', 'r') as pswd:
            self.password = pswd.read()
        with open('Account_Info/Deribit/private.pem', 'rb') as private_pem:
            self.private_key = serialization.load_pem_private_key(
                private_pem.read(), password=f'{self.password}'.encode()
            )
        ### API Info ###
        
        self.websocket = None
        self.is_running = True
        self.pending_requests = {}  # 用于存储等待响应的 Future

    # Generate ID for Requests
    def _generate_request_id(self):
        req_id = int(datetime.now().timestamp() * 1000)
        return req_id

    # Generate Authentication Message
    def _generate_auth_msg(self):
        timestamp = round(datetime.now().timestamp() * 1000)
        nonce = "abcd"
        data = ""
        data_to_sign = bytes('{}\n{}\n{}'.format(timestamp, nonce, data), "latin-1")
        signature = base64.urlsafe_b64encode(self.private_key.sign(data_to_sign)).decode('utf-8').rstrip('=')
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

    # Check Websocket Connection
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

    async def connect(self):
        """
        1. Connect to WebSocket
        2. Authenticate
        3. Start read loop
        """
        try:
            # == WebSocket Connection ==
            self.websocket = await websockets.connect('wss://www.deribit.com/ws/api/v2')
            logger.info("WebSocket connection established.")

            # == Authentication ==
            auth_msg = self._generate_auth_msg()
            await self.websocket.send(json.dumps(auth_msg))
            
            response_str = await self.websocket.recv()
            response = json.loads(response_str)
            if response.get("result") and response["result"].get("access_token"):
                logger.info("Authentication successful.")
            else:
                logger.error(f"Authentication failed. Response: {response}")
                return

            # Start the read loop
            asyncio.create_task(self._read_loop())
            logger.info("Read loop started.")
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {e}")

    # == Send Request ==
    async def send_request(self, request_msg):
        req_id = request_msg["id"]
        future = asyncio.get_event_loop().create_future()
        self.pending_requests[req_id] = future

        await self.websocket.send(json.dumps(request_msg))
        response = await future
        return response

    # Get Ticker Information
    async def ticker(self, instrument_name: str):
        if not self.websocket or not self._is_ws_connected(self.websocket):
            logger.error("WebSocket is not connected. Cannot get ticker.")
            return

        request_id = self._generate_request_id()
        ticker_msg = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "public/ticker",
            "params": {
                "instrument_name": instrument_name
            }
        }
        try:
            response = await self.send_request(ticker_msg)
            return response
        except Exception as e:
            logger.error(f"Error getting ticker: {e}")
            return None

    # Get Instrument Information
    async def get_instruments(self, currency: str, kind: str):
        if not self.websocket or not self._is_ws_connected(self.websocket):
            logger.error("WebSocket is not connected. Cannot get instruments.")
            return
        
        request_id = self._generate_request_id()
        get_instruments_msg = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "public/get_instruments",
            "params": {
                "currency": currency,
                "kind": kind
            }
        }

        try:
            response = await self.send_request(get_instruments_msg)
            return response
        except Exception as e:
            logger.error(f"Error getting instruments: {e}")
            return None

    # Send Order
    async def send_order(self, side, order_type, instrument_name, price, amount):
        if side not in ["buy", "sell"]:
            logger.error("Invalid order side. Must be 'buy' or 'sell'.")
            return
        if order_type not in ["limit", "market"]:
            logger.error("Invalid order type. Must be 'limit' or 'market'.")
            return
        if order_type == 'limit' and not isinstance(price, (int, float)):
            logger.error("Price must be a number.")
            return
        if not isinstance(amount, (int, float)):
            logger.error("Amount must be a number.")
            return
        if not self.websocket or not self._is_ws_connected(self.websocket):
            logger.error("WebSocket is not connected. Cannot send order.")
            return
        
        # Order message
        request_id = self._generate_request_id()
        order_msg = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": f"private/{side}",
            "params": {
                "instrument_name": instrument_name,
                "amount": amount,
                "type": order_type,
                "label": f"{side}_{order_type}_{price}_{amount}_{request_id}",
            }
        }

        # Add price for limit orders
        if order_type == 'limit':
            order_msg["params"]["price"] = price

        # Send the order message
        logger.info(f"Sending order message: {order_msg}")
        try:
            await self.websocket.send(json.dumps(order_msg))
            response = await self.websocket.recv()
            logger.info(f"Order response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error sending order message: {e}")
            return None
    
    # Cancel Order
    async def cancel_order(self, order_id: str):
        if not self.websocket or not self._is_ws_connected(self.websocket):
            logger.error("WebSocket is not connected. Cannot cancel order.")
            return
        
        request_id = self._generate_request_id()
        cancel_msg = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "private/cancel",
            "params": {
                "order_id": order_id
            }
        }

        logger.info(f"Cancel order request sent: {cancel_msg}")
        try:
            response = await self.send_request(cancel_msg)
            logger.info(f"Cancel order response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error sending cancel order request: {e}")
            return None

    # Subscribe
    async def subscribe(self, channels: list):
        if not self.websocket or not self._is_ws_connected(self.websocket):
            logger.error("WebSocket is not connected. Cannot subscribe.")
            return
        
        subscribe_msg = {
            "jsonrpc": "2.0",
            "id": self._generate_request_id(),
            "method": "public/subscribe",
            "params": {
                "channels": channels
            }
        }
        try:
            await self.websocket.send(json.dumps(subscribe_msg))
            logger.info(f"Sent subscribe message: {subscribe_msg}")
        except Exception as e:
            logger.error(f"Error sending subscribe message: {e}")

    # Unsubscribe
    async def unsubscribe(self, channels):
        if not self.websocket or not self._is_ws_connected(self.websocket):
            logger.error("WebSocket is not connected. Cannot unsubscribe.")
            return
        
        if not isinstance(channels, list):
            logger.error("Channels must be a list.")
            return
        
        unsubscribe_msg = {
            "jsonrpc": "2.0",
            "id": self._generate_request_id(),
            "method": "public/unsubscribe",
            "params": {
                "channels": channels
            }
        }
        try:
            await self.websocket.send(json.dumps(unsubscribe_msg))
            logger.info(f"Sent unsubscribe message: {unsubscribe_msg}")
        except Exception as e:
            logger.error(f"Error sending unsubscribe message: {e}")

    # Get Positions
    async def get_positions(self, currency, kind):
        if not self.websocket or not self._is_ws_connected(self.websocket):
            logger.error("WebSocket is not connected. Cannot get positions.")
            return
        
        request_id = self._generate_request_id()
        get_positions_msg = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "private/get_positions",
            "params": {
                "currency": currency,
                "kind": kind
            }
        }
        try:
            response = await self.send_request(get_positions_msg)
            logger.info(f"Get positions response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return None
        
    # Get Order Book Information
    async def get_order_book(self, instrument_name: str, depth: int):
        if not self.websocket or not self._is_ws_connected(self.websocket):
            logger.error("WebSocket is not connected. Cannot get order book.")
            return
        if depth <= 0:
            logger.error("Depth must be greater than 0.")
            return
        
        request_id = self._generate_request_id()
        get_order_book_msg = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "public/get_order_book",
            "params": {
                "instrument_name": instrument_name,
                "depth": depth
            }
        }
        try:
            response = await self.send_request(get_order_book_msg)
            logger.info(f"Get order book response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error getting order book: {e}")
            return None

    # Get Order State
    async def get_order_state(self, order_id: str):
        if not self.websocket or not self._is_ws_connected(self.websocket):
            logger.error("WebSocket is not connected. Cannot get order book.")
            return

        request_id = self._generate_request_id()
        state_request = {
            "jsonrpc" : "2.0",
            "id" : request_id,
            "method" : "private/get_order_state",
            "params" : {
                "order_id" : order_id
            }
        }

        try:
            response = await self.send_request(state_request)
            logger.info(f"Get order state response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error getting order state: {e}")
            return None

    async def disconnect(self):
        self.is_running = False
        if self.websocket and self._is_ws_connected(self.websocket):
            await self.websocket.close()
            logger.info("WebSocket connection closed manually.")
        else: 
            logger.error("No WebSocket connection to close.")


# async def main():
#     client = DeribitClient()
#     await client.connect()
#     await asyncio.sleep(3)

#     await client.subscribe(["deribit_price_index.btc_usd"])
#     await asyncio.sleep(3)
#     await client.unsubscribe(["deribit_price_index.btc_usd"])

#     await asyncio.sleep(3)
#     await client.disconnect()

# if __name__ == "__main__":
#     asyncio.run(main())
