### import Logger ###
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
            self.private_key = serialization.load_pem_private_key(private_pem.read(), password=f'{self.password}'.encode())
        ### API Info ###

        self.websocket = None
        self.is_running = True
    
    # Generate ID for Requests
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

    # Websocket Connection
    async def connect(self):
        try:        
            async with websockets.connect('wss://www.deribit.com/ws/api/v2') as websocket:
                logger.info("WebSocket connection established. Sending authentication message...")
                self.websocket = websocket

                await websocket.send(json.dumps(self._generate_auth_msg()))
                response = await websocket.recv()
                response = json.loads(response)
                
                if response.get("result") and response["result"].get("access_token"):
                    logger.info("Authentication successful.")
                else:
                    logger.error(f"Authentication failed. Response: {response}")
                    return
                
                while self.is_running and self._is_ws_connected(websocket):
                    try:
                        message = await websocket.recv()
                        message = json.loads(message)
                        
                        logger.info(f"Received message: {message}")
                    except Exception as e:
                        logger.error(f"Error receiving message: {e}")
                        break

        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {e}")
        finally:
            logger.info("WebSocket connection closed.")

    # Disconnect Websocket
    async def disconnect(self):
        self.is_running = False
        if self.websocket and self._is_ws_connected(self.websocket):
            await self.websocket.close()
            logger.info("WebSocket connection closed manually.")
        else: 
            logger.error("No WebSocket connection to close.")

    # Send Order
    def send_order(self):
        pass
    
    # Cancel Order
    def cancel_order(self):
        pass

    # Subscribe
    async def subscribe(self, channels):
        if not self.websocket or not self._is_ws_connected(self.websocket):
            logger.error("WebSocket is not connected. Cannot subscribe.")
            return
        if not isinstance(channels, list):
            logger.error("Channels must be a list.")
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

    # Get Position
    def get_position(self):
        pass

# async def main():
#     client = DeribitClient()
#     connect_task = asyncio.create_task(client.connect())
#     await asyncio.sleep(3)

#     await client.subscribe(["ticker.BTC-PERPETUAL.raw"])
#     await asyncio.sleep(10)
#     await client.unsubscribe(["ticker.BTC-PERPETUAL.raw"])

#     await client.disconnect()
#     await connect_task

# if __name__ == "__main__":
#     asyncio.run(main())