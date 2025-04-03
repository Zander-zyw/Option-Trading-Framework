### import DeribitClient ###
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from Deribit.DeribitClient import DeribitClient
from Logger.Logger import logger
### import DeribitClient ###

from datetime import datetime, timedelta, timezone, time
import asyncio

class CoverCallClient(DeribitClient):
    def __init__(self, symbol, iv_threshold):
        super().__init__()
        self.symbol = symbol
        self.iv_threshold = iv_threshold

    # Get the next expiry date for the given symbol
    def _get_next_expiry(self):
        today = datetime.now(timezone.utc)
        days_until_friday = (4 - today.weekday()) % 7  # friday is 4
        if days_until_friday == 0: # today is Friday
            next_friday = today + timedelta(days=(days_until_friday + 7))
        else: # today is not Friday
            next_friday = today + timedelta(days=days_until_friday)
            
        return next_friday
    
    # Wait until order is filled
    async def _wait_order_fill(self, order_id, order_name):
        start_time = datetime.now()
        timeout = 3600

        while self.is_running:
            remaining_call = await self.get_order_state(order_id)

            if remaining_call == 0:
                logger.info(f"Cover call orders have been filled.")
                break

            if (datetime.now() - start_time).total_seconds() <= timeout:
                await self.cancel_order(order_id)
                best_ask, _ = await self.get_order_book(order_name)
                order_id = await self.send_order(side="sell", order_type="limit", instrument_name=order_name, price=best_ask, amount=remaining_call)

            else:
                await self.cancel_order(order_id)
                _, best_bid = await self.get_order_book(order_name)
                order_id = await self.send_order(side="sell", order_type="limit", instrument_name=order_name, price=best_bid, amount=remaining_call)
            
            asyncio.sleep(60)
    
    # Get the mark price for the next expiry date
    async def ticker(self, instrument_name: str):
        response = await super().ticker(instrument_name=instrument_name)

        if response:
            symbol_mark_px = response["result"]["mark_price"]
            logger.info(f"Mark price for {instrument_name}: {symbol_mark_px}")
            return symbol_mark_px
        else:
            logger.error(f"Failed to get ticker for {instrument_name}")
            return None
        
    async def send_order(self, side, order_type, instrument_name, price, amount):
        response = await super().send_order(side, order_type, instrument_name, price, amount)

        if response:
            return response['result']['order']['order_id']
        else:
            return None
        
    # Get the option chain for the next expiry date
    async def get_instruments(self, currency, kind):
        response = await super().get_instruments(currency, kind)

        if response:
            instruments = response["result"]

            # Filter for options expiring next Friday
            next_friday = self._get_next_expiry()
            next_friday_utc8 = next_friday.replace(hour=8, minute=0, second=0, microsecond=0)
            next_friday_timestamp = int(next_friday_utc8.timestamp() * 1000)

            call_options = [
                instrument for instrument in instruments
                if instrument['expiration_timestamp'] == next_friday_timestamp and instrument['option_type'] == 'call'
            ]

            return call_options
        else:
            logger.error("Failed to get instruments")
            return None

    async def get_call1_iv(self, instrument_name):
        response = await super().get_order_book(instrument_name, 1)

        if response:
            call1_iv = response["result"]["mark_iv"]
            logger.info(f"Mark IV for {instrument_name}: {call1_iv}")
            return call1_iv
        else:
            logger.error(f"Failed to get mark IV for {instrument_name}")
            return None
        
    async def get_order_book(self, instrument_name, depth):
        response = await super().get_order_book(instrument_name, depth)

        best_ask = response["result"]["best_ask_price"]
        best_bid = response["result"]["best_bid_price"]
        logger.info(f"Best ask price for {instrument_name}: {best_ask}")
        logger.info(f"Best bid price for {instrument_name}: {best_bid}")

        return best_ask, best_bid
    
    async def get_order_state(self, order_id):
        response = await super().get_order_state(order_id)

        return response["result"]["amount"] - response["result"]["filled_amount"]
        
    async def execute_covercall(self):
        await self.connect()

        # Get the mark price for the next expiry date
        instrument_name = f"{self.symbol}-{self._get_next_expiry().strftime('%-d%b%y').upper()}"
        mark_price = await self.ticker(instrument_name=instrument_name)

        # if mark price is None, exit
        if not mark_price:
            return
        
        call_options = await self.get_instruments(currency="BTC", kind="option")

        # Straddle strategy (strike price is the same for both call and put, and strike > mark price)
        call1 = next((option for option in call_options if option['strike'] > mark_price), None)      

        if not call1:
            logger.error("No suitable options found for cover call strategy")
            return
        
        call1_iv = await self.get_call1_iv(call1['instrument_name'])

        if call1_iv is None:
            return
        elif call1_iv < self.iv_threshold:
            logger.info(f"Call1 IV {call1_iv} is below threshold {self.iv_threshold}, not executing cover call")
            return            
        
        call_price = mark_price * 1.05
        call_option = next((option for option in call_options if option['strike'] > call_price), None)
        best_ask_price, best_bid_price = await self.get_order_book(call_option['instrument_name'], 1)

        # =====================================================================
        # == Execute the cover call strategy here (buy call and put options) ==
        # =====================================================================

        # order_id = await self.send_order(side="sell", order_type="limit", instrument_name=call_option["instrument_name"], price=best_ask_price, amount=0.1)
        # await self._wait_order_fill(order_id)

        await self.disconnect()

if __name__ == "__main__":
    symbol = "BTC"
    iv_threshold = 60

    cover_call_client = CoverCallClient(symbol=symbol, iv_threshold=iv_threshold)
    asyncio.run(cover_call_client.execute_covercall())
