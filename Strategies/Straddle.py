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

class StraddleClient(DeribitClient):
    def __init__(self, symbol):
        super().__init__()
        self.symbol = symbol
        
    def _get_next_expiry(self):
        # Implement logic to get the next expiry date for the given symbol
        today = datetime.now(timezone.utc)
        days_until_friday = (4 - today.weekday()) % 7  # friday is 4
        if days_until_friday == 0: # today is Friday
            next_friday = today + timedelta(days=(days_until_friday + 7))
        else: # today is not Friday
            next_friday = today + timedelta(days=days_until_friday)
            
        return next_friday
    
    # Wait until the next Friday 8:00 UTC or execute directly if past
    async def _wait_until_execution(self):
        now = datetime.now(timezone.utc)
 
        # Calculate next Friday
        days_until_friday = (4 - now.weekday()) % 7  # Friday is the 4th day of the week
        next_friday = now.date() + timedelta(days=days_until_friday)
        execution_time = datetime.combine(next_friday, time(0, 30), timezone.utc)

        # If current time is past this week's Friday 8:00 UTC, move to the next week
        if now >= execution_time:
            execution_time += timedelta(days=7)

        # Calculate the waiting time
        wait_seconds = (execution_time - now).total_seconds()

        print(f"Waiting until next Friday 8:00 UTC. Remaining seconds: {wait_seconds}")
        await asyncio.sleep(wait_seconds)

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
            put_options = [
                instrument for instrument in instruments
                if instrument['expiration_timestamp'] == next_friday_timestamp and instrument['option_type'] == 'put'
            ]

            return call_options, put_options
        else:
            logger.error("Failed to get instruments")
            return None, None
        
    async def execute_straddle(self):
        await self.connect()

        # Get the mark price for the next expiry date
        instrument_name = f"{self.symbol}-{self._get_next_expiry().strftime('%-d%b%y').upper()}"
        mark_price = await self.ticker(instrument_name=instrument_name)

        call_options, put_options = await self.get_instruments(currency="BTC", kind="option")

        # Straddle strategy (strike price is the same for both call and put, and strike > mark price)
        call_option = next((option for option in call_options if option['strike'] > mark_price), None)
        put_option = next((option for option in put_options if option['strike'] > mark_price), None)
        if call_option and put_option:
            logger.info(f"Call option: {call_option['instrument_name']}, Put option: {put_option['instrument_name']}")

            # ===================================================================
            # == Execute the straddle strategy here (buy call and put options) ==
            # ===================================================================
            
        else:
            logger.error("No suitable options found for straddle strategy")
            return
        
        await self.disconnect()

if __name__ == "__main__":
    symbol = "BTC"
    straddle_client = StraddleClient(symbol)
    
    # Execute the straddle strategy
    asyncio.run(straddle_client.execute_straddle())