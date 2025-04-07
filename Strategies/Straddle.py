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
import math

class StraddleClient(DeribitClient):
    def __init__(self, symbol, side):
        super().__init__()
        self.symbol = symbol
        self.side = side

        if self.symbol == "BTC":
            self.base_amount = 0.1
        elif self.symbol == "ETH":
            self.base_amount = 1

        self.is_running = True
        self.active_positions = {}
        self.position_monitor = None
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        if sys.platform != 'win32':  # Windows doesn't support SIGINT properly
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.is_running = False

    async def save_state(self):
        """Save current state to file"""
        try:
            state = {
                'active_positions': self.active_positions,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            state_file = os.path.join(BASE_DIR, 'State', f'cover_call_state_{self.symbol}.json')
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=4)
            
            logger.info(f"State saved to {state_file}")
        except Exception as e:
            logger.error(f"Failed to save state: {str(e)}")

    async def load_state(self):
        """Load saved state from file"""
        try:
            state_file = os.path.join(BASE_DIR, 'state', f'cover_call_state_{self.symbol}.json')
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.active_positions = state.get('active_positions', {})
                    logger.info(f"Loaded {len(self.active_positions)} positions from state file")
            else:
                logger.info(f"No existing state file found at {state_file}, starting with empty positions")
                self.active_positions = {}
        except Exception as e:
            logger.error(f"Failed to load state: {str(e)}")
            logger.info("Starting with empty positions")
            self.active_positions = {}

    async def shutdown(self):
        """Graceful shutdown procedure"""
        logger.info("Starting shutdown procedure...")
        
        # Stop the main loop
        self.is_running = False
        
        # Cancel position monitoring task if it exists
        if self.position_monitor:
            self.position_monitor.cancel()
            try:
                await self.position_monitor
            except asyncio.CancelledError:
                pass
        
        # Save current state
        await self.save_state()
        
        # Disconnect from exchange
        await self.disconnect()
        
        logger.info("Shutdown completed")
    
    # get the next expiry date for the given symbol
    def _get_next_expiry(self):
        today = datetime.now(timezone.utc)
        days_until_friday = (4 - today.weekday()) % 7  # friday is 4
        if days_until_friday == 0: # today is Friday
            next_friday = today + timedelta(days=7)  # next week's Friday
        else: # today is not Friday
            next_friday = today + timedelta(days=days_until_friday)
            
        return next_friday
    
    # Wait until the next Friday 8:00 UTC or execute directly if past
    async def _wait_until_execution(self):
        now = datetime.now(timezone.utc)
 
        # Calculate next Friday
        days_until_friday = (4 - now.weekday()) % 7  # Friday is the 4th day of the week
        next_friday = now.date() + timedelta(days=days_until_friday)
        execution_time = datetime.combine(next_friday, time(8, 30), timezone.utc)

        # If current time is past this week's Friday 8:00 UTC, move to the next week
        if now >= execution_time:
            execution_time += timedelta(days=7)

        # Calculate the waiting time
        wait_seconds = (execution_time - now).total_seconds()

        logger.info(f"Waiting until next Friday 8:30 UTC. Remaining seconds: {wait_seconds}")
        await asyncio.sleep(wait_seconds)

    # Wait until order is filled
    async def _wait_order_fill(self, call_order_id, put_order_id, call_order_name, put_order_name):
        start_time = datetime.now()
        timeout = 3600

        while self.is_running:
            remaining_call = await self.get_order_state(call_order_id)
            remaining_put = await self.get_order_state(put_order_id)

            if remaining_call == 0 and remaining_put == 0:
                logger.info(f"Both call and put orders have been filled.")
                break

            if (datetime.now() - start_time).total_seconds() <= timeout:
                # Handle call order
                if remaining_call > 0:
                    await self.cancel_order(call_order_id)
                    best_ask, _ = await self.get_order_book(call_order_name)
                    call_order_id = await self.send_order(side=self.side, order_type="limit", instrument_name=call_order_name, price=best_ask, amount=remaining_call)
                
                # Handle put order
                if remaining_put > 0:
                    await self.cancel_order(put_order_id)
                    best_ask, _ = await self.get_order_book(put_order_name)
                    put_order_id = await self.send_order(side=self.side, order_type="limit", instrument_name=put_order_name, price=best_ask, amount=remaining_put)

            else:
                # Handle call order
                if remaining_call > 0:
                    await self.cancel_order(call_order_id)
                    _, best_bid = await self.get_order_book(call_order_name)
                    call_order_id = await self.send_order(side=self.side, order_type="limit", instrument_name=call_order_name, price=best_bid, amount=remaining_call)
                
                # Handle put order
                if remaining_put > 0:
                    await self.cancel_order(put_order_id)
                    _, best_bid = await self.get_order_book(put_order_name)
                    put_order_id = await self.send_order(side=self.side, order_type="limit", instrument_name=put_order_name, price=best_bid, amount=remaining_put)
            
            await asyncio.sleep(60)

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
        
    async def get_order_book(self, instrument_name, depth):
        response = await super().get_order_book(instrument_name, depth)

        best_ask = response["result"]["best_ask_price"]
        best_bid = response["result"]["best_bid_price"]
        logger.info(f"Best ask price for {instrument_name}: {best_ask}")
        logger.info(f"Best bid price for {instrument_name}: {best_bid}")

        return best_ask, best_bid

    async def get_position_by_instrument_name(self, instrument_name):
        response = await super().get_position_by_instrument_name(instrument_name)

        if response:
            return response["result"]["settlement_price"], response["result"]["size"]
        else:
            return None

    async def get_account_summary(self, currency, extended=True):
        response = await super().get_account_summary(currency, extended)

        if response:
            return response["result"]["equity"]
        else:
            return None

    async def send_order(self, side, order_type, instrument_name, price, amount):
        response = await super().send_order(side, order_type, instrument_name, price, amount)

        if response:
            return response['result']['order']['order_id']
        else:
            return None
    
    # Main Logic to execute the straddle strategy
    async def execute(self):
        await self.connect()

        # Load previous state
        await self.load_state()

        # Start position monitoring in background
        self.position_monitor = asyncio.create_task(self.monitor_positions())

        try:
            while self.is_running:
                # Check if today is Friday
                today = datetime.now(timezone.utc)
                if today.weekday() != 4:  # Not Friday
                    logger.info("Not Friday, waiting until next Friday...")
                    await self._wait_until_execution()
                    continue

                # Get account equity
                equity = await self.get_account_summary(currency=self.symbol)
                if not equity:
                    logger.error("Failed to get account equity")
                    await asyncio.sleep(60)
                    continue

                # Calculate position size based on 1x leverage
                position_size = equity * 1.0  # 1x leverage
                if self.symbol == "BTC":
                    position_size = math.floor(position_size / 0.1) * 0.1  # Round to nearest 0.1
                elif self.symbol == "ETH":
                    position_size = math.floor(position_size)  # Round to nearest 1

                # Get the mark price for the next expiry date
                instrument_name = f"{self.symbol}-{self._get_next_expiry().strftime('%-d%b%y').upper()}"
                mark_price = await self.ticker(instrument_name=instrument_name)

                if not mark_price:
                    await asyncio.sleep(60)
                    continue

                call_options, put_options = await self.get_instruments(currency=self.symbol, kind="option")

                if not call_options or not put_options:
                    logger.error("No suitable options found for straddle strategy")
                    await asyncio.sleep(60)
                    continue

                # Straddle strategy (strike price is the same for both call and put, and strike > mark price)
                call_option = next((option for option in call_options if option['strike'] > mark_price), None)
                put_option = next((option for option in put_options if option['strike'] > mark_price), None)

                if not call_option or not put_option:
                    logger.error("No suitable options found for straddle strategy")
                    await asyncio.sleep(60)
                    continue

                logger.info(f"Call option: {call_option['instrument_name']}, Put option: {put_option['instrument_name']}")

                # Get the order book for both options
                call_best_ask, call_best_bid = await self.get_order_book(call_option['instrument_name'], 5)
                put_best_ask, put_best_bid = await self.get_order_book(put_option['instrument_name'], 5)

                # Execute the straddle strategy
                call_order_id = await self.send_order(side=self.side, order_type="limit", instrument_name=call_option['instrument_name'], price=call_best_ask, amount=position_size)
                put_order_id = await self.send_order(side=self.side, order_type="limit", instrument_name=put_option['instrument_name'], price=put_best_ask, amount=position_size)

                if call_order_id and put_order_id:
                    await self._wait_order_fill(call_order_id, put_order_id, call_option['instrument_name'], put_option['instrument_name'])
                    self.active_positions[call_option['instrument_name']] = {
                        'entry_price': call_best_ask,
                        'amount': position_size
                    }
                    self.active_positions[put_option['instrument_name']] = {
                        'entry_price': put_best_ask,
                        'amount': position_size
                    }

                # Wait until next Friday
                await self._wait_until_execution()

        except Exception as e:
            logger.error(f"Error in execution loop: {str(e)}")
        finally:
            await self.shutdown()

if __name__ == "__main__":
    symbol = "BTC"
    straddle_client = StraddleClient(symbol)
    
    try:
        # Execute the straddle strategy
        asyncio.run(straddle_client.execute())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        # Ensure the event loop is properly closed
        loop = asyncio.get_event_loop()
        loop.close()