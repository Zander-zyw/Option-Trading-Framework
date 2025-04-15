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
import json
import signal

class StraddleClient(DeribitClient):
    def __init__(self, symbol, side, leverage):
        super().__init__()
        self.symbol = symbol
        self.side = side
        self.leverage = leverage

        if self.symbol == "BTC":
            self.base_amount = 0.1
        elif self.symbol == "ETH":
            self.base_amount = 1

        self.is_running = True
        self.active_positions = {}
        self.position_monitor = None
        self._setup_signal_handlers()
        
        # Account monitoring variables
        self.peak_equity = None
        self.initial_equity = None
        self.weekly_initial_equity = None
    
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
                'peak_equity': self.peak_equity,
                'initial_equity': self.initial_equity,
                'weekly_initial_equity': self.weekly_initial_equity,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            state_file = os.path.join(BASE_DIR, 'State', f'straddle_state_{self.symbol}.json')
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=4)
            
            logger.info(f"State saved to {state_file}")
        except Exception as e:
            logger.error(f"Failed to save state: {str(e)}")

    async def load_state(self):
        """Load saved state from file"""
        try:
            state_file = os.path.join(BASE_DIR, 'State', f'straddle_state_{self.symbol}.json')
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.active_positions = state.get('active_positions', {})
                    self.peak_equity = state.get('peak_equity')
                    self.initial_equity = state.get('initial_equity')
                    self.weekly_initial_equity = state.get('weekly_initial_equity')
                    logger.info(f"Loaded state from {state_file}")
            else:
                logger.info(f"No existing state file found at {state_file}, starting fresh")
                self.active_positions = {}
                self.peak_equity = None
                self.initial_equity = None
                self.weekly_initial_equity = None
        except Exception as e:
            logger.error(f"Failed to load state: {str(e)}")
            logger.info("Starting with fresh state")
            self.active_positions = {}
            self.peak_equity = None
            self.initial_equity = None
            self.weekly_initial_equity = None

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

    # Get the remaining amount of an order
    async def get_order_state(self, order_id):
        response = await super().get_order_state(order_id)

        return response["result"]["amount"] - response["result"]["filled_amount"]

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
            return response["result"]["average_price"], response["result"]["size"], response["result"]["mark_price"]
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

    async def close_positions(self, current_equity, ratio=1.0):
        """Close positions with specified ratio (1.0 for all positions)"""
        # Get all active positions
        positions = list(self.active_positions.items())
        if len(positions) != 2:  # Should have exactly 2 positions (call and put)
            logger.error(f"Unexpected number of positions: {len(positions)}")
            return

        # Check if the instrument name ends with "C" or "P"
        for instrument_name, position_info in positions:
            if instrument_name.endswith("C"):
                call_instrument = instrument_name
                call_info = position_info
            else:
                put_instrument = instrument_name
                put_info = position_info

        try:
            # Get order book for both options
            call_best_ask, call_best_bid = await self.get_order_book(call_instrument, 1)
            put_best_ask, put_best_bid = await self.get_order_book(put_instrument, 1)
            
            # Use best_bid for buy positions and best_ask for sell positions
            call_close_price = call_best_ask if self.side == "buy" else call_best_bid
            put_close_price = put_best_ask if self.side == "buy" else put_best_bid
            
            # Calculate close amount
            call_close_amount = abs(call_info["amount"]) * ratio
            put_close_amount = abs(put_info["amount"]) * ratio
            
            # Round to base amount
            call_close_amount = math.floor(call_close_amount / self.base_amount) * self.base_amount
            put_close_amount = math.floor(put_close_amount / self.base_amount) * self.base_amount
            
            # Send close orders for both positions
            call_order_id = await self.send_order(
                side="sell" if self.side == "buy" else "buy",
                order_type="limit",
                instrument_name=call_instrument,
                price=call_close_price,
                amount=call_close_amount
            )
            
            put_order_id = await self.send_order(
                side="sell" if self.side == "buy" else "buy",
                order_type="limit",
                instrument_name=put_instrument,
                price=put_close_price,
                amount=put_close_amount
            )
            
            if call_order_id and put_order_id:
                await self._wait_order_fill(call_order_id, put_order_id, call_instrument, put_instrument)
                logger.info(f"Positions closed - Call: {call_instrument} at {call_close_price}, "
                          f"Put: {put_instrument} at {put_close_price}")
                
                if ratio == 1.0:
                    # Remove positions if closing all
                    del self.active_positions[call_instrument]
                    del self.active_positions[put_instrument]
                else:
                    # Update remaining positions
                    self.active_positions[call_instrument]["amount"] -= call_close_amount
                    self.active_positions[put_instrument]["amount"] -= put_close_amount
        
        except Exception as e:
            logger.error(f"Error closing positions: {str(e)}")
            return
        
        # Save state after closing positions
        await self.save_state()
        
        # Update peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            await self.save_state()

    async def monitor_positions(self):
        while self.is_running:
            try:
                # Get current account equity
                current_equity = await self.get_account_summary(currency=self.symbol)
                if not current_equity:
                    await asyncio.sleep(60)
                    continue

                # Initialize peak equity if not set
                if self.peak_equity is None:
                    self.peak_equity = current_equity
                    self.initial_equity = current_equity
                    await self.save_state()

                # Update peak equity if current equity is higher
                if current_equity > self.peak_equity:
                    self.peak_equity = current_equity
                    await self.save_state()

                # Check if it's UTC Friday 8:00 (new week)
                now = datetime.now(timezone.utc)
                is_friday_8am = now.weekday() == 4 and now.hour == 8 and now.minute < 1  # Friday 8:00 UTC
                
                if self.weekly_initial_equity is None or is_friday_8am:
                    self.weekly_initial_equity = current_equity
                    await self.save_state()
                    logger.info(f"New week started at {now} - Weekly initial equity: {self.weekly_initial_equity}")

                # Calculate weekly return
                weekly_return = (current_equity - self.weekly_initial_equity) / self.weekly_initial_equity
                
                logger.info(f"Account Equity: {current_equity}, Weekly Initial Equity: {self.weekly_initial_equity}, "
                          f"Weekly Return: {weekly_return*100:.2f}%")

                # Check for take profit conditions
                if weekly_return >= 0.01:  # 1% weekly return
                    logger.info(f"Take profit triggered at weekly return {weekly_return*100:.2f}% - Closing all positions")
                    await self.close_positions(current_equity, ratio=1.0)
                elif weekly_return >= 0.008:  # 0.8% weekly return
                    logger.info(f"Partial take profit triggered at weekly return {weekly_return*100:.2f}% - Closing 50% of positions")
                    await self.close_positions(current_equity, ratio=0.5)

                # Check if drawdown exceeds 1%
                drawdown = (self.peak_equity - current_equity) / self.peak_equity
                
                logger.info(f"Account Equity: {current_equity}, Peak Equity: {self.peak_equity}, "
                          f"Drawdown: {drawdown*100:.2f}%")
                
                if drawdown >= 0.01:  # 1% drawdown
                    logger.warning(f"Account stop loss triggered at equity {current_equity} "
                                f"(Drawdown: {drawdown*100:.2f}%)")
                    await self.close_positions(current_equity, ratio=1.0)
                
                await asyncio.sleep(60)  # Check every 60 seconds
                
            except Exception as e:
                logger.error(f"Error in account monitoring: {str(e)}")
                await asyncio.sleep(60)

    # Main Logic to execute the straddle strategy
    async def execute(self):
        await self.connect()

        # Load previous state
        await self.load_state()

        # Start position monitoring in background
        self.position_monitor = asyncio.create_task(self.monitor_positions())

        try:
            while self.is_running:
                # Wait until next Friday 17:00 UTC
                await self._wait_until_execution()

                # Get account equity
                equity = await self.get_account_summary(currency=self.symbol)
                if not equity:
                    logger.error("Failed to get account equity")
                    await asyncio.sleep(60)
                    continue

                # Calculate position size based on 1x leverage
                position_size = 0.5 * equity * self.leverage  # 1x leverage
                position_size = math.floor(position_size / self.base_amount) * self.base_amount

                # Get the mark price for the next expiry date
                instrument_name = f"{self.symbol}-{self._get_next_expiry().strftime('%-d%b%y').upper()}"
                mark_price = await self.ticker(instrument_name=instrument_name)

                if not mark_price:
                    await asyncio.sleep(60)
                    continue

                # Get call and put options for next week
                call_options, put_options = await self.get_instruments(currency=self.symbol, kind="option")

                if not call_options or not put_options:
                    logger.error("No suitable options found for strangle strategy")
                    await asyncio.sleep(60)
                    continue

                # Strangle strategy (strike price is different for call and put)
                # For call: strike > mark price
                # For put: strike < mark price
                call_option = next((option for option in call_options if option['strike'] > mark_price), None)
                put_option = next((option for option in put_options if option['strike'] < mark_price), None)

                if not call_option or not put_option:
                    logger.error("No suitable options found for strangle strategy")
                    await asyncio.sleep(60)
                    continue

                logger.info(f"Call option: {call_option['instrument_name']}, Put option: {put_option['instrument_name']}")

                # Get the order book for both options
                call_best_ask, _ = await self.get_order_book(call_option['instrument_name'], 5)
                put_best_ask, _ = await self.get_order_book(put_option['instrument_name'], 5)

                # Execute the strangle strategy
                call_order_id = await self.send_order(
                    side=self.side,
                    order_type="limit",
                    instrument_name=call_option['instrument_name'],
                    price=call_best_ask,
                    amount=position_size
                )
                
                put_order_id = await self.send_order(
                    side=self.side,
                    order_type="limit",
                    instrument_name=put_option['instrument_name'],
                    price=put_best_ask,
                    amount=position_size
                )

                if call_order_id and put_order_id:
                    await self._wait_order_fill(call_order_id, put_order_id, call_option['instrument_name'], put_option['instrument_name'])

                    # Get position details after fill
                    call_price, call_amount, _ = await self.get_position_by_instrument_name(call_option['instrument_name'])
                    put_price, put_amount, _ = await self.get_position_by_instrument_name(put_option['instrument_name'])

                    # Update active positions
                    self.active_positions[call_option['instrument_name']] = {
                        'entry_price': call_price - max(0.0003, call_price * 0.0003),
                        'amount': call_amount
                    }
                    self.active_positions[put_option['instrument_name']] = {
                        'entry_price': put_price - max(0.0003, put_price * 0.0003),
                        'amount': put_amount
                    }

                    # Save state after opening new positions
                    await self.save_state()

        except Exception as e:
            logger.error(f"Error in execution loop: {str(e)}")
        finally:
            await self.shutdown()

if __name__ == "__main__":
    symbol = "BTC"
    side = "sell"
    leverage = 1.0
    straddle_client = StraddleClient(symbol, side, leverage)
    
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