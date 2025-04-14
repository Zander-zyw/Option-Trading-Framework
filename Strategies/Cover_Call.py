### import DeribitClient ###
import sys
import os
import signal

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from Deribit.DeribitClient import DeribitClient
from Logger.Logger import logger
### import DeribitClient ###

import json
import asyncio
from datetime import datetime, timedelta, timezone, time
import math

class CoverCallClient(DeribitClient):
    def __init__(self, symbol, position_thresholds, stop_loss_multiplier, call_level):
        super().__init__()
        self.symbol = symbol
        self.position_thresholds = position_thresholds
        self.stop_loss_multiplier = stop_loss_multiplier
        self.call_level = call_level
                
        self.is_running = True
        self.active_positions = {}  # Track active positions and their entry prices
        self.position_monitor = None
        self._setup_signal_handlers()

        if self.symbol == "BTC":
            self.base_amount = 0.1
        elif self.symbol == "ETH":
            self.base_amount = 1

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
            state_file = os.path.join(BASE_DIR, 'State', f'cover_call_state_{self.symbol}.json')
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
                best_ask, _ = await self.get_order_book(order_name, 1)
                order_id = await self.send_order(side="sell", order_type="limit", instrument_name=order_name, price=best_ask, amount=remaining_call)

            else:
                await self.cancel_order(order_id)
                _, best_bid = await self.get_order_book(order_name, 1)
                order_id = await self.send_order(side="sell", order_type="limit", instrument_name=order_name, price=best_bid, amount=remaining_call)
            
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

    async def get_account_summary(self, currency, extended=True):
        response = await super().get_account_summary(currency, extended)

        if response:
            return response["result"]["equity"]
        else:
            return None

    async def get_positions(self, currency, kind):
        response = await super().get_positions(currency, kind)

        if response:
            total_positions = sum(abs(float(position["size"])) for position in response["result"])
            logger.info(f"Total positions: {total_positions}")
            return total_positions
        else:
            return None

    async def get_position_by_instrument_name(self, instrument_name):
        response = await super().get_position_by_instrument_name(instrument_name=instrument_name)

        if response:
            return response["result"]["average_price"], response["result"]["size"], response["result"]["mark_price"]
        else:
            return None
    
    async def get_order_state(self, order_id):
        response = await super().get_order_state(order_id)

        return response["result"]["amount"] - response["result"]["filled_amount"]

    async def calculate_target_position(self, current_iv, current_position, equity):
        target_leverage = 0
        for threshold, leverage in sorted(self.position_thresholds.items()):
            if current_iv >= threshold and leverage > target_leverage:
                target_leverage = leverage
        
        if target_leverage == 0:
            return 0
        
        # 如果当前仓位已经达到或超过目标杠杆，不增加仓位
        current_leverage = abs(current_position) / equity if equity > 0 else 0
        if current_leverage >= target_leverage:
            logger.info(f"Current leverage {current_leverage} already meets or exceeds target {target_leverage}")
            return 0
        
        # 计算需要增加的仓位
        target_position = equity * target_leverage
        position_to_add = target_position - abs(current_position)

        # 取到0.1的倍数
        position_to_add = math.floor(position_to_add / self.base_amount) * self.base_amount
        
        return position_to_add

    async def monitor_positions(self):
        while self.is_running:
            try:
                # Skip monitoring if active_positions is empty
                if len(self.active_positions) == 0:
                    await asyncio.sleep(60)  # Check less frequently when no positions
                    continue

                # Only monitor positions this strategy opened
                for instrument_name, position_info in list(self.active_positions.items()):
                    try:
                        # Get current position data
                        position_data = await self.get_position_by_instrument_name(instrument_name=instrument_name)
                        if not position_data:
                            logger.warning(f"No position data found for {instrument_name}, skipping...")
                            continue

                        average_price, size, mark_price = position_data
                        
                        # Calculate stop loss price based on entry price
                        stop_loss_price = average_price * self.stop_loss_multiplier
                        
                        logger.info(f"Position: {instrument_name}, Size: {size}, Mark Price: {mark_price}, Stop Loss: {stop_loss_price}")
                        
                        # Check if we need to close position
                        if mark_price >= stop_loss_price:
                            logger.warning(f"Stop loss triggered for {instrument_name} at mark price {mark_price}")
                            
                            # Determine closing side based on position size
                            close_side = "buy" if size < 0 else "sell"
                            size_abs = abs(size)
                            
                            # Get best price for closing
                            best_ask, best_bid = await self.get_order_book(instrument_name, 1)
                            close_price = best_bid if close_side == "buy" else best_ask
                            
                            logger.info(f"Closing position with {close_side} order at price {close_price}")
                            
                            # Send close order
                            order_id = await self.send_order(
                                side=close_side,
                                order_type="limit",
                                instrument_name=instrument_name,
                                price=close_price,
                                amount=size_abs
                            )
                            
                            if order_id:
                                await self._wait_order_fill(order_id, instrument_name)
                                logger.info(f"Position closed for {instrument_name}")
                                
                                # Remove from active positions
                                del self.active_positions[instrument_name]
                    
                    except Exception as e:
                        logger.error(f"Error monitoring position {instrument_name}: {str(e)}")
                        continue
                
                await asyncio.sleep(60)  # Check positions every 60 seconds
                
            except Exception as e:
                logger.error(f"Error in position monitoring: {str(e)}")
                await asyncio.sleep(60)

    async def execute(self):
        await self.connect()
        
        # Load previous state
        await self.load_state()
        
        # Start position monitoring in background
        self.position_monitor = asyncio.create_task(self.monitor_positions())
        
        try:
            while self.is_running:
                try:
                    # Get account equity
                    equity = await self.get_account_summary(currency=self.symbol)
                    if not equity:
                        logger.error("Failed to get account equity")
                        await asyncio.sleep(60)
                        continue

                    # Get current positions
                    current_position = await self.get_positions(currency=self.symbol, kind="option")
                    if current_position is None:
                        logger.error("Failed to get current positions")
                        await asyncio.sleep(60)
                        continue

                    # Get the mark price for the next expiry date
                    instrument_name = f"{self.symbol}-{self._get_next_expiry().strftime('%-d%b%y').upper()}"
                    mark_price = await self.ticker(instrument_name=instrument_name)

                    if not mark_price:
                        await asyncio.sleep(60)
                        continue
                    
                    call_options = await self.get_instruments(currency=self.symbol, kind="option")
                    if not call_options:
                        await asyncio.sleep(60)
                        continue

                    # Find the first call option with strike price above mark price
                    call1 = next((option for option in call_options if option['strike'] > mark_price), None)      

                    if not call1:
                        logger.error("No suitable options found for cover call strategy")
                        await asyncio.sleep(60)
                        continue
                    
                    call1_iv = await self.get_call1_iv(call1['instrument_name'])

                    if call1_iv is None:
                        await asyncio.sleep(60)
                        continue
                    
                    logger.info(f"Current IV: {call1_iv}, Current Position: {current_position}, Equity: {equity}")
                    
                    # Calculate target position based on IV
                    position_to_add = await self.calculate_target_position(call1_iv, current_position, equity)
                    
                    if position_to_add > 0:
                        logger.info(f"IV {call1_iv} triggers position increase of {position_to_add}")
                        
                        call_price = mark_price * self.call_level
                        call_option = next((option for option in call_options if option['strike'] > call_price), None)
                        
                        if call_option:
                            best_ask_price, _ = await self.get_order_book(call_option['instrument_name'], 1)
                            order_id = await self.send_order(
                                side="sell",
                                order_type="limit",
                                instrument_name=call_option["instrument_name"],
                                price=best_ask_price,
                                amount=position_to_add
                            )
                            
                            if order_id:
                                await self._wait_order_fill(order_id, call_option["instrument_name"])
                                # Add to active positions tracking
                                average_price, amount, _ = await self.get_position_by_instrument_name(call_option["instrument_name"])
                                self.active_positions[call_option["instrument_name"]] = {
                                    "entry_price": average_price - max(0.0003, average_price * 0.0003),
                                    "amount": amount
                                }
                                logger.info(f"New position added: {call_option['instrument_name']}, Amount: {position_to_add}, Entry price: {best_ask_price}")
                    
                    await asyncio.sleep(60)  # Check every minute
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {str(e)}")
                    await asyncio.sleep(60)
        finally:
            # Ensure proper cleanup
            await self.shutdown()

if __name__ == "__main__":
    symbol = "ETH"
    position_thresholds = {
        60: 0.5,  # 半仓
        70: 1.0,  # 满仓
        80: 1.5,  # 1.5倍杠杆
        100: 2.0  # 2倍杠杆
    }
    stop_loss_multiplier = 4.0
    call_level = 1.2
    cover_call_client = CoverCallClient(
        symbol=symbol,
        position_thresholds=position_thresholds,
        stop_loss_multiplier=stop_loss_multiplier,
        call_level=call_level
    )
    
    try:
        asyncio.run(cover_call_client.execute())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        # Ensure the event loop is properly closed
        loop = asyncio.get_event_loop()
        loop.close()