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

class DeltaNeutralClient(DeribitClient):
    def __init__(self, symbol, strangle_ratio, position_thresholds, hedging_threshold):
        super().__init__()
        self.symbol = symbol
        self.strangle_ratio = strangle_ratio
        self.position_thresholds = position_thresholds
        self.hedging_threshold = hedging_threshold
        if self.symbol == "BTC":
            self.base_amount = 0.1
        else:   # ETH
            self.base_amount = 1

        # == Initialize variables ==
        self.is_running = True
        self.is_hedging = False
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
    
    # Wait until the next Friday 8:30 UTC or execute directly if past
    async def _wait_until_execution(self):
        """Wait until next Friday at 8:30 UTC"""
        now = datetime.now(timezone.utc)
        
        # Calculate days until next Friday (0 = Monday, 1 = Tuesday, ..., 4 = Friday)
        days_until_friday = (4 - now.weekday()) % 7
        if days_until_friday == 0 and now.hour < 8 or (now.hour == 8 and now.minute < 30):
            # If it's Friday before 8:30 UTC, wait until 8:30
            days_until_friday = 0
        elif days_until_friday == 0:
            # If it's Friday after 8:30 UTC, wait until next Friday
            days_until_friday = 7
        
        # Calculate next execution time
        next_execution = now.replace(hour=8, minute=30, second=0, microsecond=0) + timedelta(days=days_until_friday)
        
        # Calculate seconds to wait
        seconds_to_wait = (next_execution - now).total_seconds()
        
        if seconds_to_wait > 0:
            logger.info(f"Waiting until {next_execution} UTC ({seconds_to_wait/3600:.2f} hours)")
            await asyncio.sleep(seconds_to_wait)

    # == Get account summary ==
    async def get_account_summary(self, currency, extended=True):
        response = await super().get_account_summary(currency, extended)

        if response:
            return response["result"]["equity"]
        else:
            return None
        
    # == Get the mark price for the next expiry date ==
    async def ticker(self, instrument_name: str):
        response = await super().ticker(instrument_name=instrument_name)

        if response:
            symbol_mark_px = response["result"]["mark_price"]
            logger.info(f"Mark price for {instrument_name}: {symbol_mark_px}")
            return symbol_mark_px
        else:
            logger.error(f"Failed to get ticker for {instrument_name}")
            return None
    
    # == Get the option chain for the next expiry date ==
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
            return None
    
    # == Get position details by instrument name ==
    async def get_position_by_instrument_name(self, instrument_name):
        response = await super().get_position_by_instrument_name(instrument_name=instrument_name)

        if response:
            return response["result"]["average_price"], response["result"]["size"], response["result"]["mark_price"]
        else:
            return None
        
    # == Fetch best ask and bid prices ==
    async def get_order_book(self, instrument_name, depth):
        response = await super().get_order_book(instrument_name, depth)

        best_ask = response["result"]["best_ask_price"]
        best_bid = response["result"]["best_bid_price"]
        logger.info(f"Best ask price for {instrument_name}: {best_ask}")
        logger.info(f"Best bid price for {instrument_name}: {best_bid}")

        return best_ask, best_bid
    
    # == Get remaining order amount ==
    async def get_order_state(self, order_id):
        response = await super().get_order_state(order_id)

        return response["result"]["amount"] - response["result"]["filled_amount"]
    
    # == Get the mark IV for call1 option ==
    async def get_call1_iv(self, instrument_name):
        response = await super().get_order_book(instrument_name, 1)

        if response:
            call1_iv = response["result"]["mark_iv"]
            logger.info(f"Mark IV for {instrument_name}: {call1_iv}")
            return call1_iv
        else:
            logger.error(f"Failed to get mark IV for {instrument_name}")
            return None
        
    # == Get the maintenance margin ==
    async def get_maintenance_margin(self, currency, extended=True):
        response = await super().get_account_summary(currency, extended)

        if response:
            return response["result"]["maintenance_margin"]
        else:
            logger.error(f"Failed to get maintenance margin for {currency}")
            return None
        
    # == Wait for orders to fill ==
    async def wait_for_orders_to_fill(self, call_order_id, call_name, put_order_id, put_name):
        timeout = 3600
        call_filled = False
        put_filled = False
        start_time = None  # 开始计时的时间
        
        while self.is_running:
            ### deactivate_hedging() ###
            remaining_call = await self.get_order_state(call_order_id)
            remaining_put = await self.get_order_state(put_order_id)
            
            print(f"Remaining Call Order: {remaining_call}")
            print(f"Remaining Put Order: {remaining_put}")
            
            # 如果两个订单都已成交，则跳出循环
            if remaining_call == 0 and remaining_put == 0:
                print("Both orders have been filled.")
                break
            
            if not call_filled and remaining_call == 0:
                call_filled = True
                if not start_time:
                    start_time = datetime.now()
                print("Call Order has been filled.")
            
            if not put_filled and remaining_put == 0:
                put_filled = True
                if not start_time:
                    start_time = datetime.now()
                print("Put Order has been filled.")

            # 如果订单在规定时间内没有成交，则取消订单
            if start_time:
                if (datetime.now() - start_time).total_seconds() <= timeout:
                    if call_filled and remaining_put > 0:
                        await self.cancel_order(put_order_id)
                        best_ask_put, _ = await self.get_order_book(put_name, 1)
                        put_order_id = await self.send_order('sell', 'limit', put_name, best_ask_put, remaining_put)
                    if put_filled and remaining_call > 0:
                        await self.cancel_order(call_order_id)
                        best_ask_call, _ = await self.get_order_book(call_name, 1)
                        call_order_id = await self.send_order('sell', 'limit', call_name, best_ask_call, remaining_call)
                else:
                    if call_filled and remaining_put > 0:
                        # Put Order Timeout
                        await self.cancel_order(put_order_id)
                        best_bid_put, _ = await self.get_order_book(put_name, 1)
                        put_order_id = await self.send_order('sell', 'limit', put_name, best_bid_put, remaining_put)
                        
                    if put_filled and remaining_call > 0:
                        await self.cancel_order(call_order_id)
                        best_bid_call, _ = await self.get_order_book(call_name, 1)
                        call_order_id = await self.send_order('sell', 'limit', call_name, best_bid_call, remaining_call)
                        
            # 等待1分钟后再次检查
            await asyncio.sleep(60)

    # == Calculate target position ==
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

    # == Activate hedging ==
    async def activate_hedging(self):
        self.is_hedging = True

    # == Deactivate hedging ==
    async def deactivate_hedging(self):
        self.is_hedging = False

        # == Monitor account delta to hedge ==
    async def monitor_delta(self):
        """Monitor and hedge delta exposure"""
        while self.is_running:
            try:
                # Check if hedging is activated
                if not self.is_hedging:
                    logger.info("Hedging is not activated, skipping delta check")
                    await asyncio.sleep(60)
                    continue

                # Calculate total delta exposure from active positions
                total_delta = 0
                for instrument_name, position in self.active_positions.items():
                    # Get option details including delta
                    option_details = await self.get_instrument_details(instrument_name)
                    if option_details:
                        delta = option_details.get('delta', 0)
                        total_delta += delta * position['amount']

                logger.info(f"Current total delta exposure: {total_delta}")

                # Get next Friday's date
                friday_perpetual_name = f"{self.symbol}-{self._get_next_expiry().strftime('%-d%b%y').upper()}"
                
                # Get current perpetual position
                for p in self.active_positions:
                    if p == friday_perpetual_name:
                        current_perpetual_size = self.active_positions[p]['amount']
                    else:
                        current_perpetual_size = 0
                    break

                # Calculate required hedge
                position_delta = total_delta + current_perpetual_size

                if position_delta > self.hedging_threshold:
                    # Need to short perpetual to hedge positive delta
                    hedge_size = math.floor(position_delta / self.hedging_threshold) * self.hedging_threshold
                    await self.send_order(
                        side="sell",
                        order_type="market",
                        instrument_name=friday_perpetual_name,
                        amount=hedge_size
                    )
                    # Update active positions
                    if friday_perpetual_name in self.active_positions:
                        self.active_positions[friday_perpetual_name]['amount'] -= hedge_size
                    else:
                        self.active_positions[friday_perpetual_name] = {'amount': -hedge_size}
                    logger.info(f"Hedging positive delta: shorting {hedge_size} {self.symbol} in {friday_perpetual_name}")
                
                elif position_delta < -self.hedging_threshold:
                    # Need to long perpetual to hedge negative delta
                    hedge_size = math.floor(abs(position_delta) / self.hedging_threshold) * self.hedging_threshold
                    await self.send_order(
                        side="buy",
                        order_type="market",
                        instrument_name=friday_perpetual_name,
                        amount=hedge_size
                    )
                    # Update active positions
                    if friday_perpetual_name in self.active_positions:
                        self.active_positions[friday_perpetual_name]['amount'] += hedge_size
                    else:
                        self.active_positions[friday_perpetual_name] = {'amount': hedge_size}
                    logger.info(f"Hedging negative delta: longing {hedge_size} {self.symbol} in {friday_perpetual_name}")

                # Save state after each hedge operation
                await self.save_state()

                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Error in delta monitoring: {str(e)}")
                await asyncio.sleep(3600)

    # == Execute the strategy ==
    async def execute(self):
        await self.connect()
        
        # Load previous state
        await self.load_state()
        
        # Start position monitoring in background
        self.position_monitor = asyncio.create_task(self.monitor_delta())
        
        try:
            while self.is_running:
                try:
                    # Wait until next Friday 8:30 UTC
                    await self._wait_until_execution()
                    
                    # Deactivate hedging
                    await self.deactivate_hedging()

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
                    
                    call_options, put_options = await self.get_instruments(currency=self.symbol, kind="option")
                    if not call_options or not put_options:
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
                    
                    logger.info(f"Current IV: {call1_iv}, Equity: {equity}")
                    
                    # Place strangle orders
                    call_price = mark_price * self.strangle_ratio['call']
                    put_price = mark_price / self.strangle_ratio['put']

                    call_option = next((option for option in call_options if option['strike'] > call_price), None)
                    put_option = next((option for option in put_options if option['strike'] < put_price), None)
                    
                    if call_option and put_option:
                        best_ask_call, _ = await self.get_order_book(call_option['instrument_name'], 1)
                        best_ask_put, _ = await self.get_order_book(put_option['instrument_name'], 1)

                        call_order_id = await self.send_order(
                            side="sell",
                            order_type="limit",
                            instrument_name=call_option["instrument_name"],
                            price=best_ask_call,
                            amount=position_size
                        )

                        put_order_id = await self.send_order(
                            side="sell",
                            order_type="limit",
                            instrument_name=put_option["instrument_name"],
                            price=best_ask_put,
                            amount=position_size
                        )
                        
                        if call_order_id and put_order_id:
                            await self._wait_order_fill(call_order_id, call_option["instrument_name"], put_order_id, put_option["instrument_name"])

                            # Activate hedging
                            await self.activate_hedging()

                            # Add to active positions tracking
                            call_average_price, call_amount, _ = await self.get_position_by_instrument_name(call_option["instrument_name"])
                            put_average_price, put_amount, _ = await self.get_position_by_instrument_name(put_option["instrument_name"])
                            self.active_positions[call_option["instrument_name"]] = {
                                "entry_price": call_average_price - max(0.0003, call_average_price * 0.0003),
                                "amount": call_amount
                            }
                            self.active_positions[put_option["instrument_name"]] = {
                                "entry_price": put_average_price - max(0.0003, put_average_price * 0.0003),
                                "amount": put_amount
                            }
                            logger.info(f"New position added: {call_option['instrument_name']}, Amount: {position_size}, Entry price: {best_ask_call}")
                            logger.info(f"New position added: {put_option['instrument_name']}, Amount: {position_size}, Entry price: {best_ask_put}")

                            # save state
                            await self.save_state()
                    
                    await asyncio.sleep(60)  # Check every minute
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {str(e)}")
                    await asyncio.sleep(60)
        finally:
            # Ensure proper cleanup
            await self.shutdown()