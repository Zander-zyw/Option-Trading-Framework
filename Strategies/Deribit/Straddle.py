### import DeribitClient ###
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from Deribit.DeribitClient import DeribitClient
from Logger.Logger import logger
### import DeribitClient ###

import json
import signal
import argparse
import asyncio
import math

from datetime import datetime, timedelta, timezone

class StraddleClient(DeribitClient):
    def __init__(self, symbol: str, side: str, leverage: float):
        super().__init__()
        self.symbol = symbol.upper()
        self.side = side.lower()
        self.leverage = leverage
        self.base_amount = {'BTC': 0.1, 'ETH': 1.0}.get(self.symbol, 0.01)

        # trading state
        self.active_positions = {}
        self.weekly_initial_equity = None
        self.peak_equity = None

        # asyncio control
        self._shutdown_event = asyncio.Event()
        self._tasks = []

    # ----- Deribit Straddle API wrappers -----
    async def ticker(self, instrument_name: str) -> float:
        resp = await super().ticker(instrument_name=instrument_name)
        if resp and resp.get('result'):
            return resp['result']['mark_price']
        logger.error(f"ticker() failed for {instrument_name}: {resp}")
        return None

    async def get_instruments(self, currency: str, kind: str):
        resp = await super().get_instruments(currency=currency, kind=kind)
        if resp and resp.get('result'):
            instruments = resp['result']
            target_ts = int(self._next_friday().timestamp() * 1000)
            calls = [i for i in instruments if i['expiration_timestamp'] == target_ts and i['option_type'] == 'call']
            puts  = [i for i in instruments if i['expiration_timestamp'] == target_ts and i['option_type'] == 'put']
            return calls, puts
        logger.error(f"get_instruments() failed: {resp}")
        return [], []

    async def get_order_book(self, instrument_name: str, depth: int = 1):
        resp = await super().get_order_book(instrument_name=instrument_name, depth=depth)
        if resp and resp.get('result'):
            return resp['result']['best_ask_price'], resp['result']['best_bid_price']
        logger.error(f"get_order_book() failed for {instrument_name}: {resp}")
        return None, None

    async def get_position_by_instrument_name(self, instrument_name: str):
        resp = await super().get_position_by_instrument_name(instrument_name=instrument_name)
        if resp and resp.get('result'):
            r = resp['result']
            return r['average_price'], r['size'], r['mark_price']
        logger.error(f"get_position_by_instrument_name() failed: {resp}")
        return None, 0, None

    async def get_account_summary(self, currency: str, extended: bool = True) -> float:
        resp = await super().get_account_summary(currency=currency, extended=extended)
        if resp and resp.get('result'):
            return resp['result']['equity']
        logger.error(f"get_account_summary() failed: {resp}")
        return None

    async def send_order(self, side: str, order_type: str, instrument_name: str, price: float, amount: float) -> str:
        resp = await super().send_order(side=side, order_type=order_type,
                                        instrument_name=instrument_name,
                                        price=price, amount=amount)
        if resp and resp.get('result') and resp['result'].get('order'):
            return resp['result']['order']['order_id']
        logger.error(f"send_order() failed: {resp}")
        return None

    async def get_order_state(self, order_id: str) -> int:
        resp = await super().get_order_state(order_id)
        if resp and resp.get('result'):
            res = resp['result']
            return res['amount'] - res['filled_amount']
        logger.error(f"get_order_state() failed: {resp}")
        return None

    # ----- State persistence -----
    async def save_state(self):
        os.makedirs(BASE_DIR, exist_ok=True)
        path = os.path.join(BASE_DIR, f'straddle_{self.symbol}.json')
        data = {
            'active_positions': self.active_positions,
            'weekly_initial_equity': self.weekly_initial_equity,
            'peak_equity': self.peak_equity,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"State saved: {path}")
        except Exception as e:
            logger.error(f"save_state failed: {e}")

    async def load_state(self):
        path = os.path.join(BASE_DIR, f'straddle_{self.symbol}.json')
        if not os.path.exists(path):
            logger.info("No previous state, starting fresh")
            return
        try:
            with open(path) as f:
                s = json.load(f)
            self.active_positions = s.get('active_positions', {})
            self.weekly_initial_equity = s.get('weekly_initial_equity')
            self.peak_equity = s.get('peak_equity')
            logger.info(f"State loaded: {path}")
        except Exception as e:
            logger.error(f"load_state failed: {e}")

    # ----- Position management -----
    async def close_positions(self, current_equity: float, ratio: float = 1.0):
        pos = list(self.active_positions.items())
        if len(pos) != 2:
            logger.error(f"Expected 2 positions, got {len(pos)}")
            return
        call_inst, call_info = next(p for p in pos if p[0].endswith('C'))
        put_inst, put_info   = next(p for p in pos if p[0].endswith('P'))

        ask_c, bid_c = await self.get_order_book(call_inst)
        ask_p, bid_p = await self.get_order_book(put_inst)
        close_side = 'sell' if self.side == 'buy' else 'buy'
        price_c = bid_c if close_side == 'sell' else ask_c
        price_p = bid_p if close_side == 'sell' else ask_p

        amt_c = math.floor(abs(call_info['amount']) * ratio / self.base_amount) * self.base_amount
        amt_p = math.floor(abs(put_info ['amount']) * ratio / self.base_amount) * self.base_amount

        id_c = await self.send_order(close_side, 'limit', call_inst, price_c, amt_c)
        id_p = await self.send_order(close_side, 'limit', put_inst,  price_p, amt_p)

        if id_c and id_p:
            await self._wait_order_fill(id_c, id_p, call_inst, put_inst)
            if ratio == 1.0:
                self.active_positions.clear()
            else:
                self.active_positions[call_inst]['amount'] -= amt_c
                self.active_positions[put_inst]['amount']  -= amt_p
            logger.info(f"Closed positions at ratio={ratio}")
        await self.save_state()

    async def _wait_order_fill(self, call_id: str, put_id: str, call_name: str, put_name: str):
        timeout = 3600
        start = datetime.now()
        while not self._shutdown_event.is_set():
            rem_c = await self.get_order_state(call_id)
            rem_p = await self.get_order_state(put_id)
            if rem_c == 0 and rem_p == 0:
                return
            elapsed = (datetime.now() - start).total_seconds()
            # repricing logic
            side = self.side
            if elapsed < timeout:
                if rem_c > 0:
                    await super().cancel_order(call_id)
                    ask, _ = await self.get_order_book(call_name)
                    call_id = await self.send_order(side, 'limit', call_name, ask, rem_c)
                if rem_p > 0:
                    await super().cancel_order(put_id)
                    ask, _ = await self.get_order_book(put_name)
                    put_id  = await self.send_order(side, 'limit', put_name, ask, rem_p)
            else:
                if rem_c > 0:
                    await super().cancel_order(call_id)
                    _, bid = await self.get_order_book(call_name)
                    call_id = await self.send_order(side, 'limit', call_name, bid, rem_c)
                if rem_p > 0:
                    await super().cancel_order(put_id)
                    _, bid = await self.get_order_book(put_name)
                    put_id  = await self.send_order(side, 'limit', put_name, bid, rem_p)
            await asyncio.sleep(60)

    # ----- Scheduler -----
    def _next_friday(self) -> datetime:
        now = datetime.now(timezone.utc)
        days = (4 - now.weekday()) % 7 or 7
        return (now + timedelta(days=days)).replace(hour=8, minute=30, second=0, microsecond=0)

    async def _weekly_task(self):
        while not self._shutdown_event.is_set():
            next_run = self._next_friday()
            delay = (next_run - datetime.now(timezone.utc)).total_seconds()
            logger.info(f"Sleeping {int(delay)}s until Friday 08:30 UTC...")
            await asyncio.sleep(max(0, delay))
            if self._shutdown_event.is_set(): break
            await self._execute_once()

    async def _execute_once(self):
        expiry = self._next_friday()
        code = expiry.strftime('%d%b%y').upper()
        inst = f"{self.symbol}-{code}"
        mark = await self.ticker(inst)
        if mark is None: return
        calls, puts = await self.get_instruments(self.symbol, 'option')
        call = next((o for o in calls if o['strike'] > mark), None)
        put  = next((o for o in puts  if o['strike'] > mark), None)
        if not call or not put:
            logger.error("No suitable options")
            return
        ask_c, _ = await self.get_order_book(call['instrument_name'])
        ask_p, _ = await self.get_order_book(put ['instrument_name'])
        equity = await self.get_account_summary(self.symbol)
        size = math.floor(0.5 * equity * self.leverage / self.base_amount) * self.base_amount
        id_c = await self.send_order(self.side, 'limit', call['instrument_name'], ask_c, size)
        id_p = await self.send_order(self.side, 'limit', put['instrument_name'], ask_p, size)
        if id_c and id_p:
            await self._wait_order_fill(id_c, id_p, call['instrument_name'], put['instrument_name'])
            c_px, c_sz, _ = await self.get_position_by_instrument_name(call['instrument_name'])
            p_px, p_sz, _ = await self.get_position_by_instrument_name(put['instrument_name'])
            fee = 0.0003
            self.active_positions = {
                call ['instrument_name']: {'entry_price': c_px * (1 - max(fee, fee*c_sz)), 'amount': c_sz},
                put  ['instrument_name']: {'entry_price': p_px * (1 - max(fee, fee*p_sz)), 'amount': p_sz}
            }
            self.weekly_initial_equity = self.peak_equity = equity
            await self.save_state()
            logger.info("Executed straddle and reset state.")

    async def _monitor_task(self):
        while not self._shutdown_event.is_set():
            eq = await self.get_account_summary(self.symbol)
            if eq is None:
                await asyncio.sleep(60)
                continue
            # init
            if self.weekly_initial_equity is None:
                self.weekly_initial_equity = eq
                self.peak_equity = eq
                await self.save_state()
            # update peak
            if eq > self.peak_equity:
                self.peak_equity = eq
                await self.save_state()
            ret = (eq - self.weekly_initial_equity) / self.weekly_initial_equity
            dd  = (self.peak_equity - eq) / self.peak_equity
            if ret >= 0.01:
                logger.info("1% weekly return reached")
                await self.close_positions(eq, 1.0)
            elif ret >= 0.008:
                logger.info("0.8% weekly return reached")
                await self.close_positions(eq, 0.5)
            if dd >= 0.01:
                logger.warning("1% drawdown reached")
                await self.close_positions(eq, 1.0)
            await asyncio.sleep(60)

    async def run(self):
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, self._shutdown_event.set)
        loop.add_signal_handler(signal.SIGTERM, self._shutdown_event.set)

        await self.connect()
        await self.load_state()
        self._tasks = [asyncio.create_task(self._monitor_task()), asyncio.create_task(self._weekly_task())]
        done, pending = await asyncio.wait(self._tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await self.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deribit Straddle Strategy")
    parser.add_argument("--symbol",   default="BTC", choices=["BTC","ETH"], help="Underlying symbol")
    parser.add_argument("--side",     default="sell", choices=["buy","sell"], help="Direction")
    parser.add_argument("--leverage", type=float, default=1.0, help="Leverage")
    args = parser.parse_args()

    client = StraddleClient(args.symbol, args.side, args.leverage)
    try:
        asyncio.run(client.run())
    except Exception as err:
        logger.error(f"Fatal error: {err}")
        sys.exit(1)
