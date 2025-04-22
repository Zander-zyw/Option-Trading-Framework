### import OKXClient ###
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from OKX.OKXClient import OKXClient
from Logger.Logger import logger
### import OKXClient ###

import json
import signal
import argparse
import threading
import time
import math

from datetime import datetime, timedelta, timezone

class StraddleClient(OKXClient):
    def __init__(self, symbol: str, side: str, leverage: float):
        super().__init__()
        self.symbol = symbol.upper()
        self.side = side.lower()
        self.leverage = leverage
        self.base_amount = {'BTC': 0.01, 'ETH': 0.1}.get(self.symbol)
        if self.base_amount is None:
            raise ValueError(f"Unsupported symbol: {self.symbol}")

        # strategy state
        self.active_positions = {}
        self.weekly_initial_equity = None
        self.peak_equity = None
        self.is_running = True

        # set up signal handlers and load previous state
        self._setup_signals()
        self.load_state()

    def _setup_signals(self):
        if sys.platform != 'win32':
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info(f"Signal {signum} received: shutting down...")
        self.is_running = False

    # ----- State persistence -----
    def save_state(self):
        os.makedirs(BASE_DIR, exist_ok=True)
        path = os.path.join(BASE_DIR, f'okx_straddle_{self.symbol}.json')
        data = {
            'active_positions': self.active_positions,
            'weekly_initial_equity': self.weekly_initial_equity,
            'peak_equity': self.peak_equity,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"State saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def load_state(self):
        path = os.path.join(BASE_DIR, f'okx_straddle_{self.symbol}.json')
        if not os.path.exists(path):
            logger.info("No saved state, starting fresh")
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self.active_positions = data.get('active_positions', {})
            self.weekly_initial_equity = data.get('weekly_initial_equity')
            self.peak_equity = data.get('peak_equity')
            logger.info(f"Loaded state from {path}")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

    # ----- Helper methods -----
    def _next_friday_time(self) -> float:
        now = datetime.now(timezone.utc)
        days = (4 - now.weekday()) % 7 or 7
        next_friday = now + timedelta(days=days)
        exec_time = next_friday.replace(hour=8, minute=30, second=0, microsecond=0)
        return exec_time.timestamp()

    def _wait_until_friday(self):
        now_ts = datetime.now(timezone.utc).timestamp()
        target_ts = self._next_friday_time()
        delay = max(0, target_ts - now_ts)
        logger.info(f"Sleeping {int(delay)}s until Friday 08:30 UTC")
        time.sleep(delay)

    def get_mark_price(self, inst_id: str) -> float:
        resp = super().ticker(inst_id)
        if resp and resp.get('data'):
            return resp['data'][0].get('last')
        logger.error(f"Failed to fetch mark price for {inst_id}: {resp}")
        return None

    def get_option_chain(self):
        resp = super().get_instruments('OPTION', uly=f"{self.symbol}-USD")
        if resp and resp.get('data'):
            calls = [o for o in resp['data'] if o.get('optType') == 'C']
            puts  = [o for o in resp['data'] if o.get('optType') == 'P']
            return calls, puts
        logger.error(f"Failed to fetch option chain: {resp}")
        return [], []

    def get_book(self, inst_id: str, size: int = 1):
        resp = super().orderbook(inst_id, size)
        if resp and resp.get('data'):
            asks = resp['data'][0].get('asks', [])
            bids = resp['data'][0].get('bids', [])
            return (asks[0][0] if asks else None), (bids[0][0] if bids else None)
        logger.error(f"Orderbook fetch failed: {resp}")
        return None, None

    def place_order(self, inst_id, td_mode, side, ord_type, sz, px=None):
        args = (inst_id, td_mode, side, ord_type, sz, px) if px else (inst_id, td_mode, side, ord_type, sz)
        resp = super().place_order(*args)
        if resp and resp.get('data'):
            return resp['data'][0]['ordId']
        logger.error(f"Order placement failed: {resp}")
        return None

    def get_order_state(self, inst_id, ord_id):
        resp = super().get_order_details(inst_id, ord_id)
        if resp and resp.get('data'):
            d = resp['data'][0]
            return d.get('sz',0) - d.get('fillSz',0)
        logger.error(f"Order state failed: {resp}")
        return None

    def wait_for_fill(self, call_id, put_id, call_inst, put_inst):
        while self.is_running:
            rem_call = self.get_order_state(call_inst, call_id)
            rem_put  = self.get_order_state(put_inst, put_id)
            if rem_call == 0 and rem_put == 0:
                logger.info("Orders filled")
                break
            logger.info(f"Waiting fill: call={rem_call}, put={rem_put}")
            time.sleep(60)

    # ----- Core strategy -----
    def _execute_once(self):
        equity = self.get_account_equity()
        if equity is None:
            return
        size = math.floor(0.5 * equity * self.leverage / self.base_amount) * self.base_amount

        # instrument naming as D-M-YY
        dt = datetime.now(timezone.utc)
        dt_next = self._next_friday_time()
        inst_date = datetime.fromtimestamp(dt_next, timezone.utc).strftime('%-d%b%y').upper()
        call_sz, put_sz = size, size
        inst_format = f"{self.symbol}-USDT-{inst_date}"

        mark = self.get_mark_price(inst_format)
        if mark is None:
            return

        calls, puts = self.get_option_chain()
        call_opt = next((o for o in calls if o['stk'] > mark), None)
        put_opt  = next((o for o in puts  if o['stk'] < mark), None)
        if not call_opt or not put_opt:
            return

        ask_c, _ = self.get_book(call_opt['instId'])
        ask_p, _ = self.get_book(put_opt ['instId'])
        if ask_c is None or ask_p is None:
            return

        call_id = self.place_order(call_opt['instId'], 'cash', self.side, 'limit', call_sz, ask_c)
        put_id  = self.place_order(put_opt ['instId'], 'cash', self.side, 'limit', put_sz, ask_p)
        if not call_id or not put_id:
            return

        self.wait_for_fill(call_id, put_id, call_opt['instId'], put_opt['instId'])

        # record
        cp, cs = self.get_transaction_details(call_opt['instId'], call_id)
        pp, ps = self.get_transaction_details(put_opt ['instId'], put_id)
        fee = 0.0003
        self.active_positions = {
            call_opt['instId']: {'entry_price': cp*(1-fee), 'amount': cs},
            put_opt ['instId']: {'entry_price': pp*(1-fee), 'amount': ps}
        }
        self.weekly_initial_equity = equity
        self.peak_equity = equity
        self.save_state()
        logger.info(f"Executed straddle: equity reset {equity}")

    def _monitor(self):
        while self.is_running:
            eq = self.get_account_equity()
            if eq is None:
                time.sleep(60)
                continue
            if self.weekly_initial_equity is None:
                self.weekly_initial_equity = eq
                self.peak_equity = eq
            if eq > self.peak_equity:
                self.peak_equity = eq
            ret = (eq - self.weekly_initial_equity) / self.weekly_initial_equity
            dd  = (self.peak_equity - eq) / self.peak_equity
            if ret >= 0.01:
                self.close_positions(eq, 1.0)
            elif ret >= 0.008:
                self.close_positions(eq, 0.5)
            if dd >= 0.01:
                self.close_positions(eq, 1.0)
            time.sleep(60)

    def close_positions(self, current_eq: float, ratio: float = 1.0):
        # reuse async version or sync wrapper
        pass  # implement similar to async but sync

    def execute(self):
        monitor = threading.Thread(target=self._monitor, daemon=True)
        monitor.start()
        try:
            while self.is_running:
                self._wait_until_friday()
                self._execute_once()
        except Exception as e:
            logger.error(f"Execution error: {e}")
        finally:
            self.save_state()
            self.is_running = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OKX Straddle Strategy')
    parser.add_argument('--symbol', default='BTC', choices=['BTC','ETH'])
    parser.add_argument('--side',   default='sell', choices=['buy','sell'])
    parser.add_argument('--leverage', type=float, default=1.0)
    args = parser.parse_args()

    client = StraddleClient(args.symbol, args.side, args.leverage)
    client.execute()
