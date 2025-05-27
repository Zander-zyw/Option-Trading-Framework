### import DeribitClient ###
import sys
import os
import signal

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
sys.path.append(BASE_DIR)

from Deribit.DeribitClient import DeribitClient
from Logger.Logger import logger
### import DeribitClient ###

import asyncio
import math
from typing import Dict, Set, List
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pytz import timezone


class PCPPair:
    def __init__(
        self,
        strike: float,
        expiry: str,
        put_instrument: str,
        call_instrument: str,
        future_instrument: str,
    ):
        self.strike = strike
        self.expiry = expiry                # 格式：'30MAY25'
        self.put_instrument = put_instrument
        self.call_instrument = call_instrument
        self.future_instrument = future_instrument

        # put/call/fut 盘口数据
        self.pair_info: Dict[str, Dict] = {
            "put": {"ask": {"price": None, "amount": None}, "bid": {"price": None, "amount": None}},
            "call": {"ask": {"price": None, "amount": None}, "bid": {"price": None, "amount": None}},
            "fut": {"ask": {"price": None, "amount": None}, "bid": {"price": None, "amount": None}},
        }

    @property
    def days_to_expiry(self):
        expiry_ts = datetime.strptime(self.expiry, "%d%b%y").timestamp() * 1000
        now_ts = datetime.now().timestamp() * 1000
        return (expiry_ts - now_ts) / 86400000

    def __repr__(self):
        return f"PCPPair(strike={self.strike}, expiry={self.expiry})"

    def __del__(self):
        logger.info(f"🗑️ Deleting PCPPair: {self.put_instrument}|{self.call_instrument}|{self.future_instrument}")


class PCPArbitrage(DeribitClient):
    def __init__(self, symbol: str):
        super().__init__()
        self.symbol = symbol
        self.exit_event = asyncio.Event()

        # 根据symbol设置基础挂单量
        if self.symbol == 'ETH':
            self.base_amount = 1
        elif self.symbol == 'BTC':
            self.base_amount = 0.1
        else:
            raise ValueError(f"Unsupported symbol: {self.symbol}")

        # PCP Variables
        self.pcp_pairs: List[PCPPair] = []
        self.subscribed_instruments: Set[str] = set()

        # 保证金相关变量
        self.allow_maker = False

        # Locks and queues
        self._arbs_lock = asyncio.Lock()
        self._update_queue: asyncio.Queue[PCPPair] = asyncio.Queue()
        self._arb_available = asyncio.Event()

        # 当前套利机会存储
        self.active_arbs: Dict[str, Dict] = {}
        self.archived_arbs: Dict[str, Dict] = {}

        # Pause/resume flag for settlement window
        self.trading_paused = asyncio.Event()
        self.trading_paused.set()

        # Scheduler: pause at 07:50 UTC, update+resume at 08:20 UTC
        self.scheduler = AsyncIOScheduler(timezone=timezone("UTC"))
        self.scheduler.add_job(self._pause_trading, 'cron', hour=7, minute=50)
        self.scheduler.add_job(self._daily_update, 'cron', hour=8, minute=20)
        self.scheduler.start()

    async def _pause_trading(self):
        logger.info("⏸️ Pausing trading for settlement (07:50–08:20 UTC)")

        # Cancel all open maker orders to avoid unintended fills
        async with self._arbs_lock:
            for _, rec in list(self.active_arbs.items()):
                oid = rec.get('maker_order_id')
                if rec.get('status') == 'maker_open' and oid:
                    rec['status'] = 'active'
                    rec['maker_order_id'] = None

                    asyncio.create_task(self.cancel_order(oid))

                    logger.info(f"→ Cancelled maker order {oid} due to pause")

        self.trading_paused.clear()

    async def _handle_subscription(self, message):
        try:
            channel = message['params']['channel']
            data = message['params']['data']

            if channel.startswith("ticker."):
                instrument = channel.split('.')[1]
                for pair in self.pcp_pairs:
                    updated = False
                    if instrument == pair.put_instrument:
                        updated = self._update_pair_info(data, pair, "put")
                    elif instrument == pair.call_instrument:
                        updated = self._update_pair_info(data, pair, "call")
                    elif instrument == pair.future_instrument:
                        updated = self._update_pair_info(data, pair, "fut")

                    if updated:
                        await self._update_queue.put(pair)

            elif channel.startswith(f"user.orders.option."):
                if data.get('order_state') != 'cancelled':
                    order_id = data.get('order_id')
                    filled_amt = data.get('filled_amount', 0.0)

                    if filled_amt > 0:
                        matched = None
                        async with self._arbs_lock:
                            for _, rec in self.active_arbs.items():
                                if rec.get('maker_order_id') == order_id:
                                    prev = rec.get('filled_amount', 0)
                                    diff = filled_amt - prev
                                    if diff > 0:
                                        rec['filled_amount'] = filled_amt
                                        matched = (rec, diff)
                                    break

                        if matched:
                            rec, diff = matched
                            asyncio.create_task(self._place_taker(rec, diff))
                            logger.info(f"🔄 Scheduled taker for order {rec['maker_order_id']}, amount {diff}")

            elif channel.startswith(f"user.portfolio.{self.symbol.lower()}"):
                init = data['initial_margin'] / data['margin_balance']
                maint = data['maintenance_margin'] / data['margin_balance']
                self.allow_maker = not (init > 0.8 or maint > 0.5)

                if maint > 0.5:
                    logger.info(f"🚨 Maintenance margin {maint:.2%} exceeds 50%, cancelling maker orders")
                    async with self._arbs_lock:
                        for _, rec in self.active_arbs.items():
                            oid = rec.get('maker_order_id')
                            if oid:
                                rec['status'] = 'active'
                                asyncio.create_task(self.cancel_order(oid))
        except Exception as e:
            logger.error(f"_handle_subscription error: {e}")

    async def _place_taker(self, rec: Dict, amount_diff: float):
        side = rec['maker_side']
        pair = rec['pair']
        taker_ids = []

        if side == 'put':
            oid1 = await self.send_order('buy', 'limit', pair.call_instrument, pair.pair_info['call']['ask']['price'], amount_diff)
            taker_ids.append(oid1)
            oid2 = await self.send_order('sell', 'limit', pair.future_instrument, pair.pair_info['fut']['bid']['price'], amount_diff * pair.pair_info['fut']['bid']['price'])
            taker_ids.append(oid2)
        else:
            oid1 = await self.send_order('buy', 'limit', pair.put_instrument, pair.pair_info['put']['ask']['price'], amount_diff)
            taker_ids.append(oid1)
            oid2 = await self.send_order('buy', 'limit', pair.future_instrument, pair.pair_info['fut']['ask']['price'], amount_diff * pair.pair_info['fut']['ask']['price'])
            taker_ids.append(oid2)

        now = datetime.now()
        key = f"{pair.put_instrument}|{pair.call_instrument}|{pair.future_instrument}"
        async with self._arbs_lock:
            rec['taker_order_ids'] = taker_ids
            rec['status'] = 'taker_executed'
            rec['end_time'] = now
            self.active_arbs.pop(key, None)
            self.archived_arbs[key] = rec

        logger.info(f"🏁 Completed taker for {key}: orders={taker_ids}")

    def _update_pair_info(self, data: Dict, pair: PCPPair, kind: str) -> bool:
        info = pair.pair_info[kind]
        updated = False
        new_ask = data.get('best_ask_price', info['ask']['price'])
        if new_ask != info['ask']['price']:
            info['ask']['price'] = new_ask
            updated = True
        new_bid = data.get('best_bid_price', info['bid']['price'])
        if new_bid != info['bid']['price']:
            info['bid']['price'] = new_bid
            updated = True
        raw_ask = data.get('best_ask_amount', info['ask']['amount'])
        if raw_ask != info['ask']['amount']:
            if kind == 'fut':
                info['ask']['amount'] = math.floor((raw_ask / info['ask']['price']) / self.base_amount) * self.base_amount
            else:
                info['ask']['amount'] = raw_ask
            updated = True
        raw_bid = data.get('best_bid_amount', info['bid']['amount'])
        if raw_bid != info['bid']['amount']:
            if kind == 'fut':
                info['bid']['amount'] = math.floor((raw_bid / info['bid']['price']) / self.base_amount) * self.base_amount
            else:
                info['bid']['amount'] = raw_bid
            updated = True
        return updated

    async def _monitor_arbitrage_opportunities(self):
        while not self.exit_event.is_set():
            await self.trading_paused.wait()
            try:
                pair = await self._update_queue.get()
                required = (
                    pair.pair_info['put']['ask']['price'],
                    pair.pair_info['call']['ask']['price'],
                    pair.pair_info['fut']['bid']['price'],
                    pair.pair_info['fut']['ask']['price'],
                )
                if None in required or pair.days_to_expiry <= 0:
                    continue
                pa, ca = pair.pair_info['put']['ask']['price'], pair.pair_info['call']['ask']['price']
                fb, fa = pair.pair_info['fut']['bid']['price'], pair.pair_info['fut']['ask']['price']
                K, d = pair.strike, pair.days_to_expiry
                fee = 0.00075
                min_p = min(0.0003, pa * 0.125)
                min_c = min(0.0003, ca * 0.125)
                cond1 = (pa - fee - min_p - min_c - ca - K / fb + 1) / d
                cond2 = (ca - fee - min_p - min_c - pa - 1 + K / fa) / d
                if cond1 > cond2:
                    depth = min(pair.pair_info['call']['ask']['amount'] or 0,
                                pair.pair_info['fut']['bid']['amount'] or 0)
                    maker_side = 'put'
                else:
                    depth = min(pair.pair_info['put']['ask']['amount'] or 0,
                                pair.pair_info['fut']['ask']['amount'] or 0)
                    maker_side = 'call'
                key = f"{pair.put_instrument}|{pair.call_instrument}|{pair.future_instrument}"
                threshold = 0.0002
                maker_to_cancel = None
                now = datetime.now()
                async with self._arbs_lock:
                    rec = self.active_arbs.get(key)
                    if cond1 > threshold or cond2 > threshold:
                        record = {
                            'pair': pair,
                            'start_time': now,
                            'cond1': cond1,
                            'cond2': cond2,
                            'depth': depth,
                            'maker_side': maker_side,
                            'status': rec['status'] if rec else 'active',
                            'filled_amount': rec['filled_amount'] if rec else 0,
                            'maker_order_id': rec['maker_order_id'] if rec else None,
                        }
                        if rec is None:
                            self.active_arbs[key] = record
                            logger.info(f"✨ New arbitrage opportunity: {key}")
                            if not self._arb_available.is_set():
                                self._arb_available.set()
                        else:
                            self.active_arbs[key].update(record)
                    else:
                        if rec:
                            if rec.get('status') == 'maker_open' and rec.get('maker_order_id'):
                                maker_to_cancel = rec['maker_order_id']
                                rec['status'] = 'closed'
                            rec['end_time'] = now
                            self.archived_arbs[key] = rec
                            logger.info(f"❌ Closed arbitrage opportunity: {key}")
                            self.active_arbs.pop(key, None)
                if maker_to_cancel:
                    await self.cancel_order(maker_to_cancel)
            except Exception as e:
                logger.error(f"Error in _monitor_arbitrage_opportunities: {e}")

    async def place_maker_orders(self):
        while not self.exit_event.is_set():
            await self.trading_paused.wait()
            try:
                await self._arb_available.wait()
                self._arb_available.clear()
                if not self.trading_paused.is_set():
                    continue
                logger.info("🚀 Placing maker orders batch")
                async with self._arbs_lock:
                    items = list(self.active_arbs.items())
                for key, rec in items:
                    pair = rec['pair']
                    side = rec['maker_side']
                    price = pair.pair_info[side]['ask']['price']
                    depth = rec['depth']
                    amount = min(depth, 1)

                    # 如果挂单量小于基础挂单量，则跳过
                    if amount < self.base_amount:
                        continue

                    async with self._arbs_lock:
                        status = rec.get('status')
                        existing_oid = rec.get('maker_order_id')

                    # 如果订单状态为 active 且没有 maker 订单，则下 maker 单
                    if status == 'active' and not existing_oid:
                        if not self.allow_maker:
                            continue
                        
                        async with self._arbs_lock:
                            rec['status'] = 'placing_maker'

                        instr = pair.put_instrument if side == 'put' else pair.call_instrument
                        order_id = await self.send_order('sell', 'limit', instr, price, amount)
                        async with self._arbs_lock:
                            if order_id:
                                rec.update({
                                    'maker_order_id': order_id,
                                    'maker_price': price,
                                    'maker_amount': amount,
                                    'status': 'maker_open'
                                })
                                logger.info(f"→ Placed maker {key} @ {price}")
                            else:
                                rec['status'] = 'active'
                                logger.error(f"Failed to place maker for {key} @ {price}")
                        async with self._arbs_lock:
                            still_active = key in self.active_arbs and self.active_arbs[key].get('status') == 'maker_open'
                        if not still_active and order_id:
                            await self.cancel_order(order_id)
                            logger.info(f"→ Cancelled orphan maker {order_id} for expired opportunity {key}")

                    # 如果 maker 订单还在，则检查价格和数量是否变化
                    elif status == 'maker_open':
                        old_price = rec.get('maker_price')
                        old_amount = rec.get('maker_amount')
                        maker_id = rec.get('maker_order_id')
                        if price != old_price or amount != old_amount:
                            async with self._arbs_lock:
                                rec['maker_order_id'] = None
                                rec['status'] = 'active'
                            if maker_id:
                                await self.cancel_order(maker_id)
                                logger.info(f"→ Cancelled old maker {maker_id} for {key}")
                            self._arb_available.set()

            except Exception as e:
                logger.error(f"Error in place_maker_orders: {e}")

    async def _daily_update(self):
        now = datetime.now(timezone("UTC"))
        logger.info(f"[DailyUpdate] {now:%Y-%m-%d %H:%M:%S UTC}")
        try:
            raw_new = await self.get_pcp_pairs(self.symbol)

            # map old by key
            old_map = {f"{p.put_instrument}|{p.call_instrument}|{p.future_instrument}": p for p in self.pcp_pairs}
            updated = []
            new_keys = set()
            for candidate in raw_new:
                key = f"{candidate.put_instrument}|{candidate.call_instrument}|{candidate.future_instrument}"
                new_keys.add(key)
                if key in old_map:
                    existing = old_map.pop(key)
                    existing.strike = candidate.strike
                    existing.expiry = candidate.expiry
                    updated.append(existing)
                else:
                    updated.append(candidate)

            # expire old pairs
            for exp_key, rec in list(self.active_arbs.items()):
                if exp_key not in new_keys or rec['pair'].days_to_expiry <=0:
                    oid = rec.get('maker_order_id')

                    if rec.get('status')=='maker_open' and oid:
                        await self.cancel_order(oid)

                    rec['end_time'] = now
                    self.archived_arbs[exp_key] = rec
                    logger.info(f"❌ Archived expired arb {exp_key}")
                    self.active_arbs.pop(exp_key, None)

            self.pcp_pairs = updated

            # subscription diff
            subs = {f"user.orders.option.{self.symbol.lower()}.raw", f"user.portfolio.{self.symbol.lower()}"}
            for p in self.pcp_pairs:
                subs.update({f"ticker.{p.put_instrument}.raw", f"ticker.{p.call_instrument}.raw", f"ticker.{p.future_instrument}.raw"})

            to_unsub = list(self.subscribed_instruments - subs)
            to_sub   = list(subs - self.subscribed_instruments)

            if to_unsub:
                await self.unsubscribe(to_unsub)
                logger.info(f"→ Unsubscribed {to_unsub}")
            if to_sub:
                await self.subscribe(to_sub)
                logger.info(f"→ Subscribed   {to_sub}")

            self.subscribed_instruments = subs

            while not self._update_queue.empty():
                self._update_queue.get_nowait()

        except Exception as e:
            logger.error(f"DailyUpdate error: {e}")
        finally:
            self.trading_paused.set()
            logger.info("▶️ Trading resumed after daily update")


    async def get_options(self, currency: str) -> Dict[str, Dict]:
        response = await super().get_instruments(currency, 'option')
        if not response:
            raise Exception("Failed to fetch option instruments")

        filtered = {}
        for ins in response['result']:
            expiry_ms = float(ins['expiration_timestamp'])
            remain_days = (
                expiry_ms - datetime.now().timestamp() * 1000
            ) / 86400000
            if remain_days < 15:
                filtered[ins['instrument_name']] = {
                    'days_to_expiry': remain_days,
                    'option_type': ins['option_type'],
                }
        return filtered


    async def get_perpetuals(self, currency: str) -> Dict[str, Dict]:
        response = await super().get_instruments(currency, 'future')
        if not response:
            raise Exception("Failed to fetch futures instruments")

        filtered = {}
        for ins in response['result']:
            expiry_ms = float(ins['expiration_timestamp'])
            remain_days = (
                expiry_ms - datetime.now().timestamp() * 1000
            ) / 86400000
            name = ins['instrument_name']
            if remain_days < 15 or name.endswith('PERPETUAL'):
                filtered[name] = {
                    'days_to_expiry': remain_days
                }
        return filtered


    async def get_pcp_pairs(self, currency: str) -> List[PCPPair]:
        options = await self.get_options(currency)
        perps = await self.get_perpetuals(currency)

        grouped: Dict[str, Dict[float, Dict[str, str]]] = {}
        for name, info in options.items():
            parts = name.split('-')
            expiry = parts[1]
            strike = float(parts[2])
            otype = info['option_type']

            grouped.setdefault(expiry, {})
            grouped[expiry].setdefault(strike, {})
            grouped[expiry][strike][otype] = name

        pairs: List[PCPPair] = []
        for expiry, strikes in grouped.items():
            for strike, sides in strikes.items():
                if 'put' not in sides or 'call' not in sides:
                    continue

                put_name = sides['put']
                call_name = sides['call']
                future_name = None

                # 寻找匹配的 futures 或 perpetual
                for fname in perps:
                    if fname.endswith(expiry):
                        future_name = fname
                        break
                if not future_name:
                    for fname in perps:
                        if fname.endswith('PERPETUAL'):
                            future_name = fname
                            break
                if not future_name:
                    continue

                ticker_info = await self.ticker(future_name)
                mark_price = ticker_info['result']['mark_price']

                if not (mark_price * 0.95 <= strike <= mark_price * 1.05):
                    continue

                pairs.append(
                    PCPPair(
                        strike,
                        expiry,
                        put_name,
                        call_name,
                        future_name,
                    )
                )

        return pairs


    async def send_order(self, side: str, order_type: str, instrument_name: str, price: float, amount: float, time_in_force: str = 'good_til_cancelled') -> str:
        response = await super().send_order(side, order_type, instrument_name, price, amount, time_in_force)

        if "error" in response:
            logger.error(f"Send order error: {response['error']}")
            return None
        
        return response['result']['order']['order_id']
    

    async def cancel_order(self, order_id: str) -> str:
        response = await super().cancel_order(order_id)

        if "error" in response:
            logger.error(f"Cancel order error: {response['error']}")
            return None
        
        order_id = response['result']['order_id']
        logger.info(f"→ Cancelled maker order {order_id}")
        return order_id


async def main():
    pcp = PCPArbitrage('ETH')
    await pcp.connect()
    await pcp._daily_update()

    # 启动协程
    asyncio.create_task(pcp._monitor_arbitrage_opportunities())
    asyncio.create_task(pcp.place_maker_orders())

    # 使用 pcp.exit_event 控制退出
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGINT, pcp.exit_event.set)
    loop.add_signal_handler(signal.SIGTERM, pcp.exit_event.set)

    logger.info("⏳ Press Ctrl+C to exit")
    await pcp.exit_event.wait()

    logger.info("🔌 Disconnecting...")
    await pcp.disconnect()

if __name__ == '__main__':
    asyncio.run(main())
