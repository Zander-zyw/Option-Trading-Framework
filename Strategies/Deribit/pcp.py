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
        self.allow_maker = False    # 是否允许挂单

        # Locks and queues
        self._arbs_lock = asyncio.Lock()
        self._update_queue: asyncio.Queue[PCPPair] = asyncio.Queue()
        self._arb_available = asyncio.Event()  # 用于启动挂单任务

        # 当前套利机会存储
        self.active_arbs: Dict[str, Dict] = {}
        self.archived_arbs: Dict[str, Dict] = {}

        # 调度器：每天 UTC 08:30 触发
        self.scheduler = AsyncIOScheduler(timezone=timezone("UTC"))
        self.scheduler.add_job(self._daily_update, 'cron', hour=8, minute=30)
        self.scheduler.start()


    async def _handle_subscription(self, message):
        """
        处理三类推送：
        1) ticker.*
        2) user.orders.option.<sym>.raw
        3) user.portfolio.<sym>
        """
        try:
            channel = message['params']['channel']
            data    = message['params']['data']

            # ——— 1) TICKER 更新 ———
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
                # 1) 过滤掉已取消
                if data.get('order_state') != 'cancelled':
                    order_id   = data.get('order_id')
                    filled_amt = data.get('filled_amount', 0.0)

                    if filled_amt > 0:
                        # 2) 在锁内找到对应 rec，算好 diff，并更新状态，但不要 await
                        matched = None
                        async with self._arbs_lock:
                            for _, rec in self.active_arbs.items():
                                logger.info("finding pairs")
                                if rec.get('maker_order_id') == order_id:
                                    prev = rec.get('filled_amount', 0)
                                    diff = filled_amt - prev
                                    if diff > 0:
                                        rec['filled_amount'] = filled_amt
                                        matched = (rec, diff)
                                    break  # 找到就跳出

                        # 3) 锁已释放，真正下 taker 单
                        if matched:
                            rec, diff = matched
                            await self._place_taker(rec, diff)
                            logger.info(f"🔄 Placed taker for {order_id}, amount {diff}")

            # ——— 3) Portfolio 更新 ———
            elif channel.startswith(f"user.portfolio.{self.symbol.lower()}"):
                init = data['initial_margin']  / data['margin_balance']
                maint = data['maintenance_margin'] / data['margin_balance']
                # 当任一比率过高，就不允许下 Maker
                self.allow_maker = not (init > 0.8 or maint > 0.5)

                if maint > 0.5:
                    logger.info(f"🚨 Maintenance margin {maint:.2%} exceeds 50%, cancelling maker orders")
                    async with self._arbs_lock:
                        for _, rec in self.active_arbs.items():
                            mid = rec.get('maker_order_id')
                            if mid:
                                await self.cancel_order(mid)
                                rec['status'] = 'active'

        except Exception as e:
            logger.error(f"_handle_subscription error: {e}")


    async def _place_taker(self, rec: Dict, amount_diff: float):
        side = rec['maker_side']
        pair = rec['pair']

        if side == 'put':
            # 对冲买入 call
            await self.send_order('buy', 'limit', pair.call_instrument, pair.pair_info['call']['ask']['price'], amount_diff)
            # 对冲卖出 future
            await self.send_order('sell', 'limit', pair.future_instrument, pair.pair_info['fut']['bid']['price'], amount_diff * pair.pair_info['fut']['bid']['price'])
        else:
            # 对冲买入 put
            await self.send_order('buy', 'limit', pair.put_instrument, pair.pair_info['put']['ask']['price'], amount_diff)
            # 对冲买入 future
            await self.send_order('buy', 'limit', pair.future_instrument, pair.pair_info['fut']['ask']['price'], amount_diff * pair.pair_info['fut']['ask']['price'])


    def _update_pair_info(self, data: Dict, pair: PCPPair, kind: str) -> bool:
        info = pair.pair_info[kind]
        updated = False

        # Ask 价格更新
        new_ask = data.get('best_ask_price', info['ask']['price'])
        if new_ask != info['ask']['price']:
            info['ask']['price'] = new_ask
            updated = True

        # Bid 价格更新
        new_bid = data.get('best_bid_price', info['bid']['price'])
        if new_bid != info['bid']['price']:
            info['bid']['price'] = new_bid
            updated = True

        # Ask 数量更新
        raw_ask = data.get('best_ask_amount', info['ask']['amount'])
        if raw_ask != info['ask']['amount']:
            if kind == 'fut':
                info['ask']['amount'] = math.floor((raw_ask / info['ask']['price']) / self.base_amount) * self.base_amount
            else:
                info['ask']['amount'] = raw_ask
            updated = True

        # Bid 数量更新
        raw_bid = data.get('best_bid_amount', info['bid']['amount'])
        if raw_bid != info['bid']['amount']:
            if kind == 'fut':
                info['bid']['amount'] = math.floor((raw_bid / info['bid']['price']) / self.base_amount) * self.base_amount
            else:
                info['bid']['amount'] = raw_bid
            updated = True

        return updated


    async def _monitor_arbitrage_opportunities(self):
        """
        Continuously process price updates and detect/close arbitrage opportunities.
        """
        while not self.exit_event.is_set():
            try:
                pair = await self._update_queue.get()

                # 跳过数据不完整或已到期的情况
                required = (
                    pair.pair_info['put']['ask']['price'],
                    pair.pair_info['call']['ask']['price'],
                    pair.pair_info['fut']['bid']['price'],
                    pair.pair_info['fut']['ask']['price'],
                )
                if None in required or pair.days_to_expiry <= 0:
                    continue

                # 计算套利条件
                pa, ca = pair.pair_info['put']['ask']['price'], pair.pair_info['call']['ask']['price']
                fb, fa = pair.pair_info['fut']['bid']['price'], pair.pair_info['fut']['ask']['price']
                K, d = pair.strike, pair.days_to_expiry
                fee = 0.00075
                min_p = min(0.0003, pa * 0.125)
                min_c = min(0.0003, ca * 0.125)
                cond1 = (pa - fee - min_p - min_c - ca - K / fb + 1) / d
                cond2 = (ca - fee - min_p - min_c - pa - 1 + K / fa) / d

                # 根据较小一侧深度与操作方向
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

                # 可能需要等待取消的 maker_order_id
                maker_to_cancel = None
                now = datetime.now()

                async with self._arbs_lock:
                    rec = self.active_arbs.get(key)

                    # 新机会或更新机会
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

                    # 否则关闭并归档
                    else:
                        if rec:
                            # 先记下需要取消的 maker_order_id
                            if rec.get('status') == 'maker_open' and rec.get('maker_order_id'):
                                maker_to_cancel = rec['maker_order_id']
                                rec['status'] = 'closed'

                            # 归档并移除
                            rec['end_time'] = now
                            self.archived_arbs[key] = rec
                            logger.info(f"❌ Closed arbitrage opportunity: {key}")
                            self.active_arbs.pop(key, None)

                # 锁已经释放，在这里同步等待取消完成
                if maker_to_cancel:
                    await self.cancel_order(maker_to_cancel)

            except Exception as e:
                logger.error(f"Error in _monitor_arbitrage_opportunities: {e}")


    async def place_maker_orders(self):
        """
        Listen for active_arbs and place/update maker orders accordingly.
        """
        while not self.exit_event.is_set():
            try:
                # 等待套利机会触发
                await self._arb_available.wait()
                self._arb_available.clear()
                logger.info("🚀 Placing maker orders batch")

                # 拷贝当前 active_arbs 快照
                async with self._arbs_lock:
                    items = list(self.active_arbs.items())

                for key, rec in items:
                    pair = rec['pair']
                    side = rec['maker_side']
                    price = pair.pair_info[side]['ask']['price']
                    depth = rec['depth']
                    amount = min(depth, 1)

                    # 如果量不足，跳过
                    if amount < self.base_amount:
                        continue

                    # 在锁内读取状态和现有 order_id
                    async with self._arbs_lock:
                        status = rec.get('status')
                        existing_oid = rec.get('maker_order_id')

                    # 新挂单：仅当状态为 active 且无未完成订单时
                    if status == 'active' and not existing_oid:
                        if not self.allow_maker:
                            continue

                        # 预先标记为正在挂单
                        async with self._arbs_lock:
                            rec['status'] = 'placing_maker'

                        instr = pair.put_instrument if side == 'put' else pair.call_instrument
                        # 发送挂单请求并等待响应
                        order_id = await self.send_order('sell', 'limit', instr, price, amount)

                        # 在获得响应后更新本地状态
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
                                # 挂单失败，重置为 active
                                rec['status'] = 'active'
                                logger.error(f"Failed to place maker for {key} @ {price}")

                        # 如果在下单期间机会已消失，则主动取消该单
                        async with self._arbs_lock:
                            still_active = key in self.active_arbs and self.active_arbs[key].get('status') == 'maker_open'
                        if not still_active and order_id:
                            await self.cancel_order(order_id)
                            logger.info(f"→ Cancelled orphan maker {order_id} for expired opportunity {key}")

                    # 修改/替换已有挂单：当价格或数量变化时
                    elif status == 'maker_open':
                        old_price = rec.get('maker_price')
                        old_amount = rec.get('maker_amount')
                        maker_id = rec.get('maker_order_id')

                        if price != old_price or amount != old_amount:
                            # 清除本地挂单状态
                            async with self._arbs_lock:
                                rec['maker_order_id'] = None
                                rec['status'] = 'active'

                            # 撤单
                            if maker_id:
                                await self.cancel_order(maker_id)
                                logger.info(f"→ Cancelled old maker {maker_id} for {key}")

                            # 通知下一轮重挂
                            self._arb_available.set()

            except Exception as e:
                logger.error(f"Error in place_maker_orders: {e}")
        

    async def _daily_update(self):
        now = datetime.now(timezone("UTC"))
        logger.info(f"[DailyUpdate] {now:%Y-%m-%d %H:%M:%S UTC}")

        try:
            # 1) Fetch new PCP pairs
            new_pairs = await self.get_pcp_pairs(self.symbol)

            new_keys = {
                f"{p.put_instrument}|{p.call_instrument}|{p.future_instrument}"
                for p in new_pairs
            }

            # 2) Archive expired arbitrages
            expired = []
            async with self._arbs_lock:
                for k, rec in list(self.active_arbs.items()):
                    if k not in new_keys or rec['pair'].days_to_expiry <= 0:
                        expired.append((k, rec))
                        self.active_arbs.pop(k, None)

            for k, rec in expired:
                oid = rec.get('maker_order_id')
                
                if rec.get('status') == 'maker_open' and oid:
                    await self.cancel_order(oid)

                rec['end_time'] = now
                self.archived_arbs[k] = rec
                logger.info(f"❌ Archived expired arb {k}")

            # 3) Update the pcp_pairs list to the new set
            self.pcp_pairs = new_pairs

            # 4) Build new subscription set
            new_subs: Set[str] = {f"user.orders.option.{self.symbol.lower()}.raw", f"user.portfolio.{self.symbol.lower()}"}
            for pair in self.pcp_pairs:
                new_subs.add(f"ticker.{pair.put_instrument}.raw")
                new_subs.add(f"ticker.{pair.call_instrument}.raw")
                new_subs.add(f"ticker.{pair.future_instrument}.raw")

            # 5) Unsubscribe and subscribe diffs
            to_unsub = list(self.subscribed_instruments - new_subs)
            to_sub = list(new_subs - self.subscribed_instruments)
            if to_unsub:
                await self.unsubscribe(to_unsub)
                logger.info(f"→ Unsubscribed {to_unsub}")
            if to_sub:
                await self.subscribe(to_sub)
                logger.info(f"→ Subscribed {to_sub}")

            # 6) Save new subscriptions
            self.subscribed_instruments = new_subs

            # 7) Clear any pending updates in queue
            while not self._update_queue.empty():
                self._update_queue.get_nowait()

        except Exception as e:
            logger.error(f"DailyUpdate error: {e}")


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
