from abc import ABC, abstractmethod
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any, Dict, List, Optional, Tuple, Deque, Type
import numpy as np
import json
import jsonpickle
import math
from collections import deque


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Strategy(ABC):
    """策略抽象基类"""

    def __init__(self, symbol: str, position_limit: int):
        self.symbol = symbol
        self.position_limit = position_limit
        self.trader_data = {}

    @abstractmethod
    def generate_orders(self, state: TradingState) -> List[Order]:
        """生成订单逻辑"""
        raise NotImplementedError

    def run(self, state: TradingState) -> Tuple[List[Order], dict]:
        """执行策略主逻辑"""

        # 状态预处理
        order_depth = state.order_depths.get(self.symbol, OrderDepth())
        if not order_depth.buy_orders and not order_depth.sell_orders:
            return [], {}

        # 生成订单
        self.orders = self.generate_orders(state)

        # 保存策略状态，用于下次加载（包括仓位、因子等历史信息）
        strategy_state = self.save_state(state)

        return self.orders, strategy_state

    def save_state(self, state) -> dict:
        """保存策略状态"""
        return {}

    def load_state(self, state: TradingState):
        """加载策略状态"""
        pass


class MacaronStrategy(Strategy):
    """马卡龙交易策略"""

    def __init__(self,
                 symbol: str,
                 position_limit: int,
                 conversion_limit: int = 10,
                 sun_threshold: int = 60,
                 sun_coeff: float = 0.10,
                 sugar_threshold: int = 40,
                 sugar_coeff: float = 0.08):
        super().__init__(symbol, position_limit)
        self.conversion_limit = conversion_limit
        self.storage_cost = 0.1
        self.current_mode = "Normal"

        self.sun_threshold = sun_threshold
        self.sun_coeff = sun_coeff
        self.sugar_threshold = sugar_threshold
        self.sugar_coeff = sugar_coeff

        # 历史数据
        self.fair_value_history = deque(maxlen=100)
        self.sugar_price_history = deque(maxlen=100)
        self.sun_price_history = deque(maxlen=100)

        self.observation_history = deque(maxlen=100)
        self.position_history = deque(maxlen=100)
        self.position_entries = {}

        self.sunlightIndex_slope = 0
        self.sunlightIndex_history = deque(maxlen = 100)
        
        self.storage_cost = 0
        #计算持仓成本数据
        self.orders_history = {
            'long': [], 
            'short': []
        }

        #记录其他交易者的
        self.traders_info = {}

        self.target_amount = 0
        self.target_conversions = 0

        self.last_pos = 0 #用来反推conversion成交量

    def get_available_amount(self, symbol: str, state: TradingState) -> int:
        """
        返回市场上已有市价单的总数量
        sell_amount, buy_amount（注意都为正数）
        """
        order_depth = state.order_depths[symbol]
        sell_amount = -sum(order_depth.sell_orders.values())
        buy_amount = sum(order_depth.buy_orders.values())
        return sell_amount, buy_amount
    
    def quick_trade(self, symbol: str, state: TradingState, amount: int) -> Tuple[list, int]:
        """
        快速交易函数：
        给定商品名和所需数量(amount)，正数代表买入，负数代表卖出
        返回尽可能的最佳orders和剩余数量
        """
        orders = []
        order_depth = state.order_depths[symbol]
        position = state.position.get(symbol, 0)
        amount = int(amount)
        if amount > 0:
            for price, sell_amount in sorted(order_depth.sell_orders.items()):
                max_amount = min(-sell_amount, self.position_limit - position, amount)
                if max_amount > 0:
                    orders.append(Order(symbol, price, max_amount))
                    position += max_amount
                    amount -= max_amount
                if amount == 0:
                    break

        elif amount < 0:
            for price, buy_amount in sorted(order_depth.buy_orders.items()):
                max_amount = min(buy_amount, position + self.position_limit, -amount)  # amount已经是负数，卖出
                if max_amount > 0:
                    # 卖出
                    orders.append(Order(symbol, price, -max_amount))
                    position -= max_amount
                    amount += max_amount

                if amount == 0:
                    break

        return orders, amount
    
    def calculate_avg_cost(self, state: TradingState):
        """
        实时从 own_trades 中读取成交信息，计算平均持仓成本与当前净持仓
        自动处理多空平仓冲销逻辑
        """
        if not hasattr(self, "processed_trades"):
            self.processed_trades = set()
            self.orders_history = {
                'long': [],
                'short': []
            }
        def offset_inventory(order_list, amount):
            while amount > 0 and order_list:
                price, qty = order_list[0]
                if qty <= amount:
                    amount -= qty
                    order_list.pop(0)
                else:
                    order_list[0] = (price, qty - amount)
                    amount = 0
            return amount
        
        own_trades = state.own_trades.get(self.symbol, [])
        for trade in own_trades:
            trade_id = (trade.seller, trade.buyer, trade.price, trade.quantity, trade.timestamp)
            if trade_id in self.processed_trades:
                continue
            self.processed_trades.add(trade_id)
            

            # 判断你是买还是卖
            if trade.buyer == "SUBMISSION":
                direction = 1
            elif trade.seller == "SUBMISSION":
                direction = -1
            else:
                continue  # 不属于你自己的成交，跳过

            price = trade.price
            qty = trade.quantity

            # 平仓处理逻辑（offset 对手方向）
            if direction == 1:
                # 如果有空头仓位，优先冲销
                qty = offset_inventory(self.orders_history['short'], qty)
                if qty > 0:
                    self.orders_history['long'].append((price, qty))
            else:
                # 有多头仓位时优先冲销
                qty = offset_inventory(self.orders_history['long'], qty)
                if qty > 0:
                    self.orders_history['short'].append((price, qty))

        # 计算净仓与平均成本
        long_qty = sum(q for _, q in self.orders_history['long'])
        short_qty = sum(q for _, q in self.orders_history['short'])
        net_pos = long_qty - short_qty

        if net_pos > 0:
            total_cost = sum(price * qty for price, qty in self.orders_history['long'])
            avg_cost = total_cost / long_qty if long_qty else 0.0
        elif net_pos < 0:
            total_cost = sum(price * qty for price, qty in self.orders_history['short'])
            avg_cost = total_cost / short_qty if short_qty else 0.0
        else:
            avg_cost = 0.0


        #需要补上conversion成本
        return avg_cost, net_pos

    def update_storage_cost(self, state: TradingState):
        # 计算总仓储成本
        position = state.position.get(self.symbol, 0)
        if position > 0:
            self.storage_cost += 0.1 * state.position.get(self.symbol, 0)
        return

    def update_conversion_cost(self, state: TradingState):
        """
        根据仓位变化和 market orders（own_trades）反推 conversion 实际成交量，并记录到 orders_history。
        加入去重与时间戳校验，防止重复处理。
        """
        if not hasattr(self, "last_pos"):
            self.last_pos = state.position.get(self.symbol, 0)
        if not hasattr(self, "processed_trades"):
            self.processed_trades = set()

        current_pos = state.position.get(self.symbol, 0)
        delta_pos = current_pos - self.last_pos

        # 处理 own_trades 中属于自己的订单，统计 market_delta
        market_delta = 0
        own_trades = state.own_trades.get(self.symbol, [])
        for trade in own_trades:
            # 构造 trade ID 进行去重
            trade_id = (trade.seller, trade.buyer, trade.price, trade.quantity, trade.timestamp)
            if trade_id in self.processed_trades:
                continue
            self.processed_trades.add(trade_id)
            if trade.symbol != self.symbol:
                continue

            # 校验是否是自己的成交
            if trade.buyer == "SUBMISSION":
                market_delta += trade.quantity
            elif trade.seller == "SUBMISSION":
                market_delta -= trade.quantity

        # 计算 conversion 成交量 = 仓位变化 - market trade
        conversion_executed = delta_pos - market_delta
        if conversion_executed == 0:
            return

        # 获取 conversion 成本信息
        obs = state.observations.conversionObservations.get(self.symbol)
        if obs is None:
            return

        # 记录 conversion 成本进订单历史
        if conversion_executed > 0:
            cost = obs.askPrice + obs.transportFees + obs.importTariff
            self.orders_history['long'].append((cost, conversion_executed))
        else:
            revenue = obs.bidPrice - obs.transportFees - obs.exportTariff
            self.orders_history['short'].append((revenue, -conversion_executed))

    def calculate_fair_value(self, order_depth: OrderDepth) -> float:
        """计算市场公允价值，买卖单一起按数量加权"""

        buy_sum = sum(price * vol for price, vol in order_depth.buy_orders.items())
        sell_sum = sum(price * -vol for price, vol in order_depth.sell_orders.items())
        total_vol = sum(order_depth.buy_orders.values()) + sum(abs(vol) for vol in order_depth.sell_orders.values())
        return (buy_sum + sell_sum) / total_vol if total_vol != 0 else 0

    def determine_optimal_conversions(self, state: TradingState) -> int:
        """基于 avg_cost 判断并决定 conversion 数量"""
        if not state.observations or "MAGNIFICENT_MACARONS" not in state.observations.conversionObservations:
            return 0

        position = state.position.get(self.symbol, 0)
        obs = state.observations.conversionObservations["MAGNIFICENT_MACARONS"]
        avg_cost, net_pos = self.calculate_avg_cost(state)
        order_depth = state.order_depths.get(self.symbol, OrderDepth())
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float('inf')
        best_bid_amount = order_depth.buy_orders.get(best_bid, 0)
        best_ask_amount = order_depth.sell_orders.get(best_ask, 0)
        
        conversions = 0

        buy_cost = obs.askPrice + obs.transportFees + obs.importTariff
        sell_revenue = obs.bidPrice - obs.transportFees - obs.exportTariff

        import_profit = best_bid - buy_cost
        export_profit = sell_revenue - best_ask
        
        cost = 2 * obs.transportFees + obs.importTariff + obs.exportTariff
        spread = best_ask - best_bid
        logger.print(f'spread: {spread}, cost: {cost}')



        #如果还有conversions目标，优先完成
        if self.target_conversions != 0:
            #计算实际进出口数量 target_amount = -conversion
            self.target_amount = - (position - self.last_pos)
            #根据实际进出口数量
            logger.print('converting')
            conversions = min(self.target_conversions, self.conversion_limit) #进出口目标数量
            self.target_conversions = self.target_conversions - conversions
            self.target_amount = -conversions #需要配对的数量

            #清零last_pos
            self.last_pos = 0

            return conversions
        self.last_pos = position #记录仓位

        if net_pos > 0:
            # 有多头，考虑是否可以卖掉套利
            if best_bid - avg_cost > 1:
                conversions = -min(10, best_bid_amount)
        elif net_pos < 0:
            # 有空头，考虑是否可以买回套利
            if avg_cost - best_ask > 1:
                conversions = min(10, best_ask_amount)
        else:
            # 如果当前净仓为0，判断是否主动开仓套利
            if spread > cost + 1.5:
                conversion_amount = min(10, best_ask_amount, best_bid_amount)
                conversions = -conversion_amount
                self.target_amount = conversions
                self.target_conversions = -conversions

        return conversions

    def process_market_data(self, state: TradingState):
        """处理市场数据并更新策略状态"""

        #观察市场
        market_trades = state.market_trades.get(self.symbol, [])
        if market_trades:
            for trade in market_trades:
                buyer = trade.buyer
                seller = trade.seller
                price = trade.price
                quantity = trade.quantity
                timestamp = trade.timestamp
        own_trades = state.own_trades.get(self.symbol, [])
        if own_trades:
            for trade in own_trades:
                quantity = trade.quantity
                price = trade.price
        position = state.position.get(self.symbol, 0)
        self.position_history.append(position)
        
        # 计算并记录当前公允价值
        order_depth = state.order_depths.get(self.symbol, OrderDepth())
        fair_value = self.calculate_fair_value(order_depth)
        if fair_value > 0:
            self.fair_value_history.append(fair_value)
        # 记录行情观察值
        if state.observations and "MAGNIFICENT_MACARONS" in state.observations.conversionObservations:
            self.observation_history.append(state.observations.conversionObservations["MAGNIFICENT_MACARONS"])
        else: #如果没有数据，直接跳过
            return
        #记录糖价
        self.sugar_price_history.append(state.observations.conversionObservations["MAGNIFICENT_MACARONS"].sugarPrice)
        
        if state.observations.conversionObservations["MAGNIFICENT_MACARONS"].sunlightIndex is not None:
            sunlightIndex = state.observations.conversionObservations["MAGNIFICENT_MACARONS"].sunlightIndex
            self.sunlightIndex_history.append(sunlightIndex)
        else:
            sunlightIndex = None
        if sunlightIndex is not None:
            #根据SI或者相关系数设置突破条件
            if sunlightIndex < 45 and self.current_mode == 'Normal' and self.sunlightIndex_slope < 0:
                self.current_mode = "CSI"

        if len(self.sunlightIndex_history) > 5:
            self.sunlightIndex_slope = (self.sunlightIndex_history[-1] - self.sunlightIndex_history[-5]) / 5
        else:
            self.sunlightIndex_slope = 0

    
    def generate_orders(self, state: TradingState) -> List[Order]:
        """生成订单和转换请求"""
        avg_cost, net_pos = self.calculate_avg_cost(state)
        orders = []
        order_depth = state.order_depths.get(self.symbol, OrderDepth())
        position = state.position.get(self.symbol, 0)
        
        # 如果没有订单深度数据
        if not order_depth.buy_orders and not order_depth.sell_orders:
            return orders

        #普通模式
        if self.current_mode == "Normal":
            logger.print(f"target amount: {self.target_amount}, target conversions: {self.target_conversions}")
            if self.target_amount != 0:
                #配对下单
                target_orders, rest_amount = self.quick_trade(self.symbol, state, self.target_amount)
                orders.extend(target_orders)
                self.target_amount = rest_amount
            else:
                # 获取最佳买卖价格
                best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
                best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float('inf')

                # 计算市场公允价值
                fair_value = self.calculate_fair_value(order_depth)

                # 仓位调整系数 - 仓位越大，卖出意愿越强
                position_factor = 5 * position / self.position_limit
                #storage_factor = 0.2 * position if position > 0 else 0
                adjusted_fair_value = fair_value - position_factor 

                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                # 设置买卖价格
                buy_price = best_bid + 1
                sell_price = best_ask - 1
                spread = best_ask - best_bid
                buy_sum = sum(price * vol for price, vol in order_depth.buy_orders.items())
                sell_sum = sum(price * -vol for price, vol in order_depth.sell_orders.items())
                imbalance_ratio = buy_sum / (buy_sum + sell_sum)
                position_ratio = position / self.position_limit

                available_buy = max(0, self.position_limit - position)
                available_sell = max(0, position + self.position_limit)
                if spread > 4:
                    buy_amount = min(5, available_buy * imbalance_ratio * (1 - position_ratio))
                    buy_amount = int(buy_amount) if buy_amount <= 5 else 5
                    available_buy -= buy_amount
                    sell_amount = int(min(5, available_sell) * (1 - imbalance_ratio) * position_ratio)
                    sell_amount = int(sell_amount) if sell_amount <= 5 else 5
                    available_sell -= sell_amount
                    orders.append(Order(self.symbol, buy_price, buy_amount))
                    orders.append(Order(self.symbol, sell_price, -sell_amount)) 
            return orders

        #做多
        if self.current_mode == "CSI":
            if self.sunlightIndex_slope > 0 and self.sunlightIndex_history[-1] > 30: #退出条件
                self.current_mode = "Closing position"
            #做多
            max_buy_amount, _ = self.get_available_amount(self.symbol, state)
            max_buy_amount = min(max_buy_amount, self.position_limit - position)
            quick_orders, _ = self.quick_trade(self.symbol, state, max_buy_amount)
            orders.extend(quick_orders)


        #做空
        if self.current_mode == "Closing position":
            if position <= -70 and self.sunlightIndex_history[-1] > 40:
                clear_orders, _ = self.quick_trade(self.symbol, state, -position) #快速清仓
                orders.extend(clear_orders)
                self.current_mode = "Normal"
            else:
                _, max_sell_amount = self.get_available_amount(self.symbol, state)
                max_sell_amount = min(max_sell_amount, position + self.position_limit)
                quick_orders, _ = self.quick_trade(self.symbol, state, -max_sell_amount)
                orders.extend(quick_orders)

        return orders


    def run(self, state: TradingState) -> Tuple[List[Order], dict, int]:
        """执行策略主逻辑，同时返回订单、状态和转换请求"""

        # 处理市场数据      
        self.process_market_data(state)

        # 更新仓储成本
        self.update_conversion_cost(state)
        self.update_storage_cost(state)

        # 确定最优的转换数量
        conversions = self.determine_optimal_conversions(state)

        # 生成普通订单
        orders = self.generate_orders(state)

        # 保存策略状态
        strategy_state = self.save_state(state)

        return orders, strategy_state, conversions

    def save_state(self, state) -> dict:
        """保存策略状态"""
        return {
            "fair_value_history": list(self.fair_value_history),
            "position_history": list(self.position_history),
        }

    def load_state(self, state):
        """加载策略状态"""
        if not hasattr(state, 'traderData') or not state.traderData:
            return

        try:
            trader_data = json.loads(state.traderData)
            if self.symbol in trader_data:
                strategy_data = trader_data[self.symbol]

                if "fair_value_history" in strategy_data:
                    self.fair_value_history = deque(strategy_data["fair_value_history"], maxlen=100)

                if "position_history" in strategy_data:
                    self.position_history = deque(strategy_data["position_history"], maxlen=100)

        except Exception as e:
            logger.print(f"Error loading state: {str(e)}")

class Config:
    def __init__(self):
        self.PRODUCT_CONFIG = {
            "MAGNIFICENT_MACARONS": {
                "strategy_cls": MacaronStrategy,
                "symbol": "MAGNIFICENT_MACARONS",
                "position_limit": 75,
                "conversion_limit": 10
            }
        }

class Trader:
    def __init__(self, product_config=None):
        # 使用默认 config，或外部传入 config
        self.PRODUCT_CONFIG = product_config if product_config is not None else Config().PRODUCT_CONFIG
        self.strategies = {}
        self._init_strategies()

    def _init_strategies(self):
        for product, config in self.PRODUCT_CONFIG.items():
            if product == "PICNIC_BASKET_GROUP":
                cls = config["strategy_cls"]
                args = {k: v for k, v in config.items() if k != "strategy_cls" and k != "symbols"}
                self.strategies[product] = cls(symbols=config["symbols"], **args)
            else:
                # 常规产品初始化保持不变
                cls = config["strategy_cls"]
                args = {k: v for k, v in config.items() if k != "strategy_cls"}
                self.strategies[product] = cls(**args)

    def run(self, state: TradingState):
        # 加载历史状态
        trader_data = json.loads(state.traderData) if state.traderData else {}

        orders = {}
        new_trader_data = {}

        # 先处理马卡龙转换请求
        if "MAGNIFICENT_MACARONS" in self.strategies and (
                "MAGNIFICENT_MACARONS" in state.order_depths or
                (state.observations and "MAGNIFICENT_MACARONS" in state.observations.conversionObservations)):

            strategy = self.strategies["MAGNIFICENT_MACARONS"]
            if "MAGNIFICENT_MACARONS" in trader_data:
                strategy.load_state(state)

            # 马卡龙策略返回额外的conversions
            macaron_orders, strategy_state, macaron_conversions = strategy.run(state)
            orders["MAGNIFICENT_MACARONS"] = macaron_orders
            new_trader_data["MAGNIFICENT_MACARONS"] = strategy_state
            conversions = macaron_conversions
        logger.print(conversions)

        # 处理其他产品订单
        for product, strategy in self.strategies.items():
            # 跳过已处理的马卡龙
            if product == "MAGNIFICENT_MACARONS":
                continue

            if product in trader_data or product == "PICNIC_BASKET_GROUP":
                strategy.load_state(state)

            if product in state.order_depths or product == "PICNIC_BASKET_GROUP":
                product_orders, strategy_state = strategy.run(state)

                # 处理group订单
                if isinstance(product_orders, dict):
                    for symbol, symbol_orders in product_orders.items():
                        if symbol not in orders:
                            orders[symbol] = []
                        orders[symbol].extend(symbol_orders)
                else:
                    orders[product] = product_orders

                new_trader_data[product] = strategy_state

        trader_data.update(new_trader_data)
        trader_data = json.dumps(trader_data)
        logger.flush(state, orders, conversions, trader_data)

        return orders, conversions, trader_data