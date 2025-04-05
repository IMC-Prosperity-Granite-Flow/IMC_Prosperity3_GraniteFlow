from datamodel import OrderDepth, UserId, TradingState, Order, Observation
from typing import List
import string
import numpy as np
import json

from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


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
        
class Trader:
    
    def run(self, state: TradingState):
        if not state.traderData or state.traderData.strip() == "":
            trader_data = {}  # 如果为空，初始化为空字典
        else:
            try:
                trader_data = json.loads(state.traderData)
            except json.JSONDecodeError:
                print(f"Warning: Failed to decode traderData: {state.traderData}")
                trader_data = {}

        print("traderData (Loaded):", json.dumps(trader_data))
        #print("Observations: " + str(state.observations))
        result = {}
        print("Current position :", state.position)
        profit_pct_limit = 0.001
        
        position_limit = 50
        for product in state.order_depths:
            if product == 'KELP':
                profit_pct_limit = 0.0015
            if product == 'RAINFOREST_RESIN':
                profit_pct_limit = 0.0003
            #取出历史 fair_price
            historical_prices = trader_data.get(product, [])
            print(f'------------Trading {product}------------')

            fair_price, _, _ = self.estimate_fair_price(state, product)
            historical_prices.append(fair_price)

            #控制历史数据长度
            trader_data[product] = historical_prices[-20:]

            #交易
            result[product] = self.trade(state, product, profit_pct_limit, position_limit, historical_prices)

        #存回 JSON
        traderData = json.dumps(trader_data)
        
        # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
    
    def estimate_fair_price(self, state: TradingState, product: str) -> int:
        # Estimate fair price based on market data and trader's observations
    
        # 用市场买卖加权均价
        order_depth = state.order_depths.get(product, OrderDepth())
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = sum(price * amount for price, amount in order_depth.buy_orders.items()) / sum(amount for price, amount in order_depth.buy_orders.items())
            best_ask = sum(price * amount for price, amount in order_depth.sell_orders.items()) / sum(amount for price, amount in order_depth.sell_orders.items())
            #print('Using order depth to estimate fair price')
            fair_price = (best_bid + best_ask) / 2
            volatility = self.calculate_market_volatility(state, product)
            spread = self.calculate_spread(volatility)
            expect_bid = fair_price - spread / 2
            expect_ask = fair_price + spread / 2
            return fair_price, expect_bid, expect_ask
        else:
            if product == 'KELP':
                print(f'Product {product} has no order depth. Using default price of 2025')
                return 2025,1 , 1
            if product == 'RAINFOREST_RESIN':
                print(f'Product {product} has no order depth. Using default price of 10000')
                return 10000,1, 1 


    def get_best_price(self, orders: list, depth: int):
        """ 获取第 depth 层的最优价格和数量 """
        #判断是否越界
        if not (0 <= depth < len(orders)):  
            return None, None
        return orders[depth][0], orders[depth][1] 
    

    def calculate_market_volatility(self, state: TradingState, product: str) -> float:
    # 计算市场的波动性，可以用标准差来衡量
        recent_trades = state.market_trades.get(product, [])
        if len(recent_trades) > 1:
            prices = [trade.price for trade in recent_trades]
            return np.std(prices)  # 返回价格标准差作为波动性
        return 0  # 如果交易数据不足，则返回0

    def calculate_spread(self, volatility: float) -> float:
        # 根据市场波动性计算买卖差价
        if volatility > 0:
            return volatility * 2
        return 0.01  # 默认最小差价
    

    def calculate_ma(self, state):
        try:
            trader_data = json.loads(state.traderData) if state.traderData else {}
        except json.JSONDecodeError:
            trader_data = {}

        for product in state.order_depths:
            print(f'Trading {product}')
            
            # 计算 mid_price
            fair_price, _, _ = self.estimate_fair_price(state, product)
            if fair_price is None:
                continue

            # 存储 mid_price
            if product not in trader_data:
                trader_data[product] = []
            trader_data[product].append(fair_price)

            # 只保留最近 20 条数据（避免 traderData 过长）
            trader_data[product] = trader_data[product][-20:]

        # 更新 traderData（存储为 JSON 字符串）
        traderData = json.dumps(trader_data)
        return traderData
    
    def trade(self, state, product, profit_pct_limit, position_limit, historical_prices):
        order_depth: OrderDepth = state.order_depths[product]
        buy_orders = [list(order) for order in order_depth.buy_orders.items()]
        sell_orders = [list(order) for order in order_depth.sell_orders.items()]
        #判断orders的长度，如果不够，补全[0,0]
        orderbook_depth = max(len(buy_orders), len(sell_orders))
        if len(buy_orders) < orderbook_depth:
            buy_orders += [[0, 0]] * (orderbook_depth - len(buy_orders))
        if len(sell_orders) < orderbook_depth:
            sell_orders += [[0, 0]] * (orderbook_depth - len(sell_orders))
        #打印订单簿
        print(f'Product {product}, Buy orders: {buy_orders}, Sell orders: {sell_orders}')
        orders: List[Order] = []
        position = state.position.get(product, 0)
        print(f'Product {product}, Position: {position}')


        fair_price, expect_bid, expect_ask = self.estimate_fair_price(state, product)
        #暂时锁死resin的fair_price
        if product == 'RAINFOREST_RESIN':
            fair_price = 10000
        print(f"Product {product}, Fair price : " + str(fair_price))
        print(f"Product {product}, Expect bid : " + str(expect_bid))
        print(f"Product {product}, Expect ask : " + str(expect_ask))
        #根据持仓、动量预测、订单簿不平衡度调整fair price
        alpha = 0
        beta = 0
        gamma = 0
        print(f'Product {product}, Position: {position}')
        print(product, type(product))
        momentum = self.price_momentum(historical_prices, product)
        print(f'Product {product}, Price momentum: {momentum:.2f}')
        obi = self.orderbook_imbalance(state, product, fair_price)
        print(f'Product {product}, Orderbook imbalance: {obi:.2f}')
        fair_price = fair_price + (alpha * position + beta * momentum + gamma * obi)
        print(f'Product {product}, Adjusted fair price: {fair_price:.2f}, Alpha {alpha:.2f}, Beta {beta:.2f}, Gamma {gamma:.2f}')

        #从深到前检查订单簿
        i, j = len(sell_orders)-1, len(buy_orders)-1
        while i >= 0 and j >= 0:
            #注意ask_amount是负数
            ask_price, ask_amount = sell_orders[i][0], sell_orders[i][1]
            bid_price, bid_amount = buy_orders[j][0], buy_orders[j][1]
            print(f'Dealing with depth {i}, {j}, ask_price: {ask_price}, ask_amount: {ask_amount}, bid_price: {bid_price}, bid_amount: {bid_amount}')

            if ask_amount == 0:
                print(f'Product {product}, Ask amount at depth {i} has been fully filled. Skipping')
                i -= 1
                continue

            if bid_amount == 0:
                print(f'Product {product}, Bid amount at depth {j} has been fully filled. Skipping')
                j -= 1
                continue
            
            #print(f'Product {product}, depth of ask order: {i}, price: {ask_price}, amount: {ask_amount}')
            #print(f'Product {product}, depth of bid order: {j}, price: {bid_price}, amount: {bid_amount}')
            
            #主动交易，此处待引入头寸控制机制
            #ask_price + 1小于fair_price，直接买入
            if ask_price + 1 < fair_price:
                print(f'Product {product}. Asking price is lower than fair price, price: {ask_price}, fair_price: {fair_price}')
                #最大可以买入的amount
                amount = min(-ask_amount, position_limit - position)
                orders.append(Order(product, ask_price + 1 , -amount)) # 买入时传入-amount
                sell_orders[j][1] += amount
                position += amount

            #如果bid_price - 1大于fair_price，直接买入
            if bid_price - 1 > fair_price:
                print(f'Product {product}. Bidding price is higher than fair price, price: {bid_price}, fair_price: {fair_price}')
                #最大可卖出（做空）的amount
                amount = min(bid_amount, position_limit + position)
                orders.append(Order(product, bid_price - 1, amount)) # 卖出时传入amount
                buy_orders[j][1] -= amount
                position -= amount

            '''
            这一部分也可以这样做，应该更加保守？待测试
            if ask_price < expect_ask:
                print(f'Asking price too low, price: {ask_price}, expect_ask: {expect_ask}')
                i -= 1
                continue
            if bid_price > expect_bid:
                print(f'Bidding price too high, price: {bid_price}, expect_bid: {expect_bid}')
                j -= 1
                continue'
            '''

            #做市
            spread = ask_price - bid_price
            #抢单，保证成交
            ask_price -= 1
            bid_price += 1
            if spread / fair_price > profit_pct_limit:
                print(f'Product {product}. Spread is profitable, spread_pct: {spread/fair_price*100:.2f}')
                #清仓机制，待改进为软硬清仓
                if position > 0:  
                    # 如果当前持仓为正，优先卖出
                    print(f'Product {product}, Current position is negative {position}, buying to close position')
                    #只卖出到平仓
                    sell_amount = min(-ask_amount, position)
                    orders.append(Order(product, ask_price, -sell_amount))
                    sell_orders[i][1] += sell_amount
                    position -= sell_amount
                    print(f'Selling product {product}, selling {sell_amount} at {ask_price}, Current position {position}')
                elif position < 0:
                    # 如果当前持仓为负，优先买入平仓
                    print(f'Product {product}, Current position is negative {position}, buying to close position')
                    buy_amount = min(bid_amount, -position)  #仓位限制
                    orders.append(Order(product, bid_price, buy_amount))
                    buy_orders[j][1] -= buy_amount
                    position += buy_amount
                    print(f'Buying product {product}, buying {buy_amount} at {bid_price}, Current position {position}')
                else:
                    # 如果没有持仓，同时买入和卖出
                    amount = min(-ask_amount, bid_amount)
                    amount = min(amount, position_limit) #仓位限制
                    print(f'Product {product}, No position, executing market making')
                    orders.append(Order(product, bid_price, -amount))
                    orders.append(Order(product, ask_price, amount))
                    buy_orders[j][1] -= amount
                    sell_orders[i][1] += amount

                    print(f'Executing market making: Product {product}, Buy {amount} at {bid_price}, Sell {amount} at {ask_price}')
            #如果spread不够，则直接跳出（以后可以根据更多的条件确认要不要吃下这些订单， 比如跟expect bid和expect ask比较）
            else:
                print(f'Product {product}. Spread is not profitable, spread_pct: {spread/fair_price*100:.2f}, exit loop')
                break
            if i <= 0 and j <= 0:
                print(f'Product {product}, Reach orderbook edge, i: {i}, j: {j}, exit loop')
                break
            
        return orders
    
    
    def orderbook_imbalance(self, state, product: str, fair_price: float) -> float:
        """ 计算订单簿不平衡度 """  
        order_depth: OrderDepth = state.order_depths[product]
        
        buy_orders = [(price, amount) for price, amount in order_depth.buy_orders.items()]
        sell_orders = [(price, amount) for price, amount in order_depth.sell_orders.items()]
        
        # 根据价格离公平价的远近加权
        buy_pressure = sum(amount * np.exp(-(fair_price - price)) for price, amount in buy_orders if price != 0)
        sell_pressure = sum(amount * np.exp(-(price - fair_price)) for price, amount in sell_orders if price != 0)

        total_pressure = buy_pressure + sell_pressure
        if total_pressure == 0:
            return 0
        return (buy_pressure - sell_pressure) / total_pressure
    
    def calculate_ema(self, prices: List[float], span: int, product: str) -> float:
        """ 计算指数移动平均 (EMA) """
        if not prices:
            return 0.0
        alpha = 2 / (span + 1)  # EMA 平滑因子
        ema = prices[0]  # 初始化为第一天的价格
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        print(f'Product {product}, EMA({span}) = {ema:.2f}')
        return ema

    def price_momentum(self, historical_prices: List[int], product: str) -> float:
        '''计算价格动量'''
        print(f'Product {product}, historical prices:', historical_prices[-20:])
        short_ema = self.calculate_ema(historical_prices[-20:], 5, product)
        long_ema = self.calculate_ema(historical_prices[-20:], 20, product)
        return short_ema - long_ema