from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import numpy as np
import json

class Trader:
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        result = {}
        print("Current position :", state.position)
        profit_pct_limit = 0.0005
        for product in state.order_depths:
            print(f'Trading {product}')
            if product == 'RAINFOREST_RESIN':
                result[product] = self.trade_rainforest_resin(state, product, profit_pct_limit)
                continue
            if product == 'KELP':
                #result[product] = self.trade_kelp(state, product, profit_pct_limit)
                continue
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData
    
    def estimate_fair_price(self, state: TradingState, product: str) -> int:
        # Estimate fair price based on market data and trader's observations
    
        # 用市场买卖加权均价
        order_depth = state.order_depths.get(product, OrderDepth())
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = sum(price * amount for price, amount in order_depth.buy_orders.items()) / sum(amount for price, amount in order_depth.buy_orders.items())
            best_ask = sum(price * amount for price, amount in order_depth.sell_orders.items()) / sum(amount for price, amount in order_depth.sell_orders.items())
            print('Using order depth to estimate fair price')
            fair_price = (best_bid + best_ask) / 2
            volatility = self.calculate_market_volatility(state, product)
            spread = self.calculate_spread(volatility)
            expect_bid = fair_price - spread / 2
            expect_ask = fair_price + spread / 2
            return fair_price, expect_bid, expect_ask
        else:
            if product == 'KELP':
                print('Using default price of 2025')
                return 2025
            if product == 'RAINFOREST_RESIN':
                print('Using default price of 0')
                return 0


    def get_best_price(self, orders: list, depth: int):
        """ 获取第 depth 层的最优价格和数量 """
        #判断是否越界
        if not (0 <= depth < len(orders)):  
            return None, None
        return orders[depth][0], orders[depth][1] 
    
    
    def check_position_risk(self, position:dict, product:str) -> bool:
        position_limit = 1000 #持仓限制
        if product in position and abs(position[product])  >=  position_limit:
            return False
        return True

    
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
    

    def calculate_ma(self, state, product:str):
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
    
    def trade_rainforest_resin(self, state, product, profit_pct_limit):
        order_depth: OrderDepth = state.order_depths[product]
        buy_orders = [list(order) for order in order_depth.buy_orders.items()]
        sell_orders = [list(order) for order in order_depth.sell_orders.items()]
        orders: List[Order] = []
        position = state.position.get(product, 0)
        print("Buy Order depth : " + str(len(buy_orders)) + ", Sell order depth : " + str(len(sell_orders)))

        #check order pairs from the shallowest to the deepest
        i, j = 0, 0
        fair_price, expect_bid, expect_ask = self.estimate_fair_price(state, product)
        print("Fair price : " + str(fair_price))
        print("Expect bid : " + str(expect_bid))
        print("Expect ask : " + str(expect_ask))
        while i < len(sell_orders) and j < len(buy_orders):
            ask_price, ask_amount = self.get_best_price(sell_orders, i)
            bid_price, bid_amount = self.get_best_price(buy_orders, j)
            if ask_price is None or bid_price is None:
                print(f"[Warning] ask_price or bid_price is None at depth {i}, {j}")
                break  # 防止死循环

            if ask_amount == 0:
                print(f'Ask amount at depth {i} has been fully filled. Skipping')
                i += 1
                continue
            if bid_amount == 0:
                print(f'Bid amount at depth {j} has been fully filled. Skipping')
                j += 1
                continue
            
            if ask_price is None:
                print(f'Ask price at depth {i} is None. Skipping')
                i += 1
                continue
            if bid_price is None:
                print(f'Bid price at depth {j} is None. Skipping')
                j += 1
                continue

            print(f'depth of ask order: {i}, price: {ask_price}, amount: {ask_amount}')
            print(f'depth of bid order: {j}, price: {bid_price}, amount: {bid_amount}')


            if ask_price < expect_ask:
                print(f'Asking price too low, price: {ask_price}, expect_ask: {expect_ask}')
                i += 1
                continue
            if bid_price > expect_bid:
                print(f'Bidding price too high, price: {bid_price}, expect_bid: {expect_bid}')
                j += 1
                continue

            spread = ask_price - bid_price
            if spread / fair_price > profit_pct_limit:
                print(f'Spread is profitable, spread_pct: {spread/fair_price*100:.2f}')
                amount = min(-ask_amount, bid_amount)

                if position > 0:  
                    # 如果当前持仓为正，优先卖出
                    print(f'Current position is positive {position}, selling')
                    sell_amount = min(amount, position)  # 不能卖出超过持仓的数量
                    orders.append(Order(product, ask_price, -sell_amount))
                    sell_orders[i][1] += sell_amount
                    position -= sell_amount
                    print(f'Selling {sell_amount} at {ask_price}, Current position {position}')
                elif position < 0:
                    # 如果当前持仓为负，优先买入平仓
                    print(f'Current position is negative {position}, buying to close position')
                    buy_amount = min(amount, -position)  
                    orders.append(Order(product, bid_price, buy_amount))
                    buy_orders[j][1] -= buy_amount
                    position += buy_amount
                    print(f'Buying {buy_amount} at {bid_price}, Current position {position}')
                else:
                    # 如果没有持仓，同时买入和卖出
                    print('No position, executing market making')
                    orders.append(Order(product, bid_price, amount))
                    orders.append(Order(product, ask_price, -amount))
                    buy_orders[j][1] -= amount
                    sell_orders[i][1] += amount

                    print(f'Executing market making: Buy {amount} at {bid_price}, Sell {amount} at {ask_price}')
                    
            i += 1
            j += 1
            if i >= len(sell_orders) or j >= len(buy_orders):
                break
            
        return orders
    

    def trade_kelp(self, state, product, profit_pct_limit):
        order_depth: OrderDepth = state.order_depths[product]
        buy_orders = [list(order) for order in order_depth.buy_orders.items()]
        sell_orders = [list(order) for order in order_depth.sell_orders.items()]
        orders: List[Order] = []
        position = state.position.get(product, 0)
        print("Buy Order depth : " + str(len(buy_orders)) + ", Sell order depth : " + str(len(sell_orders)))

        #check order pairs from the shallowest to the deepest
        i, j = 0, 0
        fair_price, expect_bid, expect_ask = self.estimate_fair_price(state, product)
        print("Fair price : " + str(fair_price))
        print("Expect bid : " + str(expect_bid))
        print("Expect ask : " + str(expect_ask))
        while i < len(sell_orders) and j < len(buy_orders):
            ask_price, ask_amount = self.get_best_price(sell_orders, i)
            bid_price, bid_amount = self.get_best_price(buy_orders, j)

            if ask_amount == 0:
                print(f'Ask amount at depth {i} has been fully filled. Skipping')
                i += 1
                continue
            if bid_amount == 0:
                print(f'Bid amount at depth {j} has been fully filled. Skipping')
                j += 1
                continue
            
            if ask_price is None:
                print(f'Ask price at depth {i} is None. Skipping')
                i += 1
                continue
            if bid_price is None:
                print(f'Bid price at depth {j} is None. Skipping')
                j += 1
                continue

            print(f'depth of ask order: {i}, price: {ask_price}, amount: {ask_amount}')
            print(f'depth of bid order: {j}, price: {bid_price}, amount: {bid_amount}')


            if ask_price < expect_ask:
                print(f'Asking price too low, price: {ask_price}, expect_ask: {expect_ask}')
                i += 1
                continue
            if bid_price > expect_bid:
                print(f'Bidding price too high, price: {bid_price}, expect_bid: {expect_bid}')
                j += 1
                continue

            spread = ask_price - bid_price
            if spread / fair_price > profit_pct_limit:
                print(f'Spread is profitable, spread_pct: {spread/fair_price*100:.2f}')
                amount = min(-ask_amount, bid_amount)

                if position > 0:  
                    # 如果当前持仓为正，优先卖出
                    print(f'Current position is positive {position}, selling')
                    sell_amount = min(amount, position)  # 不能卖出超过持仓的数量
                    orders.append(Order(product, ask_price, -sell_amount))
                    sell_orders[i][1] += sell_amount
                    position -= sell_amount
                    print(f'Selling {sell_amount} at {ask_price}, Current position {position}')
                elif position < 0:
                    # 如果当前持仓为负，优先买入平仓
                    print(f'Current position is negative {position}, buying to close position')
                    buy_amount = min(amount, -position)  
                    orders.append(Order(product, bid_price, buy_amount))
                    buy_orders[j][1] -= buy_amount
                    position += buy_amount
                    print(f'Buying {buy_amount} at {bid_price}, Current position {position}')
                else:
                    # 如果没有持仓，同时买入和卖出
                    print('No position, executing market making')
                    orders.append(Order(product, bid_price, amount))
                    orders.append(Order(product, ask_price, -amount))
                    buy_orders[j][1] -= amount
                    sell_orders[i][1] += amount

                    print(f'Executing market making: Buy {amount} at {bid_price}, Sell {amount} at {ask_price}')
                    
                i += 1
                j += 1
            
        return orders 
    
