from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import numpy as np
import json

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
        print("Observations: " + str(state.observations))
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
            print(f'Trading {product}')

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
        return result, conversions, traderData
    
    def estimate_fair_price(self, state: TradingState, product: str) -> int:
        # Estimate fair price based on market data and trader's observations
    
        # 用市场买卖加权均价
        order_depth = state.order_depths.get(product, OrderDepth())
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = sum(price * amount  for price, amount in order_depth.buy_orders.items()) / sum(amount for price, amount in order_depth.buy_orders.items())
            best_ask = sum(price * amount  for price, amount in order_depth.sell_orders.items()) / sum(amount for price, amount in order_depth.sell_orders.items())
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
                return 2025
            if product == 'RAINFOREST_RESIN':
                print(f'Product {product} has no order depth. Using default price of 10000')
                return 10000


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
            print(f'------------Trading {product}------------')
            
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

        #加载基本信息
        order_depth: OrderDepth = state.order_depths[product]
        buy_orders = [list(order) for order in order_depth.buy_orders.items()]
        sell_orders = [list(order) for order in order_depth.sell_orders.items()]
        orders: List[Order] = []
        position = state.position.get(product, 0)
        print(f"Product {product} ,Buy Order depth : " + str(len(buy_orders)) + f", Sell order depth : " + str(len(sell_orders)))
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
        momentum = self.price_momentum(historical_prices, product)
        print(f'Product {product}, Price momentum: {momentum:.2f}')
        obi = self.orderbook_imbalance(state, product, fair_price)
        print(f'Product {product}, Orderbook imbalance: {obi:.2f}')
        fair_price = fair_price + (alpha * position + beta * momentum + gamma * obi)
        print(f'Product {product}, Adjusted fair price: {fair_price:.2f}, Alpha {alpha:.2f}, Beta {beta:.2f}, Gamma {gamma:.2f}')

        #交易逻辑
        #从浅到深检查订单簿
        i, j = 0, 0
        trading_info = str()
        count = 0
        while i < len(sell_orders) or j < len(buy_orders):
            count += 1
            print(f'Product {product}, i: {i}, j: {j}')
            print(f'Current Orderbook: Sell orders: {sell_orders}, Buy orders: {buy_orders}')
            trading_info += f"Product {product}, i: {i}, j: {j}\n"
            ask_price, ask_amount = self.get_best_price(sell_orders, i)
            bid_price, bid_amount = self.get_best_price(buy_orders, j)

            if ask_price is None or bid_price is None:
                print(f"[Warning] Product {product}, ask_price or bid_price is None at depth {i}, {j}")
                trading_info += f"Product {product}, ask_price or bid_price is None at depth {i}, {j}\n"
                break  # 防止死循环


            #跳过amount为0或者价格为None的订单
            if i < len(sell_orders) or j < len(buy_orders): #确保没有超出orderbook范围
                if ask_amount == 0 and i < len(sell_orders):
                    print(f'Product {product}, Ask amount at depth {i} has been fully filled. Skipping')
                    trading_info += f"Product {product}, Ask amount at depth {i} has been fully filled. Skipping\n"
                    i += 1
                    continue

                if bid_amount == 0 and j < len(buy_orders):
                    print(f'Product {product}, Bid amount at depth {j} has been fully filled. Skipping')
                    trading_info += f"Product {product}, Bid amount at depth {j} has been fully filled. Skipping\n"
                    j += 1
                    continue
                
                if ask_price is None and i < len(sell_orders):
                    print(f'Product {product}, Ask price at depth {i} is None. Skipping')
                    trading_info += f"Product {product}, Ask price at depth {i} is None. Skipping\n"
                    i += 1
                    continue

                if bid_price is None and j < len(buy_orders):
                    print(f'Product {product}, Bid price at depth {j} is None. Skipping')
                    trading_info += f"Product {product}, Bid price at depth {j} is None. Skipping\n"
                    j += 1
                    continue

            print(f'Product {product}, depth of ask order: {i}, price: {ask_price}, amount: {ask_amount}')
            print(f'Product {product}, depth of bid order: {j}, price: {bid_price}, amount: {bid_amount}')
            trading_info += f"Product {product}, depth of ask order: {i}, price: {ask_price}, amount: {ask_amount}\n"
            trading_info += f"Product {product}, depth of bid order: {j}, price: {bid_price}, amount: {bid_amount}\n"
            
            #主动交易
            #ask_price小于fair_price，直接买入
            
            if ask_price < fair_price:
                print(f'Product {product}. Asking price is lower than fair price, price: {ask_price}, fair_price: {fair_price}')
                trading_info += f"Product {product}. Asking price is lower than fair price, price: {ask_price}, fair_price: {fair_price}\n"
                #最大可以买入的amount
                amount = min(ask_amount, position_limit - position)
                orders.append(Order(product, ask_price, -amount))
                position += amount

            #如果bid_price大于fair_price，直接买入
            if bid_price > fair_price:
                print(f'Product {product}. Bidding price is higher than fair price, price: {bid_price}, fair_price: {fair_price}')
                trading_info += f"Product {product}. Bidding price is higher than fair price, price: {bid_price}, fair_price: {fair_price}\n"
                #最大可卖出（做空）的amount
                amount = min(bid_amount, position_limit + position)
                orders.append(Order(product, bid_price, amount))
                position -= amount
            '''
            #结合订单簿不平衡度来判断
            if ask_price < fair_price:
                spread_pct = (fair_price - ask_price) / fair_price
                obi_weight = max(0, min(1, obi))  # 限制在 [0,1] 之间
                print(f'spread_pct: {spread_pct:.2f}, obi_weight: {obi_weight:.2f}')
                if spread_pct > profit_pct_limit * (1 + obi_weight):  # 结合 OBI 限制
                    amount = min(-ask_amount, position_limit - position)
                    orders.append(Order(product, ask_price, amount))
                    position += amount
                    i += 1

            if bid_price > fair_price:
                spread_pct = (bid_price - fair_price) / fair_price
                obi_weight = max(0, min(1, -obi))  # 取 OBI 的反向
                if spread_pct > profit_pct_limit * (1 + obi_weight):  # 结合 OBI 限制
                    amount = min(bid_amount, position_limit + position)
                    orders.append(Order(product, bid_price, -amount))
                    position -= amount
                    j += 1

            
            也可以这样做，应该更加保守？待测试
            if ask_price < expect_ask:
                print(f'Asking price too low, price: {ask_price}, expect_ask: {expect_ask}')
                i += 1
                continue
            if bid_price > expect_bid:
                print(f'Bidding price too high, price: {bid_price}, expect_bid: {expect_bid}')
                j += 1
                continue'
            '''

            #做市
            spread = ask_price - bid_price
            #抢单，保证成交
            ask_price -= 1
            bid_price += 1
            if spread / fair_price > profit_pct_limit:
                print(f'Product {product}. Spread is profitable, spread_pct: {spread/fair_price*100:.2f}')
                trading_info += f"Product {product}. Spread is profitable, spread_pct: {spread/fair_price*100:.2f}\n"
    
                if position > 0:  
                    # 如果当前持仓为正，优先卖出
                    print(f'Product {product}, Current position is positive {position}, selling')
                    trading_info += f"Product {product}, Current position is positive {position}, selling\n"
                    sell_amount = min(ask_amount, position + position_limit)
                    orders.append(Order(product, ask_price, sell_amount))
                    sell_orders[i][1] += sell_amount
                    position -= sell_amount
                    print(f'Selling product {product}, selling {sell_amount} at {ask_price}, Current position {position}')
                    trading_info += f"Selling product {product}, selling {sell_amount} at {ask_price}, Current position {position}\n"

                elif position < 0:
                    # 如果当前持仓为负，优先买入平仓
                    print(f'Product {product}, Current position is negative {position}, buying to close position')
                    trading_info += f"Product {product}, Current position is negative {position}, buying to close position\n"
                    buy_amount = min(bid_amount, position_limit - position)  #仓位限制
                    orders.append(Order(product, bid_price, -buy_amount))
                    buy_orders[j][1] -= buy_amount
                    position += buy_amount
                    print(f'Buying product {product}, buying {buy_amount} at {bid_price}, Current position {position}')
                    trading_info += f"Buying product {product}, buying {buy_amount} at {bid_price}, Current position {position}\n"
                    
                else:
                    # 如果没有持仓，同时买入和卖出
                    amount = min(ask_amount, bid_amount)
                    amount = min(amount, position_limit) #仓位限制
                    print(f'Product {product}, No position, executing market making')
                    trading_info += f"Product {product}, No position, executing market making\n"
                    orders.append(Order(product, bid_price, -amount)) #买入
                    orders.append(Order(product, ask_price, amount)) #卖出
                    sell_orders[i][1] += amount
                    buy_orders[j][1] -= amount
                    


                    print(f'Product {product}, Executing market making: Buy {amount} at {bid_price}, Sell {amount} at {ask_price}')
                    trading_info += f"Product {product}, Executing market making: Buy {amount} at {bid_price}, Sell {amount} at {ask_price}\n"
            else:
                print(f'Product {product}. Spread is not profitable, spread: {spread}, fair_price: {fair_price}, spread_pct: {spread/fair_price*100:.2f}')   
                i += 1
                j += 1
            #更新i,j
            if sell_orders[i][1]== 0:
                print(f'Product {product}, Ask amount at depth {i} has been fully filled. Updating i')
                trading_info += f"Product {product}, Ask amount at depth {i} has been fully filled. Updating i\n"
                i += 1
            if buy_orders[j][1] == 0:
                print(f'Product {product}, Bid amount at depth {j} has been fully filled. Updating j')
                trading_info += f"Product {product}, Bid amount at depth {j} has been fully filled. Updating j\n"
                j += 1

            if i >= len(sell_orders) or j >= len(buy_orders):
                if i >= len(sell_orders):
                    if j < len(buy_orders):
                        j += 1  # 买单不足，继续补齐
                    else:
                        print(f'Product {product}, All orders have been filled, exit loop')
                        trading_info += f"Product {product}, All orders have been filled, exit loop\n"
                        break
                if j >= len(buy_orders):
                    if i < len(sell_orders):
                        i += 1  # 卖单不足，继续补齐
                    else:
                        print(f'Product {product}, All orders have been filled, exit loop')
                        trading_info += f"Product {product}, All orders have been filled, exit loop\n"
                        break
        #print('Product', product, 'Trading_info: ', trading_info) 
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