from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order


class Trader:

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """

        result = {}

        for product in state.order_depths.keys():
            if product != 'MAGNIFICENT_MACARONS':
                continue

            order_depth: OrderDepth = state.order_depths[product]

            orders: list[Order] = []

            position = state.position.get(product, 0)  
            if len(order_depth.sell_orders) > 0:
                for price, volume in order_depth.sell_orders.items():
                    if -volume > 0:
                        buy_amount = min(-volume, 75 - position)
                        position += volume
                        orders.append(Order(product, price, buy_amount))

            result[product] = orders
                
        traderData = "SAMPLE" 
        
        conversions = 0

                # Return the dict of orders
                # These possibly contain buy or sell orders
                # Depending on the logic above
        
        return result, conversions, traderData
