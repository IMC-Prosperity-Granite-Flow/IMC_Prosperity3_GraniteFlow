import csv
from src.strategy.datamodel import OrderDepth, TradingState, Order, Trade, Observation
#src.strategy.your_strategy_file import YourStrategyClass
from src.optimization.tutorial_v3_opti import Trader


# 从CSV文件加载数据
def load_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file, delimiter=';') 
        for row in reader:
            data.append(row)
    return data


# 解析CSV数据并转换为TradingState对象
def prepare_trading_state(csv_data):
    order_depths = {}
    market_trades = {}
    position = {}
    trading_states = []

    for row in csv_data:
        product = row['product']
        timestamp = int(row['timestamp'])
        
        # Safely convert bid/ask prices and volumes, assigning 0 if they are empty
        buy_orders = {
            int(row['bid_price_1']) if row['bid_price_1'] else 0: int(row['bid_volume_1']) if row['bid_volume_1'] else 0,
            int(row['bid_price_2']) if row['bid_price_2'] else 0: int(row['bid_volume_2']) if row['bid_volume_2'] else 0,
            int(row['bid_price_3']) if row['bid_price_3'] else 0: int(row['bid_volume_3']) if row['bid_volume_3'] else 0
        }
        #注意ask_volume 总是负数
        sell_orders = {
            int(row['ask_price_1']) if row['ask_price_1'] else 0: -int(row['ask_volume_1']) if row['ask_volume_1'] else 0,
            int(row['ask_price_2']) if row['ask_price_2'] else 0: -int(row['ask_volume_2']) if row['ask_volume_2'] else 0,
            int(row['ask_price_3']) if row['ask_price_3'] else 0: -int(row['ask_volume_3']) if row['ask_volume_3'] else 0
        }
        
        # Create OrderDepth object
        order_depth = OrderDepth()
        order_depth.buy_orders = buy_orders
        order_depth.sell_orders = sell_orders
        
        order_depths[product] = order_depth

        # Assuming initial position is 0, adjust as needed
        position[product] = 0

        # Create a TradingState object for each timestamp
        trading_state = TradingState(
            traderData={},  # Hypothetical traderData
            timestamp=timestamp,
            listings={},
            order_depths=order_depths,
            own_trades={},
            market_trades=market_trades,
            position=position,
            observations={}
        )
        
        trading_states.append(trading_state)

    return trading_states

# 读取CSV文件并准备测试数据
csv_data = load_csv('data/result/result1.csv')  # 替换为你的csv文件路径
trading_states = prepare_trading_state(csv_data)

# 创建Trader对象并运行模拟
trader = Trader()
for trading_state in trading_states[:1]:

    result, conversions, traderData = trader.run(trading_state)
    # 打印结果
    print("Result:", result)
    print("Conversions:", conversions)
    print("Trader Data:", traderData)
