import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display


# 假设 df_orderbook, df_trades, logger 已经通过 _process_data_ 等方法加载好了

def plot_orderbook_trades(product: str, timestamp: int, df_orderbook: pd.DataFrame, df_trades: pd.DataFrame,
                          logger=None):
    # 筛选出当前产品和时间戳对应的订单簿数据
    orderbook_row = df_orderbook[(df_orderbook['product'] == product) & (df_orderbook['timestamp'] == timestamp)]
    trade_rows = df_trades[(df_trades['symbol'] == product) & (df_trades['timestamp'] == timestamp)]

    if orderbook_row.empty:
        print(f"产品 {product} 在时间戳 {timestamp} 没有订单簿数据。")
        return

    row = orderbook_row.iloc[0]

    # 获取买单和卖单数据（假设各有3档）
    bids = [
        (row["bid_price_1"], row["bid_volume_1"]),
        (row["bid_price_2"], row["bid_volume_2"]),
        (row["bid_price_3"], row["bid_volume_3"])
    ]
    asks = [
        (row["ask_price_1"], row["ask_volume_1"]),
        (row["ask_price_2"], row["ask_volume_2"]),
        (row["ask_price_3"], row["ask_volume_3"])
    ]

    # 买单按价格从高到低，卖单按价格从低到高排序
    bids_sorted = sorted(bids, key=lambda x: x[0], reverse=True)
    asks_sorted = sorted(asks, key=lambda x: x[0])

    bid_prices, bid_volumes = zip(*bids_sorted)
    ask_prices, ask_volumes = zip(*asks_sorted)

    # 创建3个子图：买单、卖单和交易记录
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # 绘制买单：横坐标为价格，纵坐标为挂单量
    axs[0].bar(range(len(bid_prices)), bid_volumes, tick_label=bid_prices)
    axs[0].set_title(f"产品 {product} - 买单（时间戳 {timestamp}）")
    axs[0].set_ylabel("Volume")
    axs[0].set_xlabel("Price")

    # 绘制卖单
    axs[1].bar(range(len(ask_prices)), ask_volumes, tick_label=ask_prices)
    axs[1].set_title(f"产品 {product} - 卖单（时间戳 {timestamp}）")
    axs[1].set_ylabel("Volume")
    axs[1].set_xlabel("Price")

    # 显示交易记录（如果有则逐条显示，没有则提示无交易记录）
    axs[2].axis('off')
    trade_text = "交易记录:\n"
    if trade_rows.empty:
        trade_text += "无交易记录。"
    else:
        for idx, trade in trade_rows.iterrows():
            trade_text += f"Buyer: {trade.get('buyer', '')}, Seller: {trade.get('seller', '')}, Price: {trade.get('price', '')}, Qty: {trade.get('quantity', '')}\n"
    axs[2].text(0.01, 0.5, trade_text, fontsize=12)
    axs[2].set_title(f"产品 {product} - 交易记录（时间戳 {timestamp}）")

    plt.tight_layout()
    plt.show()

    # 如果有 Logger 对象，则打印出对应时间戳的记录
    if logger is not None:
        logger_summary = logger.store()
        if product in logger_summary:
            relevant_logs = [rec for rec in logger_summary[product] if rec[0] == timestamp]
            print("Logger记录:")
            if relevant_logs:
                for rec in relevant_logs:
                    # 假设 Logger 记录的格式为 (timestamp, attr, value)
                    print(f"   {rec[1]}: {rec[2]}")
            else:
                print("   无Logger记录。")


def interactive_orderbook(product: str, df_orderbook: pd.DataFrame, df_trades: pd.DataFrame, logger=None):
    # 提取该产品所有的时间戳
    timestamps = sorted(df_orderbook[df_orderbook['product'] == product]['timestamp'].unique())
    if not timestamps:
        print(f"没有找到产品 {product} 的数据。")
        return

    slider = widgets.IntSlider(
        value=timestamps[0],
        min=min(timestamps),
        max=max(timestamps),
        step=1,
        description='时间戳:'
    )

    # 定义一个更新函数，当滑块变化时重新绘图
    def update(timestamp):
        plot_orderbook_trades(product, timestamp, df_orderbook, df_trades, logger)

    ui = widgets.HBox([slider])
    out = widgets.interactive_output(update, {'timestamp': slider})
    display(ui, out)

# 示例用法：
# 假设通过 _process_data_ 加载数据，得到 market_data 和 trade_history，并创建 Logger 实例 record
# market_data, trade_history = _process_data_(file_path)
# record = Logger()
# record.record(199600, "RAINFOREST_RESIN", "Test_attr", 1024)

# 例如：对产品 "RAINFOREST_RESIN" 进行交互式回放
# interactive_orderbook("RAINFOREST_RESIN", market_data, trade_history, record)

if __name__ == "__main__":
    # 替换成实际的日志文件路径
    file_path = "tutorial_v1.log"
    market_data, trade_history = _process_data_(file_path)
    record = Logger()

    record.record(199600,"RAINFOREST_RESIN","Test_attr",1024)



    if not market_data.empty:
        print_orderbook_and_trades(market_data,trade_history,True,record)