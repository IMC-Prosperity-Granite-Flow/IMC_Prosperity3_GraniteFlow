import io
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
def _process_data_(file):
    with open(file, 'r') as file:
        log_content = file.read()
    sections = log_content.split('Sandbox logs:')[1].split('Activities log:')
    sandbox_log =  sections[0].strip()
    activities_log = sections[1].split('Trade History:')[0]
    # sandbox_log_list = [json.loads(line) for line in sandbox_log.split('\n')]
    trade_history =  json.loads(sections[1].split('Trade History:')[1])
    # sandbox_log_df = pd.DataFrame(sandbox_log_list)
    market_data_df = pd.read_csv(io.StringIO(activities_log), sep=";", header=0)
    trade_history_df = pd.json_normalize(trade_history)
    return market_data_df, trade_history_df

def extract_activities_log(file_path: str) -> pd.DataFrame:
    """
    从日志文件中提取 Activities log 部分并转换为 DataFrame。
    日志文件格式示例：
        Sandbox logs:
        ...（Sandbox log 内容）

        Activities log:
        <CSV 格式数据，分隔符为";">

        Trade History:
        ...（Trade log 内容）
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 定位 Activities log 部分和 Trade History 部分的起始位置
    act_idx = content.find("Activities log:")
    trade_idx = content.find("Trade History:")

    if act_idx == -1:
        print("未找到 'Activities log:' 部分")
        return pd.DataFrame()

    start = act_idx + len("Activities log:")
    if trade_idx == -1:
        csv_text = content[start:]
    else:
        csv_text = content[start:trade_idx]

    csv_text = csv_text.strip()
    if not csv_text:
        print("提取到的 Activities log 文本为空")
        return pd.DataFrame()

    csv_io = StringIO(csv_text)
    try:
        df = pd.read_csv(csv_io, sep=";")
    except Exception as e:
        print("解析 CSV 时发生错误:", e)
        return pd.DataFrame()
    return df


import pandas as pd


def print_orderbook_and_trades(df_orderbook: pd.DataFrame, df_trades: pd.DataFrame, only_with_trade: bool = False):
    """
    逐行打印每个时间戳、产品的订单簿信息和对应的交易记录。

    参数:
      - df_orderbook: 包含订单簿信息的 DataFrame，列如：
            day, timestamp, product,
            bid_price_1, bid_volume_1, bid_price_2, bid_volume_2, bid_price_3, bid_volume_3,
            ask_price_1, ask_volume_1, ask_price_2, ask_volume_2, ask_price_3, ask_volume_3
      - df_trades: 包含交易记录的 DataFrame，列如：
            timestamp, symbol, price, quantity, buyer, seller, ...
      - only_with_trade: 如果为 True，则只打印有交易发生的订单簿快照；否则全部打印。

    说明:
      - 假设 df_trades 中的 "symbol" 对应 df_orderbook 中的 "product"，
        且两个 DataFrame 的 "timestamp" 对应同一时间戳。
    """
    # 按 day, timestamp, product 排序
    df_orderbook_sorted = df_orderbook.sort_values(by=["day", "timestamp", "product"])

    for idx, row in df_orderbook_sorted.iterrows():
        day = row["day"]
        ts = row["timestamp"]
        product = row["product"]

        # 在 df_trades 中筛选对应记录
        trades_this_ts = df_trades[
            (df_trades["timestamp"] == ts) & (df_trades["symbol"] == product)
            ]

        trades_this_ts = trades_this_ts[
            (trades_this_ts["buyer"] == 'SUBMISSION') | (trades_this_ts["seller"] == 'SUBMISSION')
            ]

        # 如果设置了 only_with_trade 且当前没有交易记录，则跳过打印
        if only_with_trade and trades_this_ts.empty:
            continue

        # ============ 打印订单簿 ============
        # 假设有 3 档买单和卖单
        bids = [
            (row["bid_price_1"], row["bid_volume_1"]),
            (row["bid_price_2"], row["bid_volume_2"]),
            (row["bid_price_3"], row["bid_volume_3"]),
        ]
        asks = [
            (row["ask_price_1"], row["ask_volume_1"]),
            (row["ask_price_2"], row["ask_volume_2"]),
            (row["ask_price_3"], row["ask_volume_3"]),
        ]

        # 买单从高到低排序；卖单从低到高排序
        bids_sorted = sorted(bids, key=lambda x: x[0], reverse=True)
        asks_sorted = sorted(asks, key=lambda x: x[0])

        print("=" * 70)
        print(f"Day: {day}, Timestamp: {ts}, Product: {product}")
        print("--- BIDS (买单) [从高到低] ---")
        for price, volume in bids_sorted:
            print(f"   Price: {price}, Volume: {volume}")
        print("--- ASKS (卖单) [从低到高] ---")
        for price, volume in asks_sorted:
            print(f"   Price: {price}, Volume: {volume}")

        # ============ 打印交易信息 ============
        if trades_this_ts.empty:
            print("\n-- Trades / Orders --")
            print("   No trades at this timestamp/product.")
        else:
            print("\n-- Trades / Orders --")
            for t_idx, trade_row in trades_this_ts.iterrows():
                buyer = trade_row.get("buyer", "")
                seller = trade_row.get("seller", "")
                price = trade_row.get("price", None)
                qty = trade_row.get("quantity", None)
                print(f"   Buyer: {buyer}, Seller: {seller}, Price: {price}, Qty: {qty}")
        print("=" * 70)
        print()  # 空行分隔

if __name__ == "__main__":
    # 替换成实际的日志文件路径
    file_path = "tutorial_v1.log"
    market_data, trade_history = _process_data_(file_path)


    if not market_data.empty:
        print_orderbook_and_trades(market_data,trade_history,True)
