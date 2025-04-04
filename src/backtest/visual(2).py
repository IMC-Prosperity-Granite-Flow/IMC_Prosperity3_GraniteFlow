import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import pandas as pd

# 假设从你的项目中导入以下函数/类：
from logger import Logger
from utils import _process_data_

from matplotlib import font_manager

# 设置字体为 SimHei（或系统中的其他中文字体）
font_path = "/Library/Fonts/SimHei.ttf"  
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

def plot_orderbook_ladder(ax, bids, asks):
    """
    在坐标轴 ax 上绘制“上下对齐”的订单簿：
      - 卖单(asks) 在上方，从上往下排列（红色横条）
      - 买单(bids) 在下方，从下往上排列（绿色横条）
    价格和数量统一放在左侧，并用白底背景来提高可读性。
    """
    ax.clear()

    # 可调整的参数
    scale = 0.3         # 横向缩放：volume -> 宽度
    rect_height = 0.8   # 每档买/卖单的高度

    num_asks = len(asks)
    num_bids = len(bids)

    # =============== 绘制卖单 (asks) ===============
    # 最优卖单(asks[0])放在 y=0，下一档 y=1，依次往下
    for i, (price, volume) in enumerate(asks):
        y = i
        width = volume * scale
        ax.barh(y, width, left=0, height=rect_height,
                color='red', alpha=0.6, edgecolor='black', linewidth=1)
        # 在横条左侧放置文字：价格和数量
        x_text = -0.05 - 0.05 * scale * max(volume, 1.0)
        ax.text(x_text, y + rect_height / 2,
                f"{price:.2f} ({volume})",
                ha="right", va="center",
                fontsize=9, fontweight="bold",
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    # =============== 绘制买单 (bids) ===============
    # 最优买单(bids[0])放在 y=-1，下一档 y=-2，依次往下
    for i, (price, volume) in enumerate(bids):
        y = -(i + 1)
        width = volume * scale
        ax.barh(y, width, left=0, height=rect_height,
                color='green', alpha=0.6, edgecolor='black', linewidth=1)
        # 同样在左侧放置文字
        x_text = -0.05 - 0.05 * scale * max(volume, 1.0)
        ax.text(x_text, y + rect_height / 2,
                f"{price:.2f} ({volume})",
                ha="right", va="center",
                fontsize=9, fontweight="bold",
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    # 在 y=-0.5 处画分隔线
    ax.axhline(y=-0.5, color='black', linewidth=1)

    # 确定 x 范围：从 0 到 max_vol * scale * 1.2
    max_vol = max(
        max((v for _, v in asks), default=0),
        max((v for _, v in bids), default=0)
    )
    ax.set_xlim(0, max_vol * scale * 1.2)

    # 确定 y 范围：上方到 (num_asks - 1) + 0.5，下方到 -num_bids - 0.5
    top_y = (num_asks - 1) + 0.5 if num_asks > 0 else 0.5
    bottom_y = -num_bids - 0.5
    ax.set_ylim(bottom_y, top_y)

    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

def interactive_orderbook(product: str, df_orderbook: pd.DataFrame, df_trades: pd.DataFrame, logger=None):
    """
    使用 matplotlib.widgets.Slider + Button 实现交互式订单簿回放。
    - 卖单在上，买单在下
    - 可拖动滑条选择时间戳
    - “Prev” / “Next” 按钮可前后移动时间索引
    - 下方显示交易记录与 Logger 信息
    """
    # 提取该产品所有时间戳
    timestamps = sorted(df_orderbook[df_orderbook['product'] == product]['timestamp'].unique())
    if not timestamps:
        print(f"没有找到产品 {product} 的数据。")
        return

    fig = plt.figure(figsize=(9, 6))
    ax_orderbook = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    ax_info = plt.subplot2grid((4, 1), (3, 0))

    # 底部 Slider
    slider_ax = plt.axes([0.15, 0.01, 0.7, 0.04])
    time_slider = Slider(
        ax=slider_ax,
        label="Time Index",
        valmin=0,
        valmax=len(timestamps) - 1,
        valinit=0,
        valstep=1
    )

    # 左下角 “Prev” 按钮
    button_prev_ax = plt.axes([0.02, 0.01, 0.06, 0.04])
    button_prev = Button(button_prev_ax, 'Prev')

    # 右下角 “Next” 按钮
    button_next_ax = plt.axes([0.88, 0.01, 0.06, 0.04])
    button_next = Button(button_next_ax, 'Next')

    def update(val):
        # 当前滑条的索引
        time_index = int(time_slider.val)
        t = timestamps[time_index]

        # 筛选订单簿数据
        row = df_orderbook[(df_orderbook['product'] == product) & (df_orderbook['timestamp'] == t)]
        if row.empty:
            ax_orderbook.clear()
            ax_orderbook.text(0.5, 0.5, f"时间 {t} 下无订单簿数据", ha='center', va='center')
            ax_info.clear()
            ax_info.text(0.5, 0.5, "无交易记录/Logger 记录", ha='center', va='center')
            fig.canvas.draw_idle()
            return
        row = row.iloc[0]

        # 提取买卖档位 (假设 3 档)
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
        # 去除 NaN
        bids = [(p, v) for p, v in bids if pd.notna(p) and pd.notna(v)]
        asks = [(p, v) for p, v in asks if pd.notna(p) and pd.notna(v)]
        # 排序
        bids_sorted = sorted(bids, key=lambda x: x[0], reverse=True)
        asks_sorted = sorted(asks, key=lambda x: x[0])

        # 绘制上下对齐的订单簿
        plot_orderbook_ladder(ax_orderbook, bids_sorted, asks_sorted)
        ax_orderbook.set_title(f"{product} Order Book at time {t}", fontsize=12)

        # 显示交易记录 & Logger
        ax_info.clear()
        info_lines = [f"时间: {t}"]
        # 交易记录
        trades = df_trades[(df_trades["timestamp"] == t) & (df_trades["symbol"] == product)]
        if trades.empty:
            info_lines.append("无交易记录。")
        else:
            for _, trade in trades.iterrows():
                buyer = trade.get("buyer", "")
                seller = trade.get("seller", "")
                price = trade.get("price", "")
                qty = trade.get("quantity", "")
                info_lines.append(f"Buyer: {buyer}, Seller: {seller}, Price: {price}, Qty: {qty}")
        # Logger
        if logger is not None:
            logger_summary = logger.store()
            if product in logger_summary:
                recs = [rec for rec in logger_summary[product] if rec[0] == t]
                if recs:
                    for rec_ts, attr, value in recs:
                        info_lines.append(f"Logger - {attr}: {value}")
                else:
                    info_lines.append("无 Logger 记录。")
        info_text = "\n".join(info_lines)
        ax_info.text(0.01, 0.5, info_text, transform=ax_info.transAxes,
                     fontsize=10, verticalalignment='center')
        ax_info.axis('off')

        fig.canvas.draw_idle()

    # Slider 回调
    time_slider.on_changed(update)

    # 按钮点击回调
    def on_prev_clicked(event):
        current_val = time_slider.val
        if current_val > 0:
            time_slider.set_val(current_val - 1)

    def on_next_clicked(event):
        current_val = time_slider.val
        if current_val < len(timestamps) - 1:
            time_slider.set_val(current_val + 1)

    button_prev.on_clicked(on_prev_clicked)
    button_next.on_clicked(on_next_clicked)

    # 初始化
    update(0)
    plt.show()

def main():
    """
    主函数示例：
    1. 解析日志文件 -> (market_data, trade_history)
    2. 创建 Logger
    3. 启动交互式订单簿回放
    """
    # 假设此文件存在，并且能正确返回两个 DataFrame
    file_path = "tutorial_v1.log"
    market_data, trade_history = _process_data_(file_path)

    # 创建 Logger 并记录一个示例
    record = Logger()
    record.record(162100, "RAINFOREST_RESIN", "Test_attr", 999)

    # 启动交互式回放：上下对齐订单簿 + 滑条 + Prev/Next 按钮
    interactive_orderbook("RAINFOREST_RESIN", market_data, trade_history, record)

if __name__ == "__main__":
    main()
