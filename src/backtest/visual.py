from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import pandas as pd

from logger import Logger
from utils import _process_data_, extract_sandbox_quadruplets

from matplotlib import font_manager

# 设置字体为 SimHei
font_path = "/Library/Fonts/SimHei.ttf"  
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()


def process_sandbox_data(sandbox_quadruplets, product):
    """将四元组转换为 {timestamp: {attr: value}} 格式，便于后续按时间戳查询。"""
    sandbox_dict = defaultdict(dict)
    for ts, p, attr, value in sandbox_quadruplets:
        if p == product:
            sandbox_dict[ts][attr] = value
    return sandbox_dict


def plot_orderbook_ladder(ax, bids, asks):
    """
    在坐标轴 ax 上绘制“上下对齐”的订单簿：
      - 卖单(asks) 在上方，从上往下排列（红色横条）
      - 买单(bids) 在下方，从下往上排列（绿色横条）
    价格和数量统一放在左侧，并用白底背景来提高可读性。
    """
    ax.clear()

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

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def interactive_orderbook(product: str,
                          df_orderbook: pd.DataFrame,
                          df_trades: pd.DataFrame,
                          sandbox_data: list,
                          selected_attrs: list = None,
                          logger=None):
    """
    增强版交互式订单簿回放
      - sandbox_data: [(timestamp, product, attr, value)]
      - selected_attrs: 要显示的属性列表
    """
    # 1) 整理 sandbox 数据
    sandbox_dict = process_sandbox_data(sandbox_data, product)
    selected_attrs = selected_attrs or list({attr for _, _, attr, _ in sandbox_data})

    # 2) 提取“订单簿 + sandbox”都有的时间戳
    valid_ts = sorted([
        ts for ts in df_orderbook[df_orderbook['product'] == product]['timestamp'].unique()
        if ts in sandbox_dict
    ])

    if not valid_ts:
        print(f"没有找到产品 {product} 的有效数据。")
        return

    # 3) 准备画布
    fig = plt.figure(figsize=(10, 7))
    ax_orderbook = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    ax_info = plt.subplot2grid((4, 1), (3, 0))

    # 4) 时间滑动条
    slider_ax = plt.axes([0.15, 0.02, 0.7, 0.03])
    time_slider = Slider(
        ax=slider_ax,
        label="时间索引",
        valmin=0,
        valmax=len(valid_ts) - 1,
        valinit=0,
        valstep=1
    )

    # 5) 上下页按钮
    prev_ax = plt.axes([0.05, 0.02, 0.08, 0.04])
    prev_btn = Button(prev_ax, '◀ 上页')

    next_ax = plt.axes([0.87, 0.02, 0.08, 0.04])
    next_btn = Button(next_ax, '下页 ▶')

    # 6) 核心更新函数
    def update(_):
        idx = int(time_slider.val)
        current_ts = valid_ts[idx]

        # =============== 订单簿数据 ===============
        row = df_orderbook[
            (df_orderbook['product'] == product) &
            (df_orderbook['timestamp'] == current_ts)
        ]
        if row.empty:
            # 万一该时间戳没有订单簿数据
            ax_orderbook.clear()
            ax_orderbook.text(0.5, 0.5, "无订单簿数据", ha='center', va='center')
            ax_info.clear()
            ax_info.text(0.5, 0.5, "无交易记录/Logger记录", ha='center', va='center')
            fig.canvas.draw_idle()
            return

        row = row.iloc[0]
        bids = [
            (row[f"bid_price_{i}"], row[f"bid_volume_{i}"])
            for i in range(1, 4) if pd.notna(row[f"bid_price_{i}"])
        ]
        asks = [
            (row[f"ask_price_{i}"], row[f"ask_volume_{i}"])
            for i in range(1, 4) if pd.notna(row[f"ask_price_{i}"])
        ]
        bids = sorted(bids, key=lambda x: x[0], reverse=True)
        asks = sorted(asks, key=lambda x: x[0])

        plot_orderbook_ladder(ax_orderbook, bids, asks)
        ax_orderbook.set_title(f"{product} 订单簿 - 时间: {current_ts}", fontsize=12)

        # =============== 信息面板 ===============
        ax_info.clear()
        info_lines = [f"📅 时间戳: {current_ts}"]

        # 1) Sandbox 数据
        if current_ts in sandbox_dict:
            info_lines.append("\n🔍 Sandbox数据:")
            for attr in selected_attrs:
                if attr in sandbox_dict[current_ts]:
                    val_ = sandbox_dict[current_ts][attr]
                    info_lines.append(f"  ▪ {attr}: {val_:.4f}")

        # 2) 交易记录（trade_history）
        trades = df_trades[
            (df_trades['timestamp'] == current_ts) &
            (df_trades['symbol'] == product)
        ]
        if not trades.empty:
            info_lines.append("\n💸 最新成交:")
            for _, trade_row in trades.iterrows():
                # 如果 df_trades 有 buyer/seller，可在此处加上
                buyer = trade_row.get("buyer", "")
                seller = trade_row.get("seller", "")
                info_lines.append(f"  ▫ Buyer:{buyer}, Seller:{seller}, {trade_row['price']} × {trade_row['quantity']}")
        else:
            info_lines.append("\n💸 最新成交: 无")

        # 3) Logger
        if logger is not None:
            logger_summary = logger.store()
            if product in logger_summary:
                # 注意此处改为 current_ts，而非 t
                recs = [rec for rec in logger_summary[product] if rec[0] == current_ts]
                if recs:
                    info_lines.append("\n📝 Logger:")
                    for rec_ts, attr, value in recs:
                        info_lines.append(f"  ▫ {attr}: {value}")
                else:
                    info_lines.append("\n📝 Logger: 无记录")

        ax_info.text(
            0.02, 0.95, "\n".join(info_lines),
            transform=ax_info.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='#F8F9F9', alpha=0.9)
        )
        ax_info.axis('off')

        fig.canvas.draw_idle()

    # 7) 绑定滑条、按钮事件
    time_slider.on_changed(update)

    def on_prev_clicked(_):
        cur = time_slider.val
        if cur > 0:
            time_slider.set_val(cur - 1)

    def on_next_clicked(_):
        cur = time_slider.val
        if cur < len(valid_ts) - 1:
            time_slider.set_val(cur + 1)

    prev_btn.on_clicked(on_prev_clicked)
    next_btn.on_clicked(on_next_clicked)

    # 8) 初始化并展示
    update(0)
    plt.show()


def log_message(timestamp, product, attr, value):
    import datetime
    if isinstance(timestamp, (int, float)):
        ts = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    else:
        ts = str(timestamp)
    print(f"[{ts}] {product} - {attr}: {value}")


def main():
    """示例主函数：解析日志、生成可视化。"""
    # 读取日志文件
    print('Reading log file...')
    log_file = "data/log/tutorial_v2.log"
    with open(log_file, "r", encoding="utf-8") as f:
        log_content = f.read()

    # 提取 Sandbox 日志中的四元组
    print('Extracting sandbox quadruplets...')
    sandbox_quadruplets = extract_sandbox_quadruplets(log_content)

    # 加载订单簿与成交数据
    print('Loading market data and trade history...')
    market_data, trade_history = _process_data_(log_file)

    record = Logger()

    record.record(00,"RAINFOREST_RESIN","test",666)

    # 启动交互式回放
    print('Starting interactive replay...')
    interactive_orderbook(
        product="RAINFOREST_RESIN",
        df_orderbook=market_data,
        df_trades=trade_history,
        sandbox_data=sandbox_quadruplets,
        selected_attrs=["Fair price", "Orderbook imbalance","Expected Bid", "Expected Ask"],
        logger=record
    )


if __name__ == "__main__":
    main()
