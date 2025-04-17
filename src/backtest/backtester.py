from prosperity3bt.runner import run_backtest
from prosperity3bt.file_reader import PackageResourcesReader
from prosperity3bt.file_reader import FileSystemReader
from prosperity3bt.models import TradeMatchingMode
from collections import defaultdict
from pathlib import Path




# 运行回测
class Backtester:
       def __init__(self, trader):
              self.trader = trader()
              pass
       

       def merge_product_pnls(self, pnl_dicts: list[dict[str, float]]) -> dict[str, float]:
              merged = defaultdict(float)
              for d in pnl_dicts:
                     for product, pnl in d.items():
                            merged[product] += pnl
              return dict(merged)

       def get_final_pnl(self, result) -> int:
              last_timestamp = result.activity_logs[-1].timestamp
              return sum(
                     row.columns[-1]
                     for row in reversed(result.activity_logs)
                     if row.timestamp == last_timestamp
              )
       
       def extract_product_pnl(self, result) -> dict[str, float]:
              last_timestamp = result.activity_logs[-1].timestamp
              product_pnl = {}

              for row in reversed(result.activity_logs):
                     if row.timestamp != last_timestamp:
                            break

                     product = row.columns[2]
                     profit = row.columns[-1]
                     product_pnl[product] = profit

              return product_pnl
       
       def backtest(self, round_num: int, day_num: int = None, data_path: str = None) -> tuple[float, dict[str, float]]:
              """
              运行回测
              :param round_num: 回测轮次
              :param day_num: 单天或多天（默认 [-2, -1, 0]）
              :return: 总 PnL: float, 产品 PnL: dict[str, float]
              """
              if day_num is None:
                     day_nums = [round_num - 2, round_num - 1, round_num]
              else:
                     day_nums = [day_num]
              if not data_path:
                     file_reader = PackageResourcesReader()
              else:
                     file_reader = FileSystemReader(Path(data_path))
              final_pnl = 0.0
              product_pnl_list = []

              for day in day_nums:
                     result = run_backtest(
                            self.trader,
                            file_reader,
                            round_num=round_num,
                            day_num=day,
                            print_output=False,
                            trade_matching_mode=TradeMatchingMode.all,
                            no_names=True,
                            show_progress_bar=True,
                     )

                     product_pnl_i = self.extract_product_pnl(result)
                     product_pnl_list.append(product_pnl_i)
                     final_pnl += self.get_final_pnl(result)

              product_pnl = self.merge_product_pnls(product_pnl_list)

              print("Product PnL: ", product_pnl)
              print("Final PnL: ", final_pnl)
              return final_pnl, product_pnl
       
def test():
       trader = Trader()
       backterster = Backtester(trader)
       final_pnl, product_pnl = backterster.backtest(round_num=1)
if __name__ == '__main__':
       test()