import yaml
import pandas as pd
import time
import sys
from pathlib import Path

# Add project root to sys.path to resolve module imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.strategy import HighConfidenceReturnStrategy
from src.data_fetcher import update_all_stock_data
import threading

def load_config() -> float:
    """加载goal.yaml中的目标收益率（保持原逻辑）"""
    goal_path = project_root / "goal.yaml"
    with open(goal_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return float(config["target_return"].strip("%")) / 100

def run_strategy():
    """运行策略并输出结果（保持原逻辑）"""
    target = load_config()
    # 遍历所有股票数据文件
    all_trades = []
    data_dir = project_root / "data"
    for path in data_dir.glob("*.csv"):
        stock_data = pd.read_csv(path)
        strategy = HighConfidenceReturnStrategy(target)
        all_trades.extend(strategy.calculate_signals(stock_data))
    
    # 按实际收益率（扣除成本）降序排序
    def convert_percentage_to_decimal(percentage):
        if isinstance(percentage, str) and percentage.endswith('%'):
            return float(percentage.rstrip('%')) / 100
        return float(percentage)
    sorted_trades = sorted(all_trades, key=lambda x: convert_percentage_to_decimal(x['actual_return']), reverse=True)
    top_28 = sorted_trades[:28]  # 取前28只股票

    # 合并循环：同时处理终端显示和文件写入
    good_stocks_path = project_root / "good_stocks.txt"
    with open(good_stocks_path, "w", encoding="utf-8") as f:
        f.write("股票名称 | 股票代码 | 买入价 | 卖出价 | 预测模型 | 实际收益率（扣除成本）\n")
        f.write("-"*60 + "\n")
        
        print("\n===== 当日实际收益率（扣除成本）最高的28只股票 =====")
        print("股票代码 | 买入价 | 卖出价 | 预测模型 | 实际收益率（扣除成本）")
        print("-"*60)
        
        for trade in top_28:
            # 写入文件
            f.write(f"{trade['stock_name']} | {trade['stock_code']} | {trade['buy_price']:.2f} | {trade['sell_price']} | {trade['model']} | {trade['actual_return']}\n")
            # 终端显示
            print(f"{trade['stock_name']} | {trade['stock_code']} | {trade['buy_price']:.2f} | {trade['sell_price']} | {trade['model']} | {trade['actual_return']}")
        
        print("===============================================")
        
        
if __name__ == "__main__":
    # 启动数据定时更新线程（每20秒）
    update_thread = threading.Thread(target=lambda: update_all_stock_data(), daemon=True)
    update_thread.start()
    
    # 启动策略运行（可根据需求调整运行频率，如每5分钟执行一次策略）
    run_strategy()