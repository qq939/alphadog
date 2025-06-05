import sys
from pathlib import Path
# 新增：将项目根目录添加到Python搜索路径（解决导入问题）
sys.path.append(str(Path(__file__).parent.parent))  # 父目录是AlphaDog/src，父父目录是AlphaDog（项目根）

import requests
import pandas as pd
import time
import yaml
from typing import List
from tenacity import retry, stop_after_attempt, wait_fixed
import tushare as ts
import os
import logging

# 新增：历史记录写入函数
def log_to_history(message: str):
    """将操作记录追加到 history.md"""
    history_path = Path(r"c:\Users\jimjiang\PycharmProjects\AlphaDog\history.md")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(f"- [{timestamp}] {message}\n")

def get_a_stock_list() -> List[str]:
    """
    从targets.yaml读取需监控的股票代码列表
    Returns:
        List[str]: 股票代码列表（如["sh600519", "sh601398"]）
    """
    targets_path = Path(r"c:\Users\jimjiang\PycharmProjects\AlphaDog\targets.yaml")
    if not targets_path.exists():
        raise FileNotFoundError(f"未找到targets.yaml，请检查路径：{targets_path}")
    
    with open(targets_path, "r", encoding="utf-8") as f:
        stock_mapping = yaml.safe_load(f)  # 格式：{股票名称: 股票代码}
    
    # 提取股票代码（如["sh600519", "sh601398", ...]）
    # 返回格式：[{"name": "股票名称", "code": "股票代码"}, ...]
    return [{"name": name, "code": code} for name, code in stock_mapping.items()]

# 新增：从token.yaml读取Tushare token
def get_tushare_token() -> str:
    """从token.yaml读取Tushare API token"""
    token_path = Path(r"c:\Users\jimjiang\PycharmProjects\AlphaDog\token.yaml")
    if not token_path.exists():
        raise FileNotFoundError(f"未找到token.yaml，请检查路径：{token_path}")
    
    with open(token_path, "r", encoding="utf-8") as f:
        token_config = yaml.safe_load(f)
    
    return token_config.get("tushare_token", "")

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))  # 重试3次，每次间隔1秒
# 替换fetch_sina_history_data函数为Tushare版本
def fetch_tushare_history_data(stock_name: str, stock_code: str, days: int = 800) -> pd.DataFrame:
    """通过Tushare获取前800天历史数据"""
    # 注意：stock_code需转换为Tushare格式（如sh600519 → 600519.SH）
    ts_code = f"{stock_code[2:]}.{stock_code[:2].upper()}"  # 转换为600519.SH格式
    pro = ts.pro_api(get_tushare_token())  # 从配置文件读取token
    
    # 获取日线数据（最多取800条）
    df = pro.daily(ts_code=ts_code, end_date=pd.Timestamp.today().strftime("%Y%m%d"), limit=days)
    if df.empty:
        raise ValueError(f"Tushare未返回数据：{stock_code}")
    
    # 获取MACD指标
    df_macd = pro.macd(ts_code=ts_code, start_date='', end_date=pd.Timestamp.today().strftime("%Y%m%d"))
    
    # 合并数据
    df = pd.merge(df, df_macd, on='trade_date', how='left')
    
    # 字段映射：trade_date→日期，close→收盘价
    df = df[["trade_date", "open", "close", "macd"]].rename(columns={"trade_date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")  # 格式化为YYYY-MM-DD
    return df.assign(code=stock_code, name=stock_name)

# 在update_all_stock_data中调用新函数（替换原fetch_sina_history_data）
def update_all_stock_data():
    """
    定时任务：按接口可承受的限度循环更新所有A股前800天数据
    特性：
        1. 每只股票请求间隔1秒（避免触发接口频率限制）
        2. 单股票请求失败时自动重试3次（每次间隔1秒）
    """

    # 确保数据目录存在
    data_dir = Path(r"c:\Users\jimjiang\PycharmProjects\AlphaDog\data")
    if not data_dir.exists():
        os.makedirs(data_dir)
        log_to_history(f"创建数据目录：{data_dir}")

    stock_codes = get_a_stock_list()
    for idx, stock_info in enumerate(stock_codes, 1):
        try:
            # 传递股票名称和代码到数据获取函数
            df = fetch_tushare_history_data(stock_info["name"], stock_info["code"], days=800)  # 改为Tushare接口
            save_path = data_dir / f"{stock_info['code']}.csv"  # 使用Path对象构建路径
            df.to_csv(save_path, index=False, encoding="utf-8")
            print(f"[{idx}/{len(stock_codes)}] 更新完成：{stock_info['code']} → {save_path}")
            log_to_history(f"数据更新成功：{stock_info['name']}（{stock_info['code']}）（保存至 {save_path}）")  # 新增记录
    
        except Exception as e:
            error_msg = f"数据更新失败（已重试3次）：{stock_info['code']} → {str(e)}"
            print(f"[{idx}/{len(stock_codes)}] {error_msg}")
            error_msg = f"数据更新失败（已重试3次）：{stock_info['name']}（{stock_info['code']}）→ {str(e)}"
            log_to_history(error_msg)  # 新增记录
        
        if idx < len(stock_codes):
            time.sleep(1)

# 配置日志记录
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # 循环更新数据（按接口可承受的限度执行，无固定休眠时间）

    update_all_stock_data()
    print("\n===== 本轮数据更新完成，开始下一轮循环 =====")