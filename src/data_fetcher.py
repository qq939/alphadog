import sys
from pathlib import Path
# 新增：将项目根目录添加到Python搜索路径（解决导入问题）
project_root = Path(__file__).parent.parent  # 父目录是AphaDogTushare/src，父父目录是AphaDogTushare（项目根）
sys.path.append(str(project_root))

import requests
import pandas as pd
import numpy as np
import time
import yaml
from typing import List
from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential, retry_if_exception_type
import tushare as ts
import os
import logging
import traceback

# 新增：历史记录写入函数
def log_to_history(message: str):
    """将操作记录追加到 history.md"""
    history_path = project_root / "history.md"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(f"- [{timestamp}] {message}\n")

def get_a_stock_list() -> List[str]:
    """
    从targets.yaml读取需监控的股票代码列表
    Returns:
        List[str]: 股票代码列表（如["sh600519", "sh601398"]）
    """
    targets_path = project_root / "targets.yaml"
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
    token_path = project_root / "token.yaml"
    if not token_path.exists():
        raise FileNotFoundError(f"未找到token.yaml，请检查路径：{token_path}")
    
    with open(token_path, "r", encoding="utf-8") as f:
        token_config = yaml.safe_load(f)
    
    return token_config.get("tushare_token", "")

@retry(
    stop=stop_after_attempt(5),  # 增加到5次重试
    wait=wait_exponential(multiplier=1, min=2, max=60),  # 指数退避策略：等待时间指数增长，最小2秒，最大60秒
    retry=retry_if_exception_type((ValueError, ConnectionError, TimeoutError)),  # 只对特定异常重试
    reraise=True  # 重试失败后抛出原始异常
)
# 替换fetch_sina_history_data函数为Tushare版本
def fetch_tushare_history_data(stock_name: str, stock_code: str, days: int = 800) -> pd.DataFrame:
    """通过Tushare获取前800天历史数据"""
    try:
        # 注意：stock_code需转换为Tushare格式（如sh600519 → 600519.SH）
        ts_code = f"{stock_code[2:]}.{stock_code[:2].upper()}"  # 转换为600519.SH格式
        pro = ts.pro_api(get_tushare_token())  # 从配置文件读取token
        
        # 获取日线数据（最多取800条）
        logging.info(f"正在获取{stock_name}({stock_code})的日线数据...")
        df = pro.daily(ts_code=ts_code, end_date=pd.Timestamp.today().strftime("%Y%m%d"), limit=days)
        if df.empty:
            error_msg = f"Tushare未返回数据：{stock_code}"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        # 本地计算MACD指标，不再依赖Tushare API
        logging.info(f"正在计算{stock_name}({stock_code})的MACD指标...")
        try:
            # 重命名列并选择需要的列
            df = df[["trade_date", "open", "close"]].rename(columns={"trade_date": "date"})
            
            # 按日期降序排序，确保时间顺序正确
            df = df.sort_values(by="date", ascending=False).reset_index(drop=True)
            
            # 计算EMA和MACD
            close_prices = df["close"].values
            # 计算12日EMA
            ema12 = np.zeros_like(close_prices)
            ema12[0] = close_prices[0]  # 初始值
            for i in range(1, len(close_prices)):
                ema12[i] = (close_prices[i] * 2 / 13) + (ema12[i-1] * 11 / 13)
            
            # 计算26日EMA
            ema26 = np.zeros_like(close_prices)
            ema26[0] = close_prices[0]  # 初始值
            for i in range(1, len(close_prices)):
                ema26[i] = (close_prices[i] * 2 / 27) + (ema26[i-1] * 25 / 27)
            
            # 计算DIF (MACD Line): 12日EMA - 26日EMA
            dif = ema12 - ema26
            
            # 计算DEA (Signal Line): 9日EMA of DIF
            dea = np.zeros_like(dif)
            dea[0] = dif[0]  # 初始值
            for i in range(1, len(dif)):
                dea[i] = (dif[i] * 2 / 10) + (dea[i-1] * 8 / 10)
            
            # 计算MACD柱状图 (Histogram): (DIF - DEA) * 2
            macd = (dif - dea) * 2
            
            # 计算RSI指标 (14日)
            delta = np.zeros_like(close_prices)
            delta[1:] = close_prices[1:] - close_prices[:-1]  # 计算价格变化
            
            gain = np.copy(delta)
            loss = np.copy(delta)
            gain[gain < 0] = 0  # 只保留上涨
            loss[loss > 0] = 0  # 只保留下跌
            loss = abs(loss)    # 转为正值
            
            # 计算平均上涨和下跌
            avg_gain = np.zeros_like(gain)
            avg_loss = np.zeros_like(loss)
            
            # 初始平均值（前14天）
            window_size = 14
            if len(gain) >= window_size:
                avg_gain[window_size-1] = np.mean(gain[:window_size])
                avg_loss[window_size-1] = np.mean(loss[:window_size])
                
                # 计算后续值（使用Wilder平滑方法）
                for i in range(window_size, len(gain)):
                    avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
                    avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
                
                # 计算相对强度（RS）和RSI
                rs = np.zeros_like(avg_gain)
                rsi = np.zeros_like(avg_gain)
                
                for i in range(window_size-1, len(avg_gain)):
                    if avg_loss[i] == 0:
                        rs[i] = 100  # 避免除以零
                        rsi[i] = 100
                    else:
                        rs[i] = avg_gain[i] / avg_loss[i]
                        rsi[i] = 100 - (100 / (1 + rs[i]))
            else:
                rsi = np.zeros_like(gain)
            
            # 计算KDJ指标
            # 首先计算最高价和最低价（这里使用收盘价代替，实际应使用完整OHLC数据）
            high_prices = df["close"].rolling(window=9).max().values
            low_prices = df["close"].rolling(window=9).min().values
            
            # 计算RSV
            rsv = np.zeros_like(close_prices)
            for i in range(8, len(close_prices)):
                if high_prices[i] == low_prices[i]:
                    rsv[i] = 50  # 避免除以零
                else:
                    rsv[i] = 100 * (close_prices[i] - low_prices[i]) / (high_prices[i] - low_prices[i])
            
            # 计算K值（默认参数：3）
            k_value = np.zeros_like(rsv)
            k_value[8] = 50  # 初始值
            for i in range(9, len(rsv)):
                k_value[i] = (2/3) * k_value[i-1] + (1/3) * rsv[i]
            
            # 计算D值（默认参数：3）
            d_value = np.zeros_like(k_value)
            d_value[8] = 50  # 初始值
            for i in range(9, len(k_value)):
                d_value[i] = (2/3) * d_value[i-1] + (1/3) * k_value[i]
            
            # 计算J值
            j_value = np.zeros_like(d_value)
            for i in range(8, len(d_value)):
                j_value[i] = 3 * k_value[i] - 2 * d_value[i]
            
            # 将计算结果添加到DataFrame
            df["macd"] = macd
            df["dif"] = dif
            df["dea"] = dea
            df["rsi"] = rsi
            df["k"] = k_value
            df["d"] = d_value
            df["j"] = j_value
            
            logging.info(f"成功计算{stock_name}({stock_code})的MACD、RSI和KDJ指标")
        except Exception as e:
            # 如果计算指标失败，记录错误但继续处理，使用基本数据
            logging.warning(f"计算{stock_code}的技术指标失败，将使用基本数据：{str(e)}")
            # 添加空的指标列
            df["macd"] = None
            df["dif"] = None
            df["dea"] = None
            df["rsi"] = None
            df["k"] = None
            df["d"] = None
            df["j"] = None
        
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")  # 格式化为YYYY-MM-DD
        return df.assign(code=stock_code, name=stock_name)
    except Exception as e:
        # 记录详细错误信息
        error_detail = traceback.format_exc()
        logging.error(f"获取{stock_name}({stock_code})数据时发生错误:\n{error_detail}")
        # 重新抛出异常，让重试机制处理
        raise

# 在update_all_stock_data中调用新函数（替换原fetch_sina_history_data）
def update_all_stock_data():
    """
    定时任务：按接口可承受的限度循环更新所有A股前800天数据
    特性：
        1. 每只股票请求间隔2秒（避免触发接口频率限制）
        2. 单股票请求失败时自动重试5次（使用指数退避策略）
    """

    # 确保数据目录存在
    data_dir = Path(r"c:\Users\jimjiang\PycharmProjects\AphaDogTushare\data")
    if not data_dir.exists():
        os.makedirs(data_dir)
        log_to_history(f"创建数据目录：{data_dir}")

    stock_codes = get_a_stock_list()
    for idx, stock_info in enumerate(stock_codes, 1):
        try:
            # 传递股票名称和代码到数据获取函数
            logging.info(f"[{idx}/{len(stock_codes)}] 开始更新：{stock_info['name']}({stock_info['code']})")
            df = fetch_tushare_history_data(stock_info["name"], stock_info["code"], days=800)  # 改为Tushare接口
            save_path = data_dir / f"{stock_info['code']}.csv"  # 使用Path对象构建路径
            df.to_csv(save_path, index=False, encoding="utf-8")
            success_msg = f"[{idx}/{len(stock_codes)}] 更新完成：{stock_info['code']} → {save_path}"
            print(success_msg)
            log_to_history(f"数据更新成功：{stock_info['name']}（{stock_info['code']}）（保存至 {save_path}）")  # 新增记录
    
        except Exception as e:
            error_msg = f"数据更新失败（已重试5次）：{stock_info['code']} → {str(e)}"
            print(f"[{idx}/{len(stock_codes)}] {error_msg}")
            error_msg = f"数据更新失败（已重试5次）：{stock_info['name']}（{stock_info['code']}）→ {str(e)}"
            log_to_history(error_msg)  # 新增记录
            logging.error(f"更新{stock_info['name']}({stock_info['code']})数据失败: {traceback.format_exc()}")
        
        if idx < len(stock_codes):
            # 增加请求间隔到2秒，减少API限制风险
            time.sleep(2)

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,  # 提高日志级别，记录更多信息
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(r"c:\Users\jimjiang\PycharmProjects\AphaDogTushare\tushare_api.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

if __name__ == "__main__":
    # 循环更新数据（按接口可承受的限度执行，无固定休眠时间）

    update_all_stock_data()
    print("\n===== 本轮数据更新完成，开始下一轮循环 =====")