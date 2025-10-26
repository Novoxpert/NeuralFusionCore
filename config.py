
from dataclasses import dataclass, field
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def rel_path(*paths):
    """Helper: build an absolute path relative to BASE_DIR"""
    return os.path.join(BASE_DIR, *paths)
@dataclass
class Paths:
    data_dir: str = rel_path("data")
    raw_news_dir: str = rel_path("data/raw_news")
    ohlcv_dir: str = rel_path("data/ohlcv")
    data_realtime_dir: str = rel_path("data_realtime")
    ohlcv_realtime_dir: str = rel_path("data_realtime/ohlcv")
    news_realtime_dir: str = rel_path("data_realtime/news")
    processed_dir: str = rel_path("data/processed")
    processed_backtesting_dir: str = rel_path("data/processed/backtesting")
    outputs_dir: str = rel_path("data/outputs")
    news_pickle: str = rel_path("data/processed/news_embeddings.pkl")
    news_no_vec_pickle: str = rel_path("data/processed/news_no_vec_embeddings.pkl")
    merged_parquet: str = rel_path("data/processed/merged_3m.parquet")
    normalizer_pkl: str = rel_path("data/processed/normalizer.pkl")
    normalizer_backtesting_pkl: str = rel_path("data/processed/backtesting/normalizer.pkl")
    splits_json: str = rel_path("data/processed/splits.json")
    weights_pt: str = rel_path("data/outputs/model_weights.pt")

@dataclass
class NewsCfg:
    model_path: str = rel_path("models/bigbird-2048-final/bigbird-2048-final")
    batch_size: int = 2
    max_len: int = 2048
    pooling: str = "mean"
    rule: str = "3min"
    no_news_token: str = "no news at this time"

@dataclass
class MarketCfg:
 
    symbols_usdt: list = field(default_factory=lambda: [
        'BINANCE:BTCUSDT','BINANCE:ETHUSDT','BINANCE:BNBUSDT','BINANCE:SOLUSDT','BINANCE:XRPUSDT', 'NASDAQ:AAPL','NASDAQ:MSFT','NASDAQ:NVDA', 'NASDAQ:GooGL', 'NASDAQ:META'
        ,'NASDAQ:AMZN', 'NYSE:CRM', 'NASDAQ:COST', 'NYSE:NOW', 'NYSE:ORCL','NASDAQ:AVGO', 'NYSE:TSM', 'NASDAQ:ASML', 'NASDAQ:QCOM', 'NYSE:IBM', 'NYSE:JPM', 'NYSE:BAC'
        ,'NYSE:WFC', 'NYSE:MS', 'NYSE:BRK.A', 'NYSE:XOM', 'NYSE:CVX', 'NYSE:KO', 'NASDAQ:PEP', 'NYSE:MCD', 'SP:SPX' , 'TVC:IXIC', 'XETR:DAX', 'TVC:NI225',
        'TVC:UKX', 'FX:EURUSD', 'FX_IDC:USDJPY', 'OANDA:XAUUSD', 'BIST:XAGUSD1!', 'CBOE:VIX'      
    ])
    timeframe: str = "1m"
    start_date: str = "2025-02-26"
    end_date: str = "2025-03-01"

@dataclass
class FeatureCfg:
    agg_cols: dict = field(default_factory=lambda: {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum"
    })
    seq_len: int = 10
    horizon_steps: int = 5
    news_rule: str = "3min"
    resample: int = 3

@dataclass
class TrainCfg:
    batch_size: int = 32
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    hidden_dim: int = 64
    lr: float = 1e-4
    epochs: int = 50
    patience: int = 5
    device: str = "cuda"
    top_k: int = 30
    gross: float = 1.0
    train_config: dict = field(default_factory=lambda: {
        'task_name': 'classification',
        'seq_len': 30,
        'enc_in': 120,
        'd_model': 64,
        'c_out': 2,  # len(selected_f)
        'd_ff': 128,
        'num_kernels': 3,
        'dropout': 0.1,
        'e_layers': 2,
        'top_k': 3,
        'num_class': 2,
        'label_len': 30,
        'pred_len': 1,
        'embed': 'timeF',
        'freq': 't'
    })

@dataclass
class LossCfg:
    use_cov: bool = True
    lambda_div: float = 0.01
    lambda_net: float = 0.0
    lambda_turnover: float = 0.0

@dataclass
class BacktestCfg:
    stride: int = 80
    stoploss: int = 5
    takeprofit: int = 5

@dataclass
class MongoCfg:
    MONGO_HOST : str = '5.75.195.167'
    MONGO_PORT : int = 27017
    MONGO_DB : str = 'novoxpert'
    MONGO_COLLECTION : str ='news'
    MONGO_USER : str = 'StreamNode1'
    MONGO_PASS : str = "‌Block2#InTerCoM!"
    MONGO_AUTHSOURCE: str = 'novoxpert' 

@dataclass
class ClickhouseCfg:
    CH_HOST : str = '65.109.201.152'
    #CH_PORT : int = 8123
    CH_PORT : int = 9000
    CH_DB : str ='novoxpert'
    CH_TABLE : str ='tradingview_ohlcv'
    CH_USER : str = 'StreamNode1'
    CH_PASS : str = '‌Block2#InTerCoM!'

@dataclass
class RedisCfg:
    host: str = 'localhost'
    port: int = 6379
    db : int = 0

