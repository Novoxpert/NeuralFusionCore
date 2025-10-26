"""
backtesting_service.py
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Oct 25
Version: 1.1.0 
"""

import json, numpy as np, pandas as pd, torch
from config import Paths, TrainCfg, FeatureCfg, BacktestCfg
from ..lib.model import MarketNewsFusionWeightModel
from ..lib.dataset import make_loaders
from ..lib.backtest_weights import backtest_weight_logits
import time
from ..lib.utils import plot_equity

def main():
    start_service_time= time.time()
    torch.cuda.empty_cache()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=4, help="How many past hours of data to fetch")
    parser.add_argument("--device", type=str, default='cpu')
    args = parser.parse_args()

    P = Paths(); T = TrainCfg(); F = FeatureCfg(); B = BacktestCfg()
    df_tr = pd.read_parquet(P.processed_backtesting_dir + '/backtesting_train.parquet')
    df_va = pd.read_parquet(P.processed_backtesting_dir + '/backtesting_val.parquet')
    df_te = pd.read_parquet(P.processed_backtesting_dir + '/backtesting_test.parquet')
    meta = json.load(open(P.processed_backtesting_dir + '/meta.json'))

  

if __name__ == '__main__':
    main()
