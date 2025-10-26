#!/usr/bin/env python3
"""
train_service.py
Train from scratch on processed/train.parquet and processed/val.parquet
Usage:
  python train_service.py --epochs 50
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Sep 30
Version: 1.0.1 
"""
import logging, json, os
import torch
import pandas as pd
import argparse
from ..config import Paths, TrainCfg, FeatureCfg, LossCfg
from ..lib.dataset import make_loaders
from ..lib.model import MarketNewsFusionWeightModel
from ..lib.train import train_loop
import time

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def main():
    start_service_time= time.time()
    P = Paths(); T = TrainCfg(); F = FeatureCfg(); L = LossCfg()
    # load data
    tr_path = os.path.join(P.processed_dir, 'train.parquet')
    va_path = os.path.join(P.processed_dir, 'val.parquet')
    meta_path = os.path.join(P.processed_dir, 'meta.json')
    if not os.path.exists(tr_path) or not os.path.exists(va_path) or not os.path.exists(meta_path):
        logging.error("train.parquet, val.parquet or meta.json missing. Run features_service first.")
        return
    df_tr = pd.read_parquet(tr_path)
    df_va = pd.read_parquet(va_path)
    meta = json.load(open(meta_path))
    feat_cols = meta['feature_cols']; stock_list = meta['stock_list']; cnt_cols = meta['count_cols']; data_stamp_cols = meta['data_stamp_cols']

    tr_loader, va_loader, te_loader = make_loaders(df_tr, df_va, None, F.seq_len, F.horizon_steps,
                                                  feat_cols, data_stamp_cols, stock_list, cnt_cols, bs=T.batch_size)
    device = torch.device(T.device if torch.cuda.is_available() else 'cpu')
    n_stocks = len(stock_list)
    ts_in_dim = len(feat_cols)
    #mask_in_dim = len(data_stamp_cols)
    count_dim = len(cnt_cols) if cnt_cols else 0
    #Configs
    configs = {
        'task_name': 'classification',
        'seq_len': 10,
        'enc_in': ts_in_dim ,
        'd_model': 64,
        'c_out':2, #len(selected_f),
        'd_ff': 128,
        'num_kernels': 3,
        'dropout': 0.1,
        'e_layers': 2,
        'top_k': 3,
        'num_class': 2,
        'label_len':30,
        'pred_len':1,
        'embed':'timeF',
        'freq':'t'
    }
    model = MarketNewsFusionWeightModel(configs=configs,ts_input_dim=ts_in_dim, num_stocks=n_stocks,
                                        d_model=T.d_model, nhead=T.nhead, num_layers=T.num_layers,
                                        news_embed_dim=768, hidden_dim=T.hidden_dim, count_dim=count_dim,
                                        max_len=F.seq_len).to(device)
    logging.info("Starting training from scratch.")
    best = train_loop(model, (tr_loader, va_loader, te_loader), device=device,
                      epochs=T.epochs, patience=T.patience, lr=T.lr,
                      save_path=P.weights_pt, k=T.top_k, gross=T.gross,
                      use_cov=L.use_cov, lambda_div=L.lambda_div,
                      lambda_net=L.lambda_net, lambda_turnover=L.lambda_turnover)
    logging.info("Training finished. Best val loss: %s" % str(best))

    end_service_time = time.time()
    print(f"Time elapsed for train service: {end_service_time - start_service_time:.2f} seconds")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    if args.epochs:
        TrainCfg.epochs = args.epochs  # if you want dynamic override (or modify TrainCfg reading)
    main()
