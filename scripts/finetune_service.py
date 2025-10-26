#!/usr/bin/env python3
"""
finetune_service.py
Fine-tune an existing saved model using the latest features.
If validation loss improves, replace saved model and keep previous version with timestamp.

Usage:
  python finetune_service.py --epochs 5 --save_best
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Oct 04
Version: 1.1.0
"""
import logging, os, json, shutil, torch
from datetime import datetime
import pandas as pd
from ..config import Paths, TrainCfg, FeatureCfg, LossCfg, MarketCfg
from ..lib.model import MarketNewsFusionWeightModel
from ..lib.train import train_loop
from ..lib.dataset import make_loaders
from ..lib.redis_utils import atomic_model_swap  # utility to replace model atomically

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
P = Paths(); T = TrainCfg(); F = FeatureCfg(); L = LossCfg(); MC = MarketCfg()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tune epochs")
    parser.add_argument("--batch_size", type=int, default=T.batch_size)
    parser.add_argument("--temp_weights", type=str, default="/tmp/model_temp.pt")
    parser.add_argument("--save_best", action='store_true', help="Save fine-tuned weights if val improves")
    args = parser.parse_args()

    # 1) Load fine-tune datasets
    train_path = os.path.join(P.processed_dir, "finetune_train.parquet")
    val_path = os.path.join(P.processed_dir, "finetune_val.parquet")
    meta_path = os.path.join(P.processed_dir, "meta.json")
    normalizer_path = P.normalizer_pkl

    if not os.path.exists(train_path) or not os.path.exists(val_path) or not os.path.exists(meta_path):
        logging.error("Fine-tune train/val parquet or meta.json missing. Run feature_service first for finetune.")
        return
    df_tr = pd.read_parquet(train_path)
    df_va = pd.read_parquet(val_path)
    meta = json.load(open(meta_path))
    feat_cols = meta['feature_cols']
    data_stamp_cols = meta['data_stamp_cols']
    stock_list = meta['stock_list']
    cnt_cols = meta['count_cols']

    # 2) Create DataLoaders
    tr_loader, va_loader, _ = make_loaders(df_tr, df_va, None, F.seq_len, F.horizon_steps,
                                           feat_cols,data_stamp_cols, stock_list, cnt_cols, bs=args.batch_size)
    device = torch.device(T.device if torch.cuda.is_available() else 'cpu')

    # 3) Initialize model (load existing weights if available)
    n_stocks = len(stock_list)
    ts_in_dim = len(feat_cols)
    count_dim = len(cnt_cols) if cnt_cols else 0
    configs = {
        'task_name': 'classification',
        'seq_len': 10,
        'enc_in': len(feat_cols) ,
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
    model = MarketNewsFusionWeightModel(configs,ts_input_dim=ts_in_dim, num_stocks=n_stocks,
                                        d_model=T.d_model, nhead=T.nhead, num_layers=T.num_layers,
                                        news_embed_dim=768, hidden_dim=T.hidden_dim, count_dim=count_dim,
                                        max_len=F.seq_len).to(device)

    if os.path.exists(P.weights_pt):
        model.load_state_dict(torch.load(P.weights_pt, map_location=device))
        logging.info("Loaded existing model for fine-tuning.")
    else:
        logging.info("No existing model found — fine-tuning will train from scratch.")

    # 4) Fine-tune
    best_val_loss = train_loop(model, (tr_loader, va_loader, None), device=device,
                               epochs=args.epochs, patience=2, lr=1e-5,
                               save_path=args.temp_weights, k=T.top_k, gross=T.gross,
                               use_cov=L.use_cov, lambda_div=L.lambda_div,
                               lambda_net=L.lambda_net, lambda_turnover=L.lambda_turnover)
    logging.info(f"Fine-tune best val loss (temp): {best_val_loss}")

    # 5) Swap model if improved
    if args.save_best and os.path.exists(args.temp_weights):
        # Version current model
        if os.path.exists(P.weights_pt):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{P.weights_pt}.v{timestamp}"
            shutil.copy2(P.weights_pt, backup_path)
            logging.info(f"Backed up previous model to {backup_path}")

        # Atomic swap
        atomic_model_swap(args.temp_weights, P.weights_pt)
        logging.info("Replaced model weights with fine-tuned weights.")
    else:
        logging.info("No improvement by Finetunning...")

if __name__ == "__main__":
    main()
