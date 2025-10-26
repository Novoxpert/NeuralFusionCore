#!/usr/bin/env python3
"""
features_service.py
-------------------
Builds features from Redis.

Modes:
  - train:     full rebuild (includes normalizer + meta)
  - finetune:  incremental build (reuse existing normalizer/meta)
  - inference: build features for inference only (produces online_test.parquet)
  - bridge:    build features for ChronobBridge only
  - time:      select data by start_time/end_time for any mode

Examples:
  python features_service.py --mode train --history_days 30 --val_frac 0.2
  python features_service.py --mode finetune --latest_hours 24
  python features_service.py --mode inference --latest_hours 4
  python features_service.py --mode train --start_time "2025-10-01T00:00" --end_time "2025-10-02T00:00"

Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Updated: 2025 Oct 4
"""

import argparse, logging, os, json, pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from ..config import Paths, FeatureCfg, NewsCfg, MarketCfg
from ..lib import features as F, news as N
from ..lib.redis_utils import get_all_redis_data
import time

P = Paths(); FC = FeatureCfg(); NC = NewsCfg(); MC = MarketCfg()
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


# ----------------------------------------------------------------------
def make_features_from_redis(start_time=None, end_time=None, no_news_vec=None, mode="train"):
    """Fetch OHLCV + News and build merged feature table."""
    ohlcv_dict, df_news = get_all_redis_data(
        #[f"BINANCE:{s}" for s in MC.symbols_usdt],
        [f"{s}" for s in MC.symbols_usdt],
        start_time=start_time,
        end_time=end_time
    )
    per_asset = []
    for sym, df3m in ohlcv_dict.items():
        if df3m is None or df3m.empty:
            continue
        
        per_asset.append(F.add_targets_and_features(
            df3m, FC.horizon_steps, FC.seq_len, None, sym, mode
        ))

    if not per_asset:
        logging.warning("No OHLCV data available in Redis.")
        return pd.DataFrame(), None, None

    merged = F.merge_assets(per_asset)

    # --- ensure all symbols exist ---
    for sym in MC.symbols_usdt:
        if mode == 'bridge':
            cols_needed = [f"{sym}_{c}" for c in
                       ["open","high","low","close", "volume", "prev_return", "prev_volatility", "return", "volatility", "target_return"]]
        else:
            cols_needed = [f"{sym}_{c}" for c in
                        ["close", "volume", "prev_return", "prev_volatility", "return", "volatility", "target_return"]]
        for c in cols_needed:
            if c not in merged.columns:
                merged[c] = 0.0

    #---- add timesnet features---
    tcols = F.make_time_cols(merged)
    merged = pd.concat([merged, tcols], axis=1)

    # --- add news embeddings ---
    if df_news is not None and not df_news.empty:
        df_news = N.add_onehot_columns(df_news, MC.symbols_usdt)
        tok, mdl, device, max_len = N.load_text_encoder(NC.model_path)
        df_news['embedding'] = [e for e in N.embed_texts(
            df_news['content'].to_list(), tok, mdl, device, max_len, NC.pooling, NC.batch_size
        )]
        df_news['news_count'] = 1
        if no_news_vec is None:
            no_news_vec = N.embed_texts(
                [NC.no_news_token], tok, mdl, device, max_len, NC.pooling, batch_size=1
            )[0]

        news_3m = N.resample_news_3m(
            df_news[['releasedAt', 'embedding', 'news_count'] + [s for s in MC.symbols_usdt ]].copy(),
            np.asarray(no_news_vec), rule=NC.rule)
        
        merged = F.attach_news(merged, news_3m)


    merged['embedding'] = merged['embedding'].apply(
        lambda x: np.asarray(x, dtype=np.float32) if isinstance(x, (list, np.ndarray)) else no_news_vec
    )

    non_embed_cols = [c for c in merged.columns if c != 'embedding']
    merged[non_embed_cols] = merged[non_embed_cols].fillna(0)
    return merged, df_news, no_news_vec


# ----------------------------------------------------------------------
def time_split_and_save(merged, val_frac=0.2, mode="train"):
    """Split dataset into train/val and normalize depending on mode."""
    os.makedirs(P.processed_dir, exist_ok=True)

    # determine columns
    all_cols = [c for c in merged.columns if any(
        c.endswith(x) for x in ['close', 'volume', 'prev_return', 'prev_volatility', 'return', 'volatility']
    )]
    feat_cols = [c for c in all_cols if (('return' not in c) or ('prev_return' in c))]
    feat_cols = [c for c in feat_cols if (('volatility' not in c) or ('prev_volatility' in c))]
    stock_list = [s for s in MC.symbols_usdt]
    #count_cols = [c for c in merged.columns if c in [s[:-4] for s in MC.symbols_usdt]]
    count_cols = [c for c in merged.columns if c in [s for s in MC.symbols_usdt]]
    data_stamp_cols = ['month', 'day', 'weekday', 'hour', 'minute']
    merged = merged.sort_values("dateTime").reset_index(drop=True)
    for c in [c for c in merged.columns if c.endswith(("return", "volatility", "close", "volume","open","high","low"))]:
        merged[c] = pd.to_numeric(merged[c], errors='coerce').fillna(0.0)

    n = len(merged)
    n_tr = int((1.0 - val_frac) * n)
    split_idx = (range(0, n_tr), range(n_tr, n), [])
  
    # --- Mode-specific normalization ---
    if mode == "train":
        logging.info("Mode: TRAIN → Fitting new normalizer...")
        df_tr, df_va, df_te, stats = F.normalize_train_val_test_stream(
            merged.fillna(0),
            feature_cols=feat_cols,
            split_idx=split_idx,
            save_path=P.normalizer_pkl
        )
        # save meta
        with open(os.path.join(P.processed_dir, 'meta.json'), 'w') as f:
            json.dump({'feature_cols': feat_cols, 'stock_list': stock_list, 'count_cols': count_cols, 'data_stamp_cols': data_stamp_cols}, f)

        # save parquet
        df_tr.to_parquet(os.path.join(P.processed_dir, 'train.parquet'), index=False)
        df_va.to_parquet(os.path.join(P.processed_dir, 'val.parquet'), index=False)

    elif mode == "finetune":
        logging.info("Mode: FINETUNE → Reusing existing normalizer...")
        merged_norm = F.apply_existing_normalizer(
            df=merged.fillna(0),
            feature_cols=feat_cols,
            normalizer_path=P.normalizer_pkl
        )
        df_tr = merged_norm.iloc[:n_tr].reset_index(drop=True)
        df_va = merged_norm.iloc[n_tr:].reset_index(drop=True)
        df_te = pd.DataFrame()
        df_tr.to_parquet(os.path.join(P.processed_dir, 'finetune_train.parquet'), index=False)
        df_va.to_parquet(os.path.join(P.processed_dir, 'finetune_val.parquet'), index=False)
        logging.info("Saved finetune train/val parquet files.")

    elif mode == "back_testing":
        logging.info("Mode: BACK_TESTING → Fitting new normalizer...")
        n = len(merged)
        val_frac = 0.2
        test_frac = 0.1
        n_tr = int((1.0 - val_frac - test_frac) * n)  # number of training samples
        n_va = int(val_frac * n)                       # number of validation samples
        n_te = n - n_tr - n_va                         # remaining samples for test

        split_idx = (
            range(0, n_tr),                # train indices
            range(n_tr, n_tr + n_va),      # validation indices
            range(n_tr + n_va, n)          # test indices
        )

        df_tr, df_va, df_te, stats = F.normalize_train_val_test_stream(
            merged.fillna(0),
            feature_cols=feat_cols,
            split_idx=split_idx,
            save_path=P.normalizer_backtesting_pkl
        )
        # save meta
        with open(os.path.join(P.processed_backtesting_dir, 'meta.json'), 'w') as f:
            json.dump({'feature_cols': feat_cols, 'stock_list': stock_list, 'count_cols': count_cols, 'data_stamp_cols': data_stamp_cols}, f)

        # save parquet
        df_tr.to_parquet(os.path.join(P.processed_backtesting_dir, 'backtesting_train.parquet'), index=False)
        df_va.to_parquet(os.path.join(P.processed_backtesting_dir, 'backtesting_val.parquet'), index=False)
        df_te.to_parquet(os.path.join(P.processed_backtesting_dir, 'backtesting_val.parquet'), index=False)

    elif mode == "inference":
        logging.info("Mode: INFERENCE → Reusing existing normalizer...")
        merged_norm = F.apply_existing_normalizer(
            df=merged.fillna(0),
            feature_cols=feat_cols,
            normalizer_path=P.normalizer_pkl
        )
        # Save online_test.parquet
        online_path = os.path.join(P.processed_dir, "online_test.parquet")
        merged_norm.to_parquet(online_path, index=False)
        logging.info(f"Saved inference parquet file: {online_path}")

    elif mode == "bridge":
        logging.info("Mode: BRIDGE → Reusing existing normalizer...")
        merged_not_norm = merged.fillna(0).copy()
        # Drop columns that end with 'open', 'high', or 'low'
        merged = merged.drop(
            columns=[col for col in merged.columns if col.endswith(("open", "high", "low"))]
        )
        merged_norm = F.apply_existing_normalizer(
            df=merged.fillna(0),
            feature_cols=feat_cols,
            normalizer_path=P.normalizer_pkl
        )
        # Save online_test.parquet
        online_bridge_path = os.path.join(P.processed_dir, "online_bridge.parquet")
        merged_norm.to_parquet(online_bridge_path, index=False)
        logging.info(f"Saved bridge parquet file: {online_bridge_path}")
        online_bridge_not_norm_path = os.path.join(P.processed_dir, "online_bridge_not_norm.parquet")
        merged_not_norm.to_parquet(online_bridge_not_norm_path, index=False)
        logging.info(f"Saved bridge parquet file: {online_bridge_not_norm_path}")

    else:
        logging.error(f"Mode {mode} not recognized.")


# ----------------------------------------------------------------------
def parse_time_args(start_time_str, end_time_str):
    start_time, end_time = None, None
    if start_time_str:
        start_time = pd.to_datetime(start_time_str)
    if end_time_str:
        end_time = pd.to_datetime(end_time_str)
    return start_time, end_time


# ----------------------------------------------------------------------
def main():
     
    start_service_time= time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "finetune", "inference", "bridge", "back_testing", "future_testing"], required=True)
    parser.add_argument("--latest_hours", type=int, default=None)
    parser.add_argument("--history_days", type=int, default=None)
    parser.add_argument("--start_time", type=str, default=None)
    parser.add_argument("--end_time", type=str, default=None)
    parser.add_argument("--val_frac", type=float, default=0.2)
    args = parser.parse_args()

    # compute start/end time based on latest_hours or history_days if not explicitly provided
    start_time, end_time = parse_time_args(args.start_time, args.end_time)
    if args.latest_hours is not None:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=args.latest_hours)
    elif args.history_days is not None:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=args.history_days)

    logging.info(f"Running feature builder in {args.mode.upper()} mode from {start_time} to {end_time}...")
    merged, df_news, no_news_vec = make_features_from_redis(start_time=start_time, end_time=end_time , mode=args.mode)
    if merged.empty:
        logging.error("No merged features created. Exiting.")
        return

    time_split_and_save(merged, val_frac=args.val_frac, mode=args.mode)
    end_service_time = time.time()
    print(f"Time elapsed for features service: {end_service_time - start_service_time:.2f} seconds")


if __name__ == "__main__":
    main()
    