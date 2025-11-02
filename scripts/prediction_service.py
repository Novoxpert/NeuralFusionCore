#!/usr/bin/env python3
"""
prediction_service.py
=====================

Scheduled prediction pipeline for the NeuralFusionCore portfolio model.

This service automates the full inference workflow:
1. Fetch the latest market and news data.
2. Generate time-series & news fusion features in inference mode.
3. Load the latest trained model weights.
4. Run model inference to produce stock allocation logits.
5. Convert logits into portfolio weights using top-k long/short strategy.
6. Persist predictions to Redis, MongoDB, and JSON output.

Intended to run as a scheduled job (e.g., cron or Kubernetes CronJob), enabling
continuous and near-real-time portfolio signal updates.

Features
--------
- Market + news fusion model inference (PyTorch)
- On-demand latest-data ingestion
- Sliding window feature generation
- Portfolio weight generation with long/short allocation
- Multi-destination persistence layer (Redis, MongoDB, JSON)
- Config-driven architecture via project `config` modules

Usage
-----
Run end-to-end pipeline (data ingest → features → inference → save):

    python3 prediction_service.py --hours 4 --device cpu --mode synchrone

Run inference only (assumes latest features already exist):

    python3 prediction_service.py --device cuda --mode inference

Arguments
---------
--hours : int, default=4  
    Number of past hours of data to fetch for feature generation.

--device : str, default="cpu"  
    Compute device (`cpu` or `cuda`).

--mode : {"synchrone", "inference"}, default="synchrone"  
    Execution type:
      inference  – Fetch data, build features, infer, save outputs
      synchrone  – Only run inference + saving (data assumed prepared)

Environment
-----------
Requires access to:
- MongoDB (for persistent logs)
- Redis (for low-latency serving of latest predictions)
- Trained model weight file
- Preprocessed parquet + metadata files produced by the pipeline

Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Oct 05
Version: 1.2.0
"""

import os, sys, subprocess, logging, pickle, torch, numpy as np, pandas as pd, json
from datetime import datetime, timezone
from pymongo import MongoClient
from ..config import Paths, FeatureCfg, MarketCfg, TrainCfg, BacktestCfg
from ..lib.model import MarketNewsFusionWeightModel
from ..lib.dataset import make_loaders
from ..lib.backtest_weights import backtest_weight_logits, weights_long_short_topk_abs
from ..lib.redis_utils import redis_client
import time

P = Paths(); F = FeatureCfg(); MC = MarketCfg(); T = TrainCfg(); B = BacktestCfg()
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# --------------------------- MongoDB setup ---------------------------
mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["portfolio_db"]
mongo_col = mongo_db["NeuralFusionCore_predictions"]

# --------------------------- Data ingest & feature service ---------------------------
def run_data_ingest(hours):
    logging.info(f"Running data_ingest_service to fetch last {hours} hour(s) of data")
    subprocess.run([sys.executable, '-m', 'apps.NeuralFusionCore.scripts.data_ingest_service', '--mode', 'latest', '--hours', str(hours)], check=True)

def run_feature_service(hours):
    logging.info(f"Running features_service in INFERENCE mode for last {hours} hour(s)")
    subprocess.run([sys.executable, '-m', 'apps.NeuralFusionCore.scripts.features_service', '--mode', 'inference', '--latest_hours', str(hours)], check=True)

# --------------------------- Model loader ---------------------------
def load_model(configs, feat_cols_len, stock_list_len, count_dim, device='cpu'):
    model = MarketNewsFusionWeightModel(
        configs=configs,
        ts_input_dim=feat_cols_len,
        num_stocks=stock_list_len,
        d_model=T.d_model,
        nhead=T.nhead,
        num_layers=T.num_layers,
        news_embed_dim=768,
        hidden_dim=T.hidden_dim,
        count_dim=count_dim,
        max_len=F.seq_len
    ).to(device)

    weights_path = getattr(P, "weights_pt", "data/outputs/model_weights.pt")

    try:
        print(f"⚙️ [load_model] weights_path={weights_path}")
        if os.path.exists(weights_path):
            print("⚙️ [load_model] file exists, loading weights")
            state_dict = torch.load(weights_path, map_location=device)
            #print(f"⚙️ [load_model] torch.load returned keys: {list(state_dict.keys())}")
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
                #print(f"⚙️ [load_model] unwrapped 'model_state_dict', now keys: {list(state_dict.keys())}")
            model.load_state_dict(state_dict)
            #print(f"⚙️ [load_model] model.load_state_dict() CALLED ✅")
            logging.info(f"✅ Loaded model weights from {weights_path}")
        else:
            logging.warning(f"⚠️ No model weights found at {weights_path}, using untrained model.")
    except Exception as e:
        logging.error(f"❌ Failed to load weights: {e}")

    model.eval()
    return model

# --------------------------- Inference ---------------------------
def run_inference(df_tr, df_va, df_te, feat_cols, data_stamp_cols, stock_list, cnt_cols, device='cpu', mode='synchrone'):
    _, _, te_loader = make_loaders(df_tr, df_va, df_te, F.seq_len, F.horizon_steps,
                                   feat_cols, data_stamp_cols, stock_list, cnt_cols, bs=T.batch_size)
    
    configs = {
        'task_name': 'classification',
        'seq_len': F.seq_len,
        'enc_in': len(feat_cols),
        'd_model': 64,
        'c_out': 2,
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
    }

    model = load_model(configs, len(feat_cols), len(stock_list), len(cnt_cols) if cnt_cols else 0, device)

    all_predictions = []
   
    with torch.no_grad():
        if mode == 'synchrone':
            # only take the last seq_len rows for a single prediction
            df_seq = df_te.iloc[-F.seq_len:]
            ts_t = torch.tensor(df_seq[feat_cols].values.astype(np.float32)).unsqueeze(0).to(device)
            news_t = torch.tensor(np.stack(df_seq['embedding'].values), dtype=torch.float32).unsqueeze(0).to(device)
            cnt_t = torch.tensor(df_seq[cnt_cols].values.astype(np.float32)).unsqueeze(0).to(device) if cnt_cols else torch.zeros((1, F.seq_len, 1), dtype=torch.float32).to(device)
            mask = torch.tensor(df_seq[data_stamp_cols].values.astype(np.float32)).unsqueeze(0).to(device)

            logits_t = model(ts_t, mask, cnt_t, news_t)
            w = weights_long_short_topk_abs(logits_t, k=T.top_k, gross=T.gross).squeeze(0).cpu().numpy()
            pred_time = df_seq["dateTime"].iloc[-1] + pd.to_timedelta(F.horizon_steps * F.resample, unit='min')
            print("last timestamp:")
            print(df_seq["dateTime"].iloc[-1])
            print("predicted timestamp")
            print(pred_time)
            all_predictions.append({"ts": pred_time, "weights": w, "stocks": stock_list})

        else:  # mode == inference
            for b_idx, b in enumerate(te_loader):
                ts_t = b['timeseries'].to(device)
                news_t = b['news'].to(device)
                cnt_t = b['news_count'].to(device)
                mask = b['time_mask'].to(device)

                logits_t = model(ts_t, mask, cnt_t, news_t)
                w_batch = weights_long_short_topk_abs(logits_t, k=T.top_k, gross=T.gross).squeeze(0).cpu().numpy()

                # calculate prediction timestamp for each sequence
                for i in range(len(w_batch)):
                    seq_last_time = df_te["dateTime"].iloc[i + F.seq_len - 1]
                    pred_time = seq_last_time  + pd.to_timedelta(F.horizon_steps * F.resample, unit='min')
                    all_predictions.append({"ts": pred_time, "weights": w_batch[i], "stocks": stock_list})

    return all_predictions

# --------------------------- Save predictions ---------------------------
def save_predictions(predictions):
    for pred in predictions:
        ts_now = pred['ts']
        print("inference dates are:") 
        print(ts_now)
        weights = pred['weights']
        stock_list = pred['stocks']

        payload = {"ts": ts_now, "weights": weights.tolist(), "stocks": stock_list}

        # Save to Redis (latest only)
        redis_client.set("predictions", pickle.dumps(payload))

        # Save to MongoDB
        mongo_col.insert_one(payload)

        # Save to JSON file
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NeuralFusionCore_prediction.json")
        try:
            with open(json_path, "w") as f:
                json.dump(payload, f, indent=4, default=str)
        except Exception as e:
            logging.error(f"Error saving predictions to JSON: {e}")

# --------------------------- Main ---------------------------
def main():
    start_service_time= time.time()
    torch.cuda.empty_cache()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=4, help="How many past hours of data to fetch")
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--mode", type=str,  default="synchrone", choices=["synchrone", "inference"], help="Execution mode")
    args = parser.parse_args()

    if args.mode == "inference":
        # Fetch latest data
        run_data_ingest(args.hours)
        run_feature_service(args.hours)

    # Load online_test.parquet
    online_path = os.path.join(P.processed_dir, "online_test.parquet")
    if not os.path.exists(online_path):
        logging.error(f"{online_path} not found. Exiting.")
        return
    df_te = pd.read_parquet(online_path)

    # 3) Load meta info
    meta_path = os.path.join(P.processed_dir, 'meta.json')
    meta = json.load(open(meta_path))
    feat_cols = meta['feature_cols']
    data_stamp_cols = meta['data_stamp_cols']
    stock_list = meta['stock_list']
    cnt_cols = meta.get('count_cols', [])

    # Use df_te as both train/val to satisfy make_loaders
    df_tr = df_va = df_te.copy()

    # 4) Run inference & convert logits -> weights
    predictions = run_inference(df_tr, df_va, df_te, feat_cols, data_stamp_cols, stock_list, cnt_cols, device=args.device, mode=args.mode)

    # 5) Save predictions
    save_predictions(predictions)
    logging.info("Prediction cycle complete.")
    end_service_time = time.time()
    print(f"Time elapsed for prediction service: {end_service_time - start_service_time:.2f} seconds")

if __name__ == "__main__":
    main()
