#!/usr/bin/env python3
"""
prediction_service.py
--------------------
Scheduled inference: fetch latest data, compute features, infer model, 
                     transform logits into portfolio weights, and save predictions to MongoDB and Redis.
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
            print(f"⚙️ [load_model] torch.load returned keys: {list(state_dict.keys())}")
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
                print(f"⚙️ [load_model] unwrapped 'model_state_dict', now keys: {list(state_dict.keys())}")
            model.load_state_dict(state_dict)
            print(f"⚙️ [load_model] model.load_state_dict() CALLED ✅")
            logging.info(f"✅ Loaded model weights from {weights_path}")
        else:
            logging.warning(f"⚠️ No model weights found at {weights_path}, using untrained model.")
    except Exception as e:
        logging.error(f"❌ Failed to load weights: {e}")

    model.eval()
    return model

# --------------------------- Inference ---------------------------
def run_inference(df_tr, df_va, df_te, feat_cols,data_stamp_cols, stock_list, cnt_cols, device='cpu'):
    _, _, te_loader = make_loaders(df_tr, df_va, df_te, F.seq_len, F.horizon_steps,
                                   feat_cols, data_stamp_cols, stock_list, cnt_cols, bs=T.batch_size)
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
    model = load_model(configs, len(feat_cols), len(stock_list), len(cnt_cols) if cnt_cols else 0, device)
    all_weights = []

    with torch.no_grad():
        for b in te_loader:
            ts_t = b['timeseries'].to(device)
            news_t = b['news'].to(device)
            cnt_t = b['news_count'].to(device)
            mask = b['time_mask'].to(device)

            logits_t = model(ts_t, mask, cnt_t, news_t)

            # ---- transform logits into weights ----
            w = weights_long_short_topk_abs(logits_t,
                                            k=T.top_k,
                                            gross=T.gross).squeeze(0).cpu().numpy()
            all_weights.append(w)

    weights = np.vstack(all_weights)
    return weights

# --------------------------- Save predictions ---------------------------
def save_predictions(weights, stock_list):
    ts_now = datetime.now(timezone.utc)
    payload = {"ts": ts_now, "weights": weights.tolist(), "stocks": stock_list}

    # Save to Redis
    redis_client.set("predictions", pickle.dumps(payload))
    logging.info("Saved predictions to Redis")

    # Save to MongoDB
    mongo_col.insert_one(payload)
    logging.info("Saved predictions to MongoDB")

    # Save to JSON file
    ts_now = datetime.now(timezone.utc).isoformat()  # ✅ convert to string
    payload = {
        "ts": ts_now,
        "weights": weights.tolist(),
        "stocks": stock_list
    }
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "prediction.json")
    try:
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=4)
        logging.info(f"Saved predictions to JSON file: {json_path}")
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
    args = parser.parse_args()

    # 1) Fetch latest data
    run_data_ingest(args.hours)
    run_feature_service(args.hours)

    # 2) Load online_test.parquet
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
    weights = run_inference(df_tr, df_va, df_te, feat_cols, data_stamp_cols, stock_list, cnt_cols, device=args.device)

    # 5) Save predictions
    save_predictions(weights, stock_list)
    logging.info("Prediction cycle complete.")
    end_service_time = time.time()
    print(f"Time elapsed for prediction service: {end_service_time - start_service_time:.2f} seconds")

if __name__ == "__main__":
    main()
