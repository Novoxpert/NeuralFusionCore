"""
backtesting_service.py
======================

Backtesting & Model Evaluation Service for Market-News Fusion Model.

Workflow
--------
1) Load parquet train/val/test sets + metadata
2) Build sliding-window dataloaders
3) Train model if no weights exist (early stopping)
4) Load MarketNewsFusionWeightModel
5) Forward pass on masked sequences
6) Convert logits -> long/short portfolio weights (top-K)
7) Backtest:
      a) Raw logits → weights → equity
      b) SL/TP per-asset backtest
8) Compute:
      * Sharpe/Sortino/CAGR/MaxDD
      * Underwater curve
      * Rolling Sharpe
      * Turnover
9) Save equity curve + portfolio pickle

-----
Usage:
  python backtesting_service.py --epochs 50 --mode fetch --hours 12
Notes
-----
• False alignment avoided (no repeating logits!)
• Handles variable sequence windows via mask
• Use research only; not live trading engine

Author  : Elham Esmaeilnia (elham.e.shirvani@gmail.com)
Updated : 2025-11-02
Version : 1.3.1
"""

import os, sys, subprocess, logging, torch, numpy as np, pandas as pd, json
import matplotlib.pyplot as plt
import argparse
from ..config import Paths, TrainCfg, FeatureCfg, BacktestCfg, MarketCfg, LossCfg
from ..lib.model import MarketNewsFusionWeightModel
from ..lib.dataset import make_loaders
from ..lib.backtest_weights import backtest_weight_logits, weights_long_short_topk_abs
from ..lib.utils import plot_equity
from ..lib.train import train_loop
from ..lib.backtest import (
    backtest_sl_tp_per_asset,
    summarize_curves as summarize_curves_sl,
    plot_equity as plot_equity_sl
)
import time

# ---- Configs ----
P = Paths(); F = FeatureCfg(); MC = MarketCfg(); T = TrainCfg(); B = BacktestCfg(); L = LossCfg()

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# ---- Reproducibility ----
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

# ================================================================
#   Helpers
# ================================================================
def annualized_sharpe(r, steps_per_year=252):
    r = np.asarray(r, float)
    return 0 if r.std() == 0 else (r.mean() / r.std()) * np.sqrt(steps_per_year)

def sortino_ratio(r, steps_per_year=252):
    r = np.asarray(r, float)
    neg = r[r < 0]
    return np.nan if neg.std() == 0 else (r.mean() / neg.std()) * np.sqrt(steps_per_year)

def max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    return float((equity/peak - 1).min())

def compute_trade_stats(equity):
    s = pd.Series(equity).pct_change().dropna()
    return {
        "total_return": float(equity[-1] - equity[0]),
        "sharpe_ann": annualized_sharpe(s),
        "sortino_ann": sortino_ratio(s),
        "cagr": ((equity[-1]/equity[0]) ** (252/len(equity)) - 1) if equity[0] > 0 else np.nan,
        "max_drawdown": max_drawdown(equity),
        "win_rate": float((s > 0).mean()) if len(s) else np.nan,
        "avg_win": float(s[s>0].mean()) if (s>0).any() else np.nan,
        "avg_loss": float(s[s<0].mean()) if (s<0).any() else np.nan
    }

# ---- Plot Utils ----
def plot_underwater(dates, equity):
    eq, peak = np.asarray(equity), np.maximum.accumulate(equity)
    dd = eq / peak - 1
    plt.figure(figsize=(10,3.5))
    plt.plot(dates, dd)
    plt.fill_between(dates, dd, 0, alpha=0.2)
    plt.title("Underwater (Drawdown)"); plt.grid(); plt.tight_layout(); plt.show()

def plot_rolling_sharpe(dates, equity, window=5):
    """
    Rolling Sharpe plot.
    window: number of periods for rolling calculation
    """
    r = pd.Series(equity).pct_change().dropna()
    if len(r) < window:
        print(f"⚠️ Not enough data points ({len(r)}) for rolling Sharpe with window={window}")
        return

    roll = r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(252)
    roll = roll.dropna()

    plt.figure(figsize=(10,3.5))
    plt.plot(dates[-len(roll):], roll)
    plt.title(f"Rolling Sharpe (window={window})")
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_turnover(df, weight_cols, dates):
    W = df[weight_cols].fillna(0).values
    turnover = np.sum(np.abs(np.diff(W, axis=0)), axis=1)
    plt.figure(figsize=(10,3.5))
    plt.plot(dates[1:], turnover)
    plt.title("Turnover"); plt.grid(); plt.tight_layout(); plt.show()

# ================================================================
#  Data ingest & feature service
# ================================================================
def run_data_ingest(hours):
    logging.info(f"Running data_ingest_service to fetch last {hours} hour(s) of data")
    subprocess.run([sys.executable, '-m', 'apps.NeuralFusionCore.scripts.data_ingest_service', '--mode', 'latest', '--hours', str(hours)], check=True)

def run_feature_service(hours):
    logging.info(f"Running features_service in INFERENCE mode for last {hours} hour(s)")
    subprocess.run([sys.executable, '-m', 'apps.NeuralFusionCore.scripts.features_service', '--mode', 'backtesting', '--latest_hours', str(hours)], check=True)
# ================================================================
#   Load Model
# ================================================================
def load_model(feat_dim, num_stocks, count_dim, device):

    model = MarketNewsFusionWeightModel(
        configs={
            'task_name':'classification','seq_len':F.seq_len,'enc_in':feat_dim,'d_model':T.d_model,
            'c_out':2,'d_ff':128,'num_kernels':3,'dropout':0.1,'e_layers':T.num_layers,
            'top_k':3,'num_class':2,'label_len':30,'pred_len':1,'embed':'timeF','freq':'t'
        },
        ts_input_dim=feat_dim,
        num_stocks=num_stocks,
        d_model=T.d_model,
        nhead=T.nhead,
        num_layers=T.num_layers,
        news_embed_dim=768,
        hidden_dim=T.hidden_dim,
        count_dim=count_dim,
        max_len=F.seq_len
    ).to(device)

    if os.path.exists(P.weights_pt):
        logging.info(f"✅ Loading weights: {P.weights_pt}")
        state = torch.load(P.weights_pt, map_location=device)
        state = state.get("model_state_dict", state)
        model.load_state_dict(state)
    else:
        logging.warning("⚠️ No weights found -> train first run")

    model.eval()
    return model

# ================================================================
#   MAIN
# ================================================================
def main():
    start = time.time()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="fetch",choices=["fetch", "use_saved"], help="Execution mode")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hours", type=int, default=4, help="How many past hours of data to fetch")
    parser.add_argument("--device", type=str, default='cpu')
    args = parser.parse_args()
    if args.epochs:
        TrainCfg.epochs = args.epochs

    # ---- Fetch latest data ---
    if args.mode == "fetch":
        run_data_ingest(args.hours)
        run_feature_service(args.hours)

    # ---- Load data ----
    df_tr = pd.read_parquet(f"{P.processed_backtesting_dir}/backtesting_train.parquet")
    df_va = pd.read_parquet(f"{P.processed_backtesting_dir}/backtesting_val.parquet")
    df_te = pd.read_parquet(f"{P.processed_backtesting_dir}/backtesting_test.parquet")
    meta = json.load(open(f"{P.processed_backtesting_dir}/meta.json"))

    feat_cols, stock_list = meta['feature_cols'], meta['stock_list']
    cnt_cols, stamp_cols = meta['count_cols'], meta['data_stamp_cols']
    device = torch.device(T.device if torch.cuda.is_available() else 'cpu')

    tr_loader, va_loader, te_loader = make_loaders(
        df_tr, df_va, df_te,
        F.seq_len, F.horizon_steps,
        feat_cols, stamp_cols,
        stock_list, cnt_cols,
        bs=T.batch_size
    )

    # ---- Train if no weights ----
    if not os.path.exists(P.weights_pt):
        m = load_model(len(feat_cols), len(stock_list), len(cnt_cols or []), device)
        train_loop(m,(tr_loader,va_loader,te_loader),device=device,
                   epochs=T.epochs,patience=T.patience,lr=T.lr,
                   save_path=P.weights_pt,k=T.top_k,gross=T.gross,
                   use_cov=L.use_cov,lambda_div=L.lambda_div,
                   lambda_net=L.lambda_net,lambda_turnover=L.lambda_turnover)

    # ---- Load trained model ----
    model = load_model(len(feat_cols), len(stock_list), len(cnt_cols or []), device)

    # ---- Inference ----
    all_logits = []

    with torch.no_grad():
        for b in te_loader:
            ts   = b['timeseries'].to(device)
            news = b['news'].to(device)
            cnt  = b['news_count'].to(device)

            # Do NOT use the internal time_mask for indexing
            logits = model(ts, b['time_mask'].to(device), cnt, news) 

            if logits.ndim == 3:  # (B, 1, N)
                logits = logits.squeeze(1)

            out = logits.cpu().numpy().reshape(-1, len(stock_list))

            # Append all rows directly, no mask indexing
            all_logits.append(out)

    # Stack all logits from all batches
    logits_full = np.vstack(all_logits)

    # ---- Align logits with test dataframe ----
    # If your te_loader produces padded sequences, drop padding using te dataframe
    valid_idx = np.arange(len(df_te))  # All rows assumed valid if no padding
    logits_full = logits_full[:len(valid_idx)]
    df_te_valid = df_te.iloc[valid_idx].reset_index(drop=True)

    # ---- Backtest (weights) ----
    ret_cols = [f"{s}_target_return" for s in stock_list]
    rets = df_te_valid[ret_cols].to_numpy(float)
    dates = pd.to_datetime(df_te_valid["dateTime"])

    # ---- Compute weights and equity ----
    def safe_backtest_weight_logits(pred_logits, returns_matrix, dates, k=T.top_k, gross=T.gross, stride=B.stride):
        M, N = pred_logits.shape
        T_steps = returns_matrix.shape[0]
        
        # Compute weights
        with torch.no_grad():
            logits_t = torch.tensor(pred_logits, dtype=torch.float32)
            w_t = weights_long_short_topk_abs(logits_t, k=k, gross=gross).cpu().numpy()
        
        # Forward-fill weights to match all time steps
        w_series = np.zeros((T_steps, N), dtype=float)
        idxs = list(range(0, T_steps, stride))
        for j, t0 in enumerate(idxs):
            end_idx = idxs[j + 1] if j + 1 < len(idxs) else T_steps
            w_series[t0:end_idx, :] = w_t[min(j, M-1)]
        
        # Compute portfolio returns
        rp = (w_series * returns_matrix).sum(axis=1)
        equity = rp.cumsum()
        equity = equity - equity[0] + 1  # Start at 1

        # Save portfolio dataframe
        df_portfolio = pd.DataFrame({'dateTime': dates})
        for i, sym in enumerate(MarketCfg().symbols_usdt):
            df_portfolio[f"{sym}_return"] = returns_matrix[:, i]
            df_portfolio[f"{sym}_weight"] = w_series[:, i]
        
        df_portfolio.to_pickle(os.path.join(P.outputs_dir, 'df_portfolio.pickle'))
        
        return {'dates': dates, 'equity': equity, 'rp': rp, 'weights': w_series}

    # Run backtest
    curves = safe_backtest_weight_logits(logits_full, rets, dates)

    # ---- Plot equity ----
    out_png = os.path.join(P.outputs_dir, "equity_weights.png")
    plot_equity(curves['dates'], curves['equity'], out_png)

    eq = np.array(curves['equity'])
    returns = pd.Series(eq).pct_change().dropna()

    print("\n📈 Weights Backtest")
    print("Equity preview:", eq[:10])
    print("Min/Max equity:", eq.min(), eq.max())
    print("Returns preview:", returns[:10])
    print("Std:", returns.std())
    print(f"Sharpe   : {annualized_sharpe(returns):.2f}")
    print(f"CAGR     : {((eq[-1]/eq[0]) ** (252/len(eq)) - 1):.2%}")
    print(f"MaxDD    : {((eq / np.maximum.accumulate(eq)) - 1).min():.2%}")
    print(f"Final Eq : {eq[-1]:.2f}")

    # ---- SL/TP backtest ----
    dfp = df_te_valid.copy().reset_index(drop=True)
    W = curves['weights']  # Already forward-filled

    for i, s in enumerate(stock_list):
        dfp[f"{s}_weight"] = W[:, i]

    weight_cols = [f"{s}_weight" for s in stock_list]

    res_sl = backtest_sl_tp_per_asset(
        dfp, weight_cols, ret_cols, "dateTime",
        stride=F.horizon_steps,
        sl=B.stoploss/100,
        tp=B.takeprofit/100,
        ret_scale=100,
        redistribute=False
    )

    metrics_no = compute_trade_stats(res_sl["equity_no_stops"])
    metrics_w  = compute_trade_stats(res_sl["equity_with_stops"])

    print("\n📊 SL/TP Summary")
    print("No Stops :", metrics_no)
    print("Stops    :", metrics_w)

    plot_equity_sl(res_sl["dates"], res_sl["equity_no_stops"], res_sl["equity_with_stops"],
                title="Equity: Raw vs SL/TP")

    plot_underwater(res_sl["dates"], res_sl["equity_with_stops"])
    plot_rolling_sharpe(res_sl["dates"], res_sl["equity_with_stops"])
    plot_turnover(dfp, weight_cols, res_sl["dates"])

    print(f"\n✅ Finished in {time.time()-start:.1f}s")
if __name__ == "__main__": 
    main()