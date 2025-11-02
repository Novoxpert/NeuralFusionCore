import os, numpy as np, pandas as pd
from .loss_weights import weights_long_short_topk_abs
import torch
from apps.NeuralFusionCore.config import MarketCfg
from apps.NeuralFusionCore.config import Paths
P = Paths();

def backtest_weight_logits(pred_logits, returns_matrix, dates, k=30, gross=1.0, stride=80):
    """
    Compute portfolio equity curve from model logits.
    
    Args:
        pred_logits: (M, N) numpy array of model predictions per timestep per stock
        returns_matrix: (T, N) numpy array of asset returns
        dates: pd.Series or np.array of timestamps
        k: top-K stocks to long/short
        gross: leverage factor
        stride: forward-fill stride for weights
    
    Returns:
        dict with 'dates', 'equity', 'rp' (portfolio returns), and 'weights'
    """
    symbols = MarketCfg().symbols_usdt
    M, N = pred_logits.shape
    T = returns_matrix.shape[0]
    assert T == len(dates), "Returns matrix and dates length mismatch"

    # --- Compute weights ---
    with torch.no_grad():
        logits_t = torch.tensor(pred_logits, dtype=torch.float32)
        w_t = weights_long_short_topk_abs(logits_t, k=k, gross=gross).cpu().numpy()

    # --- Forward-fill weights for each stride ---
    w_series = np.zeros((T, N), dtype=float)
    idxs = list(range(0, T, stride))
    for j, t0 in enumerate(idxs):
        w_series[t0:, :] = w_t[min(j, M-1)]

    # --- Portfolio returns & equity ---
    rp = (w_series * returns_matrix).sum(axis=1)  # weighted portfolio return per timestep

    # Shift equity to start at 1 for stable Sharpe/CAGR calculation
    equity = rp.cumsum()
    equity = equity - equity[0] + 1

    # --- Save portfolio DataFrame ---
    df_portfolio = pd.DataFrame({'dateTime': dates})
    for i, sym in enumerate(symbols):
        df_portfolio[f'{sym}_return'] = returns_matrix[:, i]
        df_portfolio[f'{sym}_weight'] = w_series[:, i]

    out_path = os.path.join(P.outputs_dir, 'df_portfolio.pickle')
    os.makedirs(P.outputs_dir, exist_ok=True)
    df_portfolio.to_pickle(out_path)

    return {'dates': dates, 'equity': equity, 'rp': rp, 'weights': w_series}
