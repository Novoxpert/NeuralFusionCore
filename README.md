# NeuralFusionCore

---

# Direct Portfolio Weight Forecasting with News + OHLCV (Cross‑Gated **Attention** Fusion)

This variant directly forecasts **portfolio weights** using multi‑modal inputs (news + OHLCV) and fuses the streams with **Cross‑Gated Attention (CGA)**.  
CGA lets each stream attend to the other **via gates** that modulate information flow, improving robustness compared to naive concatenation.

---

## Architecture Overview

- **Timeframe:** 3‑minute bars  
- **Input Window:** 80 timestamps (≈ 4 hours)  
- **Prediction Horizon:** next 80 timestamps (≈ 4 hours)  
- **Assets:** configurable universe

### Encoders

1. **News stream (single LSTM)**  
   - Each article → **BigBird** embedding  
   - Per bar: **average** embeddings of all articles in that 3‑min window  
   - If no news: use learned **[NO_NEWS]** embedding  
   - **Coverage one‑hot** (which stocks are mentioned) is **concatenated** to the news embedding at each timestamp  
   - The combined sequence is fed to **one LSTM** → produces news sequence embedding  

2. **OHLCV stream (TimesNet)**  
   - A **TimesNetBlock** processes per‑asset OHLCV sequences → produces market embedding  

---

### Fusion — Cross‑Gated **Attention** (CGA)

- Let **N** be the news embedding and **M** the market (OHLCV) embedding.  
- Compute **cross‑attention** in both directions (N→M and M→N).  
- Apply **gates** (sigmoid/tanh) to the attended features before adding residuals:

$$
\tilde{N} = g_N \odot \text{Attn}(N \rightarrow M) + (1-g_N) \odot N
$$

$$
\tilde{M} = g_M \odot \text{Attn}(M \rightarrow N) + (1-g_M) \odot M
$$

- Concatenate or sum \(\tilde{N}\) and \(\tilde{M}\) to form the **fused embedding**.

---

### Output Head

- A linear layer maps the fused embedding to **portfolio weights** for all assets.

---

## Training Objective

Two-term loss:

1. **Sharpe Ratio Loss** (maximize risk-adjusted return):

$$
\mathcal{L}_{\text{Sharpe}} = - \frac{\mathbb{E}[R_p]}{\sqrt{\operatorname{Var}(R_p) + \epsilon}}
$$

2. **Distribution Regularizer** on weights (prevents concentration):  
   e.g., negative entropy, L2 concentration, or KL divergence to uniform

$$
\mathcal{L}_{\text{dist}} = \lambda \cdot f(w_t)
$$

**Total Loss:**

$$
\mathcal{L} = \mathcal{L}_{\text{Sharpe}} + \mathcal{L}_{\text{dist}}
$$

---

## Portfolio Construction (Test Phase)

- Use **predicted weights** directly  
- Optional **top‑k selection**:  
  1. Keep top‑k assets by absolute weight  
  2. Re-normalize to sum to 1 (long-only) or L1=1 (long/short)

---

## How to Run

1. Prepare data:  
   - **OHLCV** (3‑min bars) aligned across assets  
   - **News** table with text + **asset coverage one‑hot**  
2. Open the notebook for this CGA variant and configure paths/hyperparameters  
3. Train:  
   - Select best checkpoint by **validation Sharpe**  
4. Test:  
   - Predict weights across test period  
   - Apply **top‑k** if desired  
   - Evaluate portfolio metrics  

---

## Dependencies

- Python 3.10+  
- PyTorch 2.x  
- Hugging Face `transformers` (BigBird)  
- numpy, pandas, matplotlib, scikit-learn  

Install:
```bash
   pip install torch transformers numpy pandas matplotlib scikit-learn
```
---

## Outputs

- Predicted weights per timestamp
- Performance metrics:
  - Sharpe ratio
  - Cumulative P&L
  - Max Drawdown
  - Turnover
- Plots: equity curve, rolling Sharpe, weights heatmap

---

## Notes

- **CGA** allows **directional, gated cross‑attention** between news and market signals.
- The **distribution loss** helps prevent one‑asset collapse.

---

