# NeuralFusionCore

## Direct Portfolio Weight Forecasting with (News + OHLCV) Cross‑Gated Attention Fusion

This variant directly forecasts **portfolio weights** using multi‑modal inputs (news + OHLCV) and fuses the streams with **Cross‑Gated Attention (CGA)**.  
CGA lets each stream attend to the other via gates that modulate information flow, improving robustness over naive concatenation.

---
## 📋 Table of Contents

- [Direct Portfolio Weight Forecasting](#direct-portfolio-weight-forecasting-with-news--ohlcv-cross-gated-attention-fusion)
- [🏗️ Architecture Overview](#-architecture-overview)
  - [Encoders](#encoders)
  - [Fusion — Cross-Gated Attention (CGA)](#fusion--cross-gated-attention-cga)
  - [Output Head](#output-head)
- [🎯 Training Objective](#training-objective)
  - [Portfolio Weighting (Top-k Long/Short)](#portfolio-weighting-top-k-longshort)
  - [Sharpe Ratio Loss](#sharpe-ratio-loss-maximize-risk-adjusted-return)
  - [Regularization Terms](#regularization-terms)
  - [Total Loss](#total-loss)
- [📁 Repository Layout](#-repository-layout-exact)
- [⚙️ Setup](#-setup)
- [🧩 Script Cheat-Sheet](#-script-cheat-sheet)
- [🚀 Pipeline (Direct Weights)](#-pipeline-direct-weights)
- [📦 Dependencies](#dependencies)
- [📊 Outputs](#outputs)
- [🧠 Notes](#notes)
- [📚 Appendix](#appendix)
  - [Upstream Repositories](#upstream-repositories)
  - [Inspiration](#inspiration)
- [👥 Authors & Citation](#-authors--citation)
- [📞 Support](#-support)
---
## 🏗️ Architecture Overview

- **Timeframe:** 3‑minute bars  
- **Input Window:** 80 timestamps (~4 hours)  
- **Prediction Horizon:** next 80 timestamps (~4 hours)  
- **Assets:** configurable universe

### Encoders

1. **News stream (single LSTM)**  
   - Each article → **BigBird embedding**  
   - **Average embeddings** of all articles per 3-min window  
   - If no news: use learned **[NO_NEWS]** embedding  
   - **Coverage one-hot** (which stocks are mentioned) is concatenated to the news embedding at each timestamp  
   - The sequence is fed to **one LSTM** → produces news sequence embedding  

2. **OHLCV stream (TimesNet)**  
   - A **TimesNetBlock** processes per-asset OHLCV sequences → produces market embedding  

---

### Fusion — Cross‑Gated Attention (CGA)

- Let **N** be the news embedding and **M** the market (OHLCV) embedding  
- Compute **cross‑attention** in both directions (N→M and M→N)  
- Apply **gates** (sigmoid/tanh) to the attended features before adding residuals:

```math
\tilde{N} = g_N \odot \text{Attn}(N \rightarrow M) + (1-g_N) \odot N
````

```math
\tilde{M} = g_M \odot \text{Attn}(M \rightarrow N) + (1-g_M) \odot M
```

* Concatenate or sum $\tilde{N}$ and $\tilde{M}$ to form the **fused embedding**

---

### Output Head

* A linear layer maps the fused embedding to **portfolio weights** for all assets

---

# Training Objective

The model uses a **top-k long/short portfolio construction** and optimizes a **risk-adjusted return loss with regularization**.

Let:

- $w \in \mathbb{R}^N$ be the portfolio weights computed from logits  
- $R \in \mathbb{R}^{H \times N}$ be the returns matrix for a batch (H time steps, N assets)  
- $k$ be the number of assets to select for active trading  
- $\epsilon$ a small constant for numerical stability  

---
## Portfolio Weighting (Top-k Long/Short)

Weights are computed as:

$$
w = \text{apply}(\text{logits}, k)
$$

where the function `topk_long_short_abs` selects the top-k absolute logits and normalizes them.

Only the top-k largest absolute values of logits are selected, and the weights are normalized:

$$
w_i =
\begin{cases}
\dfrac{\text{sign}(\text{logits}_i) \cdot |\text{logits}_i|}{\sum_{j \in \text{top-k}} |\text{logits}_j|}, & i \in \text{top-k} \\
0, & \text{otherwise}
\end{cases}
$$

---

## Sharpe Ratio Loss (maximize risk-adjusted return)

Portfolio returns:

$$
R_p = \sum_{i=1}^{N} w_i \cdot R_i
$$

Sharpe ratio:

$$
\text{Sharpe} = \frac{\mathbb{E}[R_p]}{\sqrt{\text{Var}(R_p) + \epsilon}}
$$

Sharpe loss:

$$
\mathcal{L}_{\text{Sharpe}} = - \text{Sharpe}
$$

---

## Regularization Terms

1. **Distribution regularizer** (prevents concentration):

$$
\mathcal{L}_{\text{dist}} = \lambda_{\text{div}} \cdot \frac{1}{N} \sum_{i=1}^N w_i^2
$$

2. **Net exposure regularizer** (encourages market-neutral portfolio):

$$
\mathcal{L}_{\text{net}} = \lambda_{\text{net}} \cdot \left(\sum_{i=1}^N w_i \right)^2
$$

3. **Turnover regularizer** (optional, penalizes large changes in weights):

$$
\mathcal{L}_{\text{turnover}} = \lambda_{\text{turnover}} \cdot \sum_{i=1}^N | w_i - w_i^{\text{prev}} |
$$

---

## Total Loss

The total loss optimized:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Sharpe}} + \mathcal{L}_{\text{dist}} + \mathcal{L}_{\text{net}} + \mathcal{L}_{\text{turnover}}
$$

- Only the turnover regularizer (`L_turnover`) is applied if previous weights `w_prev` are provided.
- $\lambda_{\text{div}}, \lambda_{\text{net}}, \lambda_{\text{turnover}}$ control the regularization strength.
---

## 📁 Repository Layout (exact)
```
NeuralFusionCore/
     ├── data/
     │   ├── outputs/
     │   │   └── model_weights.pt        
     │   └── processed/
     │       └── show_files.py                   
     │   
     ├── lib/
     │   ├── backtest.py
     │   ├── backtest_weights.py        
     │   ├── dataset.py
     │   ├── features.py
     │   ├── loss_weights.py            
     │   ├── market.py
     │   ├── model.py
     │   ├── news.py
     │   ├── redis_utils.py
     │   ├── train.py
     │   └── utils.py
     ├──_init__.py
     ├── README.md
     ├── requirements.txt
     ├── config.py
     └── scripts/
          ├── data_ingest_service.py
          ├── features_service.py
          ├── train_service.py
          ├── finetune_service.py
          ├── prediction_service.py 
          └── api_service.py

```
> Any folders missing on your machine will be created by the scripts if needed.

---

## ⚙️ Setup

```bash

# Clone repository
git clone https://github.com/Novoxpert/NeuralFusionCore.git
cd NeuralFusionCore


# (optional) create a virtual environment
python -m venv .venv

# Linux/macOS:
source .venv/bin/activate

# Windows (PowerShell):
 .\.venv\Scripts\Activate.ps1

# install exact dependencies
pip install -r requirements.txt
```

---
## 🧩 Script Cheat‑Sheet

- **`lib/*.py`** — internal modules for datasets, models, features, news embeddings,training loops, utilities, and backtesting specialized for direct weights.  
- **`config.py`** — central configuration / argument helpers used by the scripts.
- **`scripts/data_ingest_service.py`** — fetch OHLCV from ClickHouse and news from Mongo for the given interval, and push results (per-symbol ohlcv DataFrame pickles and news DataFrame) to Redis.

🔧 Usage examples:
one-shot latest 4h (use scheduler to run every 4h)
```bash
python -m scripts.data_ingest_service --mode latest --hours 4
```
- **`scripts/features_service.py`** — Builds features from Redis.

Modes:
  - train:     full rebuild (includes normalizer + meta)
  - finetune:  incremental build (reuse existing normalizer/meta)
  - inference: build features for inference only (produces online_test.parquet)
  - bridge:    build features for ChronobBridge only
  - time:      select data by start_time/end_time for any mode

🔧 Usage Examples:
```bash
python -m scripts.features_service --mode finetune --latest_hours 24
```
- **`scripts/train_service.py`** — Train from scratch on processed/train.parquet and processed/val.parquet
🔧 Usage Example:
```bash
python -m scripts.train_service --epocha 50 
```
- **`scripts/finetune_service.py`** —Fine-tune an existing saved model using the latest features. If validation loss improves, replace saved model and keep previous version with timestamp.

🔧 Usage Example:
```bash
python -m scripts.finetune_service --epocha 10 --save_best
```
- **`scripts/prediction_service.py`** —Scheduled inference: fetch latest data, compute features, infer model, transform logits into portfolio weights, and save predictions to MongoDB and Redis.

🔧 Usage Example:
```bash
python -m scripts.prediction_service --hours 4 
```
- **`scripts/api_service.py`** — create API for Get NeuralFusion weights from Mongodb.
---
## 🚀 Pipeline (Direct Weights)

#### 1) run data_ingest_service
#### 2) run features_service
#### 3) run train_service
#### 4) run prediction_service
---

## Dependencies

* Python 3.12+
* PyTorch 2.x
* Hugging Face `transformers` (BigBird)
---

## Outputs

* Predicted weights per timestamp
* Performance metrics:

  * Sharpe ratio
  * Cumulative P&L
  * Max Drawdown
  * Turnover
* Plots: equity curve, rolling Sharpe, weights heatmap

---

## Notes

* **CGA** allows **directional, gated cross‑attention** between news and market signals
* The **distribution loss** helps prevent one-asset collapse
* **Total Loss** combines Sharpe ratio maximization and regularizations

---

## Appendix

### Upstream Repositories

Influential upstream repositories:

* [**BigBird**](https://github.com/google-research/bigbird): A sparse-attention transformer model enabling efficient processing of longer sequences
* [**finBERT**](https://github.com/ProsusAI/finBERT): A pre-trained NLP model fine-tuned for financial sentiment analysis
* [**Time-Series-Library (TSlib)**](https://github.com/thuml/Time-Series-Library): Library providing deep learning-based time series analysis, covering forecasting, anomaly detection, and classification
---
### Inspiration

This work is inspired by the article:

* [**Stock Movement Prediction with Multimodal Stable Fusion via Gated Cross-Attention Mechanism**](https://arxiv.org/abs/2406.06594): Introduces the Multimodal Stable Fusion with Gated Cross-Attention (MSGCA) architecture, designed to robustly integrate multimodal inputs for stock movement prediction.
---
## 👥 Authors & Citation

**Developed by the [Novoxpert Research Team](https://github.com/Novoxpert)**  
Lead Contributors:
 - [Elham Esmaeilnia](https://github.com/Elham-Esmaeilnia), [Hamidreza Naeini](https://github.com/)
 

If you use this repository or build upon our work, please cite:

> Novoxpert Research (2025). *NeuralFusionCore: Direct Portfolio Weight Forecasting with Cross-Gated Attention Fusion.*  
> GitHub: [https://github.com/Novoxpert/NeuralFusionCore](https://github.com/Novoxpert/NeuralFusionCore)

```bibtex
@software{novoxpert_neuralfusioncore_2025,
  author       = {Elham Esmaeilnia and Hamidreza Naeini},
  title        = {NeuralFusionCore: Direct Portfolio Weight Forecasting with Cross-Gated Attention Fusion},
  organization = {Novoxpert Research},
  year         = {2025},
  url          = {https://github.com/Novoxpert/NeuralFusionCore}
}
```
---
## 📞 Support

- **Issues & Bugs**: [Open on GitHub](https://github.com/Novoxpert/neuralfusioncore/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Novoxpert/neuralfusioncore/discussions)
- **Feature Requests**: Open a feature request issue
---