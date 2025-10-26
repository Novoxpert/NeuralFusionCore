import os, numpy as np, pandas as pd, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def add_onehot_columns(df_news: pd.DataFrame, tradable_symbols: list) -> pd.DataFrame:
    """
    Convert `assets` column (list of dicts) to one-hot columns for all tradable symbols.
    Only symbols in tradable_symbols are used; missing symbols will have 0.
    """
    #syms = [s.replace('USDT','') for s in tradable_symbols]
    #syms = [s.replace('BINANCE:','') for s in tradable_symbols]
    syms = tradable_symbols

    def parse_symbols(asset_list):
        if not isinstance(asset_list, list):
            return []
        return [a['symbol'] for a in asset_list if isinstance(a, dict) and 'symbol' in a and a['symbol'] in syms]

    # safely handle missing / non-list entries
    df_news["asset_symbols"] = df_news["assets"].apply(lambda x: x if isinstance(x, list) else []).apply(parse_symbols)

    # create one-hot
    onehot = df_news["asset_symbols"].explode().str.get_dummies().groupby(level=0).sum()

    # ensure all tradable symbols columns exist
    for s in syms:
        if s not in onehot.columns:
            onehot[s] = 0
    onehot = onehot[syms]

    df_news = df_news.drop(columns=[c for c in onehot.columns if c in df_news.columns], errors='ignore')
    df_news = df_news.join(onehot)

    return df_news.reset_index(drop=True)



def load_text_encoder(model_path: str, device: str = None, max_len: int = 2048):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_path)
    #Edited by "Elham Esmaeilnia" (2025 sep 7): 
    #mdl = AutoModel.from_pretrained(model_path).to(device).eval()
    mdl = AutoModel.from_pretrained(model_path).to(device).eval()
    return tok, mdl, device, max_len

@torch.no_grad()
def embed_texts(texts, tok, mdl, device, max_len=2048, pooling="mean", batch_size=32):
    
    out = []
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding news"):
        batch = texts[i:i+batch_size]
        #Edited by "Elham Esmaeilnia" (2025 Sep 8). Clean batch: force everything to string, replace None/NaN with ""
        def clean_texts(batch):
            cleaned = []
            for t in batch:
                if t is None or (isinstance(t, float) and np.isnan(t)):
                    cleaned.append(None)  # mark invalid
                elif not isinstance(t, str):
                    t = str(t)
                    if not t.strip():
                        cleaned.append(None)
                    else:
                        cleaned.append(t.strip())
                else:
                    cleaned.append(t.strip() if t.strip() else None)
            return cleaned
        #batch = clean_texts(texts[i:i+batch_size])
        #enc = tok(clean_batch, truncation=True, padding=True, max_length=max_len, return_tensors="pt").to(device)
        enc = tok(batch, truncation=True, padding=True, max_length=max_len, return_tensors="pt").to(device)
        hs = mdl(**enc).last_hidden_state
        if pooling == "mean":
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = hs[:, 0, :]
        out.append(pooled.detach().cpu().numpy())
        if device == "cuda": torch.cuda.empty_cache()
    return np.vstack(out)


def resample_news_3m(df_news: pd.DataFrame, no_news_vec: np.ndarray, rule: str = "3min") -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    # Ensure no_news_vec is float32
    no_news_vec = np.asarray(no_news_vec, dtype=np.float32)

    df = df_news.copy()
    df["releasedAt"] = pd.to_datetime(df["releasedAt"], errors="coerce")
    df = df.dropna(subset=["releasedAt"])
    df["t3"] = df["releasedAt"].dt.floor(rule)

    base = {"releasedAt", "content", "embedding", "news_count", "asset_symbols", "t3"}
    asset_cols = [c for c in df.columns if c not in base and df[c].dtype != "O"]

    if "news_count" not in df.columns:
        df["news_count"] = 1

    g = df.groupby("t3", sort=True)
    agg = g[asset_cols + ["news_count"]].sum()

    # Function to average embeddings safely
    def _avg(arr):
        arr = [np.asarray(x, dtype=np.float32) for x in arr if isinstance(x, (list, np.ndarray))]
        if not arr:
            return np.zeros_like(no_news_vec, dtype=np.float32)
        shapes = {a.shape for a in arr}
        if len(shapes) != 1:
            raise ValueError(f"Embedding length mismatch {shapes}")
        return np.mean(np.stack(arr, axis=0), axis=0).astype(np.float32)

    emb = g["embedding"].apply(_avg)
    news_3m = agg.join(emb.rename("embedding"))

    # Reindex to regular 3-minute intervals
    idx = pd.date_range(news_3m.index.min(), news_3m.index.max(), freq=rule)
    news_3m = news_3m.reindex(idx)

    # Fill missing values in numeric columns
    news_3m["news_count"] = news_3m["news_count"].fillna(0).astype(int)
    for c in asset_cols:
        news_3m[c] = news_3m[c].fillna(0).astype(int)

    # Fill missing embeddings with no_news_vec (ensuring float32)
    mask = news_3m["embedding"].isna()
    news_3m.loc[mask, "embedding"] = news_3m.loc[mask, "embedding"].apply(
        lambda _: np.copy(no_news_vec)
    )

    # Ensure all embeddings are float32 (in case some were different)
    news_3m["embedding"] = news_3m["embedding"].apply(lambda x: np.asarray(x, dtype=np.float32))

    news_3m.index.name = "t3"
    return news_3m
