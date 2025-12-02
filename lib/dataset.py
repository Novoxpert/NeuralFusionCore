
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader

class WeightDataset(Dataset):
    def __init__(self, df, seq_len, horizon_steps, feature_cols,data_stamp_cols, stock_list, news_vec_col='embedding', news_count_cols=None, inference=False):
        self.df = df.reset_index(drop=True)
        self.seq_len = seq_len
        self.h = horizon_steps
        self.fcols = feature_cols
        self.tmask = data_stamp_cols
        self.news_col = news_vec_col
        self.count_cols = news_count_cols or []
        self.stock_ret_cols = [f'{s}_target_return' for s in stock_list]
        self.df[self.news_col] = self.df[self.news_col].apply(
            lambda x: np.asarray(x, dtype='float32') if not isinstance(x, np.ndarray) else x
        )
        self.inference = inference
        if inference:
            # We can predict starting from seq_len history, no target needed
            self.N = max(len(self.df) - self.seq_len, 0)
        else:
            self.N = max(len(self.df) - self.seq_len - self.h, 0)

    def __len__(self): return self.N

    def __getitem__(self, idx):
        lo, hi = idx, idx + self.seq_len
        fut_lo, fut_hi = hi, hi + self.h
        X_ts = self.df.loc[lo:hi-1, self.fcols].astype('float32').values
        X_mask = self.df.loc[lo:hi-1, self.tmask].astype('float32').values
        X_news = np.stack(self.df.loc[lo:hi-1, self.news_col].values).astype('float32')
        if self.count_cols:
            X_cnt = self.df.loc[lo:hi-1, self.count_cols].astype('float32').values
        else:
            X_cnt = np.zeros((self.seq_len, 1), dtype='float32')
        if self.inference:
            return {
                    'timeseries': torch.tensor(X_ts),
                    'news': torch.tensor(X_news),
                    'news_count': torch.tensor(X_cnt),
                    'time_mask': torch.tensor(X_mask)
            }
        else:
            fut_lo, fut_hi = hi, hi + self.h
            Y = self.df.loc[fut_lo:fut_hi-1, self.stock_ret_cols].astype('float32').values
            return {
                    'timeseries': torch.tensor(X_ts),
                    'news': torch.tensor(X_news),
                    'news_count': torch.tensor(X_cnt),
                    'time_mask': torch.tensor(X_mask),
                    'target': torch.tensor(Y)
            }

def make_loaders(
        df_tr,
        df_va,
        df_te,
        seq_len,
        horizon_steps,
        feature_cols,
        data_stamp_cols,
        stock_list,
        news_count_cols,
        bs,
        inference_only: bool = False,
    ):
        """
        Create DataLoaders for train/val/test.

        - Training mode (inference_only=False):
            * Requires df_tr and df_va to be long enough.
            * Optionally builds test loader if df_te is provided.

        - Inference mode (inference_only=True):
            * Ignores df_tr and df_va completely.
            * Only builds a test loader from df_te with inference=True.
            * No length checks; if df_te is too short, the loader just has 0 batches.
        """

        # -----------------------
        # INFERENCE-ONLY MODE
        # -----------------------
        if inference_only:
            if df_te is None:
                raise ValueError("df_te must be provided when inference_only=True")

            ds_te = WeightDataset(
                df_te,
                seq_len,
                horizon_steps,
                feature_cols,
                data_stamp_cols,
                stock_list,
                news_count_cols=news_count_cols,
                inference=True,
            )
            te_loader = DataLoader(ds_te, batch_size=bs, shuffle=False)
            # No train/val loaders in prediction stage
            return None, None, te_loader

        # -----------------------
        # TRAINING MODE
        # -----------------------
        # Skip datasets that are too short
        if df_tr is None or len(df_tr) <= seq_len + horizon_steps:
            raise ValueError(
                f"Train dataset too short: {0 if df_tr is None else len(df_tr)} "
                f"<= seq_len + horizon_steps ({seq_len + horizon_steps})"
            )

        if df_va is None or len(df_va) <= seq_len + horizon_steps:
            raise ValueError(
                f"Val dataset too short: {0 if df_va is None else len(df_va)} "
                f"<= seq_len + horizon_steps ({seq_len + horizon_steps})"
            )

        ds_tr = WeightDataset(
            df_tr,
            seq_len,
            horizon_steps,
            feature_cols,
            data_stamp_cols,
            stock_list,
            news_count_cols=news_count_cols,
        )
        ds_va = WeightDataset(
            df_va,
            seq_len,
            horizon_steps,
            feature_cols,
            data_stamp_cols,
            stock_list,
            news_count_cols=news_count_cols,
        )

        # Only create test dataset if df_te is provided
        if df_te is not None:
            ds_te = WeightDataset(
                df_te,
                seq_len,
                horizon_steps,
                feature_cols,
                data_stamp_cols,
                stock_list,
                news_count_cols=news_count_cols,
                inference=True,
            )
            te_loader = DataLoader(ds_te, batch_size=bs, shuffle=False)
        else:
            te_loader = None

        return (
            DataLoader(ds_tr, batch_size=bs, shuffle=True),
            DataLoader(ds_va, batch_size=bs, shuffle=False),
            te_loader,
        )
