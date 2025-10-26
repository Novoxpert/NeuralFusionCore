# lib/redis_utils.py
"""
Description: redis helpers.
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Sep 30
Version: 1.0.1 
"""
import pickle
from redis import Redis
from apps.NeuralFusionCore.config import RedisCfg, Paths
import logging
import pandas as pd

RE = RedisCfg(); P = Paths()
redis_client = Redis(host=RE.host, port=RE.port, db=RE.db)

def get_all_redis_data_version1(symbols, start_time=None, end_time=None):
    """
    Return dict of symbol->df (unpickled) and news df (if exists)
    symbols: list of 'BINANCE:BTCUSDT' etc.
    """
    ohlcv = {}
    for s in symbols:
        v = redis_client.get(f"ohlcv:{s}")
        if v:
            try:
                df = pickle.loads(v)
                ohlcv[s] = df
            except Exception:
                ohlcv[s] = None
        else:
            ohlcv[s] = None
    news_v = redis_client.get("news")
    news_df = pickle.loads(news_v) if news_v else None
    return ohlcv, news_df

def get_all_redis_data(symbols, start_time=None, end_time=None):
    """
    Return dict of symbol -> df (unpickled and filtered by time) and news df (if exists).

    Args:
        symbols (list): list of 'BINANCE:BTCUSDT' etc.
        start_time (datetime, optional): filter rows >= start_time
        end_time (datetime, optional): filter rows <= end_time

    Returns:
        ohlcv (dict): symbol -> filtered DataFrame
        news_df (DataFrame or None)
    """
    ohlcv = {}
    for s in symbols:
        v = redis_client.get(f"ohlcv:{s}")
        if v:
            try:
                df = pickle.loads(v)
                # Ensure dateTime column is datetime
                if not df.empty and 'dateTime' in df.columns:
                    df['dateTime'] = pd.to_datetime(df['dateTime'])
                    if start_time:
                        df = df[df['dateTime'] >= start_time]
                    if end_time:
                        df = df[df['dateTime'] <= end_time]
                ohlcv[s] = df
            except Exception as e:
                logging.warning(f"Error loading Redis data for {s}: {e}")
                ohlcv[s] = None
        else:
            ohlcv[s] = None

    # News
    news_v = redis_client.get("news")
    news_df = pickle.loads(news_v) if news_v else None
    if news_df is not None:
        news_df['releasedAt'] = pd.to_datetime(news_df['releasedAt'])
        if start_time:
            news_df = news_df[news_df['releasedAt'] >= start_time]
        if end_time:
            news_df = news_df[news_df['releasedAt'] <= end_time]

    return ohlcv, news_df


def atomic_model_swap(src_path, dest_path):
    """
    Atomically replace dest_path with src_path (works locally). You might want to do
    object storage / tagging for distributed envs. This is a simple approach.
    """
    import os, shutil
    bak = dest_path + ".bak"
    if os.path.exists(dest_path):
        shutil.move(dest_path, bak)
    shutil.move(src_path, dest_path)
    if os.path.exists(bak):
        os.remove(bak)
