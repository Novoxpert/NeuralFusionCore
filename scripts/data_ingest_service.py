"""
data_ingest_service.py
Stateless fetcher: fetch OHLCV from ClickHouse and news from Mongo for the given interval,
and push results (per-symbol ohlcv DataFrame pickles and news DataFrame) to Redis.
Usage examples:
  # historical: fetch last 30 days and save to disk (also push latest window to Redis)
  python data_ingest_service.py --mode historical --days 30 --save_dir data/ohlcv_history

  # one-shot latest 4h (use scheduler to run every 4h)
  python data_ingest_service.py --mode latest --hours 4
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Sep 30
Version: 1.0.1 
"""
import argparse, logging, os, pickle
from datetime import datetime, timedelta, timezone
import pandas as pd
from clickhouse_driver import Client as CHClient
from pymongo import MongoClient
from redis import Redis
from ..config import ClickhouseCfg, MongoCfg, RedisCfg, Paths, MarketCfg, FeatureCfg
from ..lib import market as M
import time

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

CH = ClickhouseCfg(); MO = MongoCfg(); RE = RedisCfg(); P = Paths(); MC = MarketCfg(); FC = FeatureCfg()

ch_client = CHClient(host=CH.CH_HOST, port=CH.CH_PORT, user=CH.CH_USER,
                     password=CH.CH_PASS, database=CH.CH_DB)
mongo_client = MongoClient(host=MO.MONGO_HOST, port=MO.MONGO_PORT,
                          username=MO.MONGO_USER, password=MO.MONGO_PASS,
                          authSource=getattr(MO, 'MONGO_AUTHSOURCE', MO.MONGO_DB))
redis_client = Redis(host=RE.host, port=RE.port, db=RE.db)

def fetch_ohlcv_range(symbol, start_utc, end_utc):
    start_str = start_utc.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    end_str = end_utc.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    q = f"""
        SELECT *
        FROM {CH.CH_TABLE}
        WHERE symbol = '{symbol}'
          AND candle_time >= '{start_str}'
          AND candle_time <= '{end_str}'
        ORDER BY candle_time ASC
    """
    data = ch_client.execute(q)
    if not data:
        return pd.DataFrame()
    cols = [c[0] for c in ch_client.execute(f"DESCRIBE TABLE {CH.CH_TABLE}")]
    df = pd.DataFrame(data, columns=cols)
    df['dateTime'] = pd.to_datetime(df['candle_time'], utc=True)
    df3m = M.resample_to_3m(df, FC.agg_cols)
    return df3m

def fetch_news_range(start_utc, end_utc):
    col = mongo_client[MO.MONGO_DB][MO.MONGO_COLLECTION]
    cursor = col.find({"releasedAt": {"$gte": start_utc, "$lte": end_utc}})
    df = pd.DataFrame(list(cursor))
    if df.empty:
        return df
    df['releasedAt'] = pd.to_datetime(df['releasedAt'], utc=True)
    return df

def push_ohlcv_to_redis(sym, df):
    if df is None or df.empty:
        return
    key = f"ohlcv:{sym}"
    redis_client.set(key, pickle.dumps(df))
    logging.info("Pushed to redis: %s (%d rows)" % (key, len(df)))

def push_news_to_redis(df):
    if df is None or df.empty:
        return
    redis_client.set("news", pickle.dumps(df))
    logging.info("Pushed news to redis (%d rows)" % len(df))

def main():
    start_service_time= time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["historical", "latest"], required=True)
    parser.add_argument("--days", type=int, default=30, help="for historical mode")
    parser.add_argument("--hours", type=int, default=4, help="for latest mode")
    parser.add_argument("--save_dir", type=str, default=None, help="optional dir to save historical files")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    if args.mode == "historical":
        start = now - timedelta(days=args.days)
        logging.info(f"Fetching historical data from {start} to {now}")
        # For each symbol, fetch entire interval, save optionally and push latest window to Redis
        #symbols_ch = [f"BINANCE:{s}" for s in MC.symbols_usdt]
        symbols_ch = [f"{s}" for s in MC.symbols_usdt]
        for sym in symbols_ch:
            df_all = fetch_ohlcv_range(sym, start, now)
            if args.save_dir and not df_all.empty:
                os.makedirs(args.save_dir, exist_ok=True)
                fname = os.path.join(args.save_dir, f"{sym.replace(':','_')}.parquet")
                df_all.to_parquet(fname, index=False)
                logging.info(f"Saved historical {sym} -> {fname}")
            # push only last args.hours to redis for immediate consumption
            last_window = df_all.set_index('dateTime').last(f"{args.hours}H").reset_index() if not df_all.empty else pd.DataFrame()
            if not last_window.empty:
                push_ohlcv_to_redis(sym, last_window)
        # fetch & save news
        df_news = fetch_news_range(start, now)
        if args.save_dir and not df_news.empty:
            os.makedirs(args.save_dir, exist_ok=True)
            df_news.to_parquet(os.path.join(args.save_dir, "news.parquet"), index=False)
        push_news_to_redis(df_news)
        logging.info("Historical ingest complete.")
        end_service_time = time.time()
        print(f"Time elapsed for data ingestion service: {end_service_time - start_service_time:.2f} seconds")
        return

    # latest mode (one-shot)
    if args.mode == "latest":
        start = now - timedelta(hours=args.hours)
        #symbols_ch = [f"BINANCE:{s}" for s in MC.symbols_usdt]
        symbols_ch = [f"{s}" for s in MC.symbols_usdt]
        for sym in symbols_ch:
            df = fetch_ohlcv_range(sym, start, now)
            if not df.empty:
                push_ohlcv_to_redis(sym, df)
        df_news = fetch_news_range(start, now)
        if not df_news.empty:
            push_news_to_redis(df_news)
        logging.info("Latest ingest complete.")
        end_service_time = time.time()
        print(f"Time elapsed for data ingestion service: {end_service_time - start_service_time:.2f} seconds")
        return
    

if __name__ == "__main__":
    main()