"""
Description: View Parquet or PKL files in a spreadsheet-like GUI inside VSCode/Streamlit.
Author:  Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 23 Sep 2025
Version: 1.1.0
"""
import streamlit as st
import pandas as pd
from pathlib import Path

# --- Locate the file ---
script_dir = Path(__file__).parent
parquet_file = script_dir / "online_bridge.parquet"
pkl_file = script_dir / "online_bridge.pkl"

if parquet_file.exists():
    file_path = parquet_file
elif pkl_file.exists():
    file_path = pkl_file
else:
    st.error("No Parquet or PKL file found in this folder!")
    st.stop()

# --- Load data ---
if file_path.suffix == ".parquet":
    df = pd.read_parquet(file_path)
else:  # .pkl
    import pickle
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
        # If dict with 'df_news' key, use that
        if isinstance(obj, dict) and "df_news" in obj:
            df = obj["df_news"]
        else:
            df = pd.DataFrame(obj)

# --- Preview options ---
st.title(f"Preview of {file_path.name}")
st.write(f"Data shape: {df.shape}")

# Let user select columns (exclude huge columns like embeddings by default)
all_cols = df.columns.tolist()
default_cols = [c for c in all_cols if "embedding" not in c]
selected_cols = st.multiselect("Select columns to display", options=all_cols, default=all_cols)

# Limit number of rows to avoid browser overload
max_rows = st.slider("Number of rows to preview", min_value=100, max_value=5000, value=1000, step=100)

st.dataframe(df[selected_cols].head(max_rows))
