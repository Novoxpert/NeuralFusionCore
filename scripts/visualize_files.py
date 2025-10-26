"""
visualize_files.py
Description: Visualize pickle files for monitoring.
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Sep 24
Version:1.0.0
"""
import pandas as pd
import dtale
import time

# Load your pickle file
df = pd.read_pickle("data/outputs/df_portfolio.pickle")

# Start DTale server
d = dtale.show(df, host='0.0.0.0', port=40000, subprocess=False)

print("\nDTale is running!")
print("Open your browser and go to:")
print("👉 http://127.0.0.1:40000")
print("👉 or http://localhost:40000\n")

# Keep the server alive
while True:
    time.sleep(1)
