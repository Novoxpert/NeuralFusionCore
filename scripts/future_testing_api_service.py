#!/usr/bin/env python3
"""
future_testing_api_service.py
Author: Elham Esmaeilnia
Date: 2025 Nov 03
Version: 1.0.0
"""

from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from ..config import NOVOMongoCfg
import pandas as pd
import uvicorn

NMO = NOVOMongoCfg()
mongo_client = MongoClient(
    host=NMO.NOVO_MONGO_HOST,
    port=NMO.NOVO_MONGO_PORT,
    username=NMO.NOVO_MONGO_USER,
    password=NMO.NOVO_MONGO_PASS,
    authSource=getattr(NMO, "NOVO_MONGO_AUTH_DB", NMO.NOVO_MONGO_DB)
)
mongo_db = mongo_client[NMO.NOVO_MONGO_DB]
future_testing_col = mongo_db["AlphaFusionNet_future_testing"]

app = FastAPI(title="AlphaFusionNet Future Testing API")


@app.get("/future-testing/latest")
def get_latest_future_testing():
    """
    Fetch the latest saved future testing results from MongoDB.
    Returns:
        {
            "timestamp": <prediction timestamp>,
            "features": [...],
            "weights": {...},
            "created_at": <timestamp>
        }
    """
    doc = future_testing_col.find_one(sort=[("timestamp", -1)])
    if doc is None:
        raise HTTPException(status_code=404, detail="No future testing data found.")

    # Optional: convert features list back to DataFrame for API consumption
    features = pd.DataFrame(doc["features"]).to_dict(orient="records")

    return {
        "timestamp": str(doc["timestamp"]),
        "features": features,
        "weights": doc["weights"],
        "created_at": str(doc["created_at"])
    }
# -------------------- Run --------------------
if __name__ == "__main__":
    uvicorn.run("apps.NeuralFusionCore.scripts.future_testing_api_service:app", host="0.0.0.0", port=8005, reload=True)