#!/usr/bin/env python3
"""
api_service.py
--------------
Description: FastAPI service to serve latest portfolio predictions from MongoDB.
             Supports CORS for React frontend and serializes MongoDB BSON types.
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Oct 05
Version: 1.1.0
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from bson import ObjectId, json_util
from datetime import datetime, timezone
import uvicorn
import json

# -------------------- App & CORS --------------------
app = FastAPI(title="Portfolio API")

origins = [
    "http://localhost:3000", "http://192.168.20.82:3000" # React dev server
    # Add production frontend origin if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- MongoDB --------------------
client = MongoClient("mongodb://127.0.0.1:27017/")
db = client["portfolio_db"]
collection = db["NeuralFusionCore_predictions"]

# -------------------- Helpers --------------------
def serialize_doc(doc):
    """Convert MongoDB BSON doc to JSON-serializable dict"""
    if not doc:
        return None
    # Use bson.json_util to handle ObjectId and datetime
    safe_json = json.loads(json_util.dumps(doc))
    # Optional: flatten $date and $oid
    def flatten(obj):
        if isinstance(obj, dict):
            if "$oid" in obj:
                return obj["$oid"]
            if "$date" in obj:
                return obj["$date"]
            return {k: flatten(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [flatten(i) for i in obj]
        return obj
    return flatten(safe_json)

# -------------------- Endpoints --------------------
@app.get("/latest_prediction")
def latest_prediction():
    """Return the most recent prediction"""
    doc = collection.find_one(sort=[("ts", -1)])
    if not doc:
        return JSONResponse(content={"error": "No prediction found"}, status_code=404)
    return JSONResponse(content=serialize_doc(doc))

@app.get("/prediction_history")
def prediction_history(
    start: str = Query(None, description="Start ISO timestamp, e.g. 2025-10-01T00:00:00"),
    end: str = Query(None, description="End ISO timestamp"),
    limit: int = 100
):
    """Return historical predictions within a time range"""
    query = {}
    if start:
        query["ts"] = {"$gte": datetime.fromisoformat(start)}
    if end:
        query.setdefault("ts", {})["$lte"] = datetime.fromisoformat(end)

    cursor = collection.find(query).sort("ts", 1).limit(limit)
    docs = [serialize_doc(d) for d in cursor]
    return JSONResponse(content=docs)

@app.get("/health")
def health_check():
    """Simple health check endpoint"""
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}

# -------------------- Run --------------------
if __name__ == "__main__":
    uvicorn.run("NeuralFusionCore.scripts.api_service:app", host="0.0.0.0", port=8000, reload=True)
