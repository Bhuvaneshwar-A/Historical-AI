from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
import polars as pl
from data_proessor import DataProcessor
from typing import List, Dict, Optional
import polars as pl
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timedelta
import pytz
from typing import List, Dict, Any, Tuple, Optional, Iterator
from pymongo import ReplaceOne, ASCENDING, DESCENDING
import asyncio
import numpy as np
from collections import deque
from datetime import timezone
import orjson
from functools import lru_cache
import io
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# data_processor = DataProcessor("mongodb://shamlaTech:ShamlaSts023@localhost:27017/HistoricalAI")
data_processor = DataProcessor("mongodb://localhost:27017")

# Initialize data processor on startup
@app.on_event("startup")
async def startup_event():
    await data_processor.initialize()

# Pydantic models
class CategoryCreate(BaseModel):
    category: str = Field(..., description="Name of the category to create")

class SymbolAdd(BaseModel):
    symbol: str = Field(..., description="Symbol to add to the category")

class PatternSearchRequest(BaseModel):
    category: str = Field(..., description="Category containing the symbol")
    symbol: str = Field(..., description="Trading symbol to search patterns for")
    start_date: datetime = Field(..., description="Start date for historical data")
    end_date: datetime = Field(..., description="End date for historical data")
    sample_data: List[float] = Field(..., description="Sample price data to find patterns for")
    ema_period: int = Field(default=14, description="Period for EMA calculation")
    include_volume: bool = Field(default=False, description="Whether to include volume in pattern matching")

# Category management endpoints
@app.post("/categories")
async def create_category(category_data: CategoryCreate):
    result = await data_processor.add_category(category_data.category)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.delete("/categories/{category}")
async def delete_category(category: str):
    result = await data_processor.delete_category(category)
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    return result

@app.post("/categories/{category}/symbols")
async def add_symbol(category: str, symbol_data: SymbolAdd):
    result = await data_processor.add_symbol_to_category(category, symbol_data.symbol)
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    return result

@app.delete("/categories/{category}/symbols/{symbol}")
async def remove_symbol(category: str, symbol: str):
    result = await data_processor.remove_symbol_from_category(category, symbol)
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    return result

@app.get("/categories")
async def list_categories(query: str = ""):
    return {"categories": await data_processor.list_categories(query)}

@app.get("/categories/{category}/symbols")
async def search_symbols(category: str, query: str = ""):
    symbols = await data_processor.search_symbols_in_category(category, query)
    if not symbols and query == "":
        raise HTTPException(status_code=404, detail=f"Category {category} not found")
    return {"symbols": symbols}

@app.post("/upload/{category}/{symbol}")
async def upload_files(
    category: str,
    symbol: str,
    files: List[UploadFile] = File(...),
    chunk_size: Optional[int] = Query(50000, description="Number of lines to process in each chunk"),
    max_workers: Optional[int] = Query(4, description="Maximum number of parallel workers")
):
    try:
        # Update processor settings if provided
        data_processor.upload_chunk_size = chunk_size
        data_processor.max_workers = max_workers

        # Process files with progress tracking
        result = await data_processor.process_multiple_files(files, category, symbol)
        
        return {
            "status": "completed",
            "details": result,
            "message": f"Processed {result['processed_files']} out of {result['total_files']} files in {result['processing_time_seconds']:.2f} seconds"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": f"An error occurred during file processing: {str(e)}"
            }
        )

@app.get("/historical_data/{category}/{symbol}")
async def get_historical_data(
    category: str,
    symbol: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    timeframe: str = Query(..., regex="^(1T|5T|15T|30T|1H|4H|1D|1W|1M)$"),
    max: bool = Query(False, description="If true, retrieves all available data regardless of start/end dates")
):
    try:
        if max:
            # Get the first and last available dates for the symbol
            collection_name = data_processor._get_collection_name(category, symbol, timeframe)
            first_doc = await data_processor.db[collection_name].find_one(
                sort=[("timestamp", ASCENDING)]
            )
            last_doc = await data_processor.db[collection_name].find_one(
                sort=[("timestamp", DESCENDING)]
            )
            
            if not first_doc or not last_doc:
                raise HTTPException(
                    status_code=404,
                    detail=f"No data found for symbol {symbol} in category {category}"
                )
            
            start_date = first_doc["timestamp"]
            end_date = last_doc["timestamp"]
        else:
            if not start_date or not end_date:
                raise HTTPException(
                    status_code=400,
                    detail="start_date and end_date are required when max=false"
                )

        chunk_size = 86400  # One day in seconds
        time_chunks = []
        current = start_date
        
        while current < end_date:
            next_chunk = min(current + timedelta(seconds=chunk_size), end_date)
            time_chunks.append((current, next_chunk))
            current = next_chunk

        async with asyncio.TaskGroup() as tg:
            chunk_tasks = [
                tg.create_task(data_processor.get_data_chunk(category, symbol, start, end, timeframe))
                for start, end in time_chunks
            ]
        
        chunk_results = [task.result() for task in chunk_tasks]
        combined_data = {
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": []
        }
        
        for chunk in chunk_results:
            for key in combined_data:
                combined_data[key].extend(chunk.get(key, []))

        if not any(combined_data.values()):
            raise HTTPException(
                status_code=404,
                detail=f"No data found for symbol {symbol} in category {category}"
            )

        return {
            "data": combined_data,
            "metadata": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "timeframe": timeframe,
                "category": category,
                "symbol": symbol,
                "total_points": len(combined_data["timestamp"])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Additional utility endpoints
@app.get("/category_stats/{category}")
async def get_category_stats(category: str):
    """Get statistics for a category including number of symbols and data points"""
    try:
        category_doc = await data_processor.categories_collection.find_one({"category": category})
        if not category_doc:
            raise HTTPException(status_code=404, detail=f"Category {category} not found")
        
        total_symbols = len(category_doc.get("symbols", []))
        symbol_stats = []
        
        for symbol_info in category_doc.get("symbols", []):
            symbol = symbol_info["symbol"]
            collection_name = data_processor._get_collection_name(category, symbol, "1T")
            count = await data_processor.db[collection_name].count_documents({})
            
            first_doc = await data_processor.db[collection_name].find_one(
                sort=[("timestamp", ASCENDING)]
            )
            last_doc = await data_processor.db[collection_name].find_one(
                sort=[("timestamp", DESCENDING)]
            )
            
            symbol_stats.append({
                "symbol": symbol,
                "data_points": count,
                "first_date": first_doc["timestamp"].isoformat() if first_doc else None,
                "last_date": last_doc["timestamp"].isoformat() if last_doc else None
            })
        
        return {
            "category": category,
            "total_symbols": total_symbols,
            "symbols": symbol_stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/search_patterns/{category}/{symbol}")
async def search_patterns(
    category: str,
    symbol: str,
    pattern_start: datetime = Body(...),
    pattern_end: datetime = Body(...),
    search_start: Optional[datetime] = Body(None),
    search_end: Optional[datetime] = Body(None),
    timeframe: str = Body(..., regex="^(1T|5T|15T|30T|1H|4H|1D|1W|1M)$"),
    ema_period: int = Body(14),
    include_volume: bool = Body(False),
    top_n: int = Body(10),
    max: bool = Body(False, description="If true, searches through all available data")
):
    try:
        # Verify category and symbol exist
        category_doc = await data_processor.categories_collection.find_one({
            "category": category,
            "symbols.symbol": symbol
        })
        
        if not category_doc:
            if not await data_processor.categories_collection.find_one({"category": category}):
                raise HTTPException(status_code=404, detail=f"Category {category} not found")
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found in category {category}")

        # If max is true, get the entire date range
        if max:
            collection_name = data_processor._get_collection_name(category, symbol, timeframe)
            first_doc = await data_processor.db[collection_name].find_one(
                sort=[("timestamp", ASCENDING)]
            )
            last_doc = await data_processor.db[collection_name].find_one(
                sort=[("timestamp", DESCENDING)]
            )
            
            if not first_doc or not last_doc:
                raise HTTPException(
                    status_code=404,
                    detail=f"No data found for symbol {symbol} in category {category}"
                )
            
            search_start = first_doc["timestamp"]
            search_end = last_doc["timestamp"]
        else:
            if not search_start or not search_end:
                raise HTTPException(
                    status_code=400,
                    detail="search_start and search_end are required when max=false"
                )

        # Get pattern data
        pattern_data = await data_processor.get_data(
            category, 
            symbol,
            pattern_start,
            pattern_end,
            timeframe
        )

        # Validate minimum data points
        if len(pattern_data["timestamp"]) < 3:
            raise HTTPException(
                status_code=400, 
                detail="Pattern must contain at least 3 data points for meaningful comparison"
            )

        # Convert to Polars DataFrame for efficient processing
        pattern_df = pl.DataFrame({
            "timestamp": pl.Series(pattern_data["timestamp"]).cast(pl.Datetime),
            "close": pl.Series(pattern_data["close"]).cast(pl.Float64),
            "volume": pl.Series(pattern_data["volume"]).cast(pl.Float64)
        })

        # Calculate pattern EMA
        pattern_df = pattern_df.with_columns([
            pl.col("close").ewm_mean(span=ema_period).alias("ema")
        ])

        # Convert to NumPy arrays and ensure correct normalization
        pattern_close = pattern_df["close"].to_numpy()
        pattern_ema = pattern_df["ema"].to_numpy()
        pattern_volume = pattern_df["volume"].to_numpy() if include_volume else None

        # Normalize pattern data using min-max normalization
        pattern_close_norm = (pattern_close - np.min(pattern_close)) / (np.max(pattern_close) - np.min(pattern_close))
        pattern_ema_norm = (pattern_ema - np.min(pattern_ema)) / (np.max(pattern_ema) - np.min(pattern_ema))
        if include_volume:
            pattern_volume_norm = (pattern_volume - np.min(pattern_volume)) / (np.max(pattern_volume) - np.min(pattern_volume))

        pattern_length = len(pattern_df)
        chunk_size = timedelta(days=7)
        current_start = search_start
        all_matches = []
        
        while current_start < search_end:
            current_end = min(current_start + chunk_size, search_end)
            
            try:
                historical_data = await data_processor.get_data(
                    category,
                    symbol, 
                    current_start,
                    current_end,
                    timeframe
                )
            except Exception as e:
                logger.warning(f"No data found for period {current_start} to {current_end}: {str(e)}")
                current_start = current_end
                continue

            if not historical_data["timestamp"]:
                current_start = current_end
                continue

            historical_df = pl.DataFrame({
                "timestamp": pl.Series(historical_data["timestamp"]).cast(pl.Datetime),
                "close": pl.Series(historical_data["close"]).cast(pl.Float64),
                "volume": pl.Series(historical_data["volume"]).cast(pl.Float64),
                "open": pl.Series(historical_data["open"]).cast(pl.Float64),
                "high": pl.Series(historical_data["high"]).cast(pl.Float64),
                "low": pl.Series(historical_data["low"]).cast(pl.Float64)
            })

            historical_df = historical_df.with_columns([
                pl.col("close").ewm_mean(span=ema_period).alias("ema")
            ])

            hist_close = historical_df["close"].to_numpy()
            hist_ema = historical_df["ema"].to_numpy()
            hist_volume = historical_df["volume"].to_numpy() if include_volume else None
            
            chunk_matches = []
            
            i = 0
            while i < len(historical_df) - pattern_length + 1:
                # Extract and validate window data
                window_close = hist_close[i:i + pattern_length]
                window_ema = hist_ema[i:i + pattern_length]
                
                if len(window_close) < 3:
                    i += 1
                    continue

                # Normalize window data
                window_range = np.max(window_close) - np.min(window_close)
                if window_range == 0:  # Skip flat price sequences
                    i += 1
                    continue
                
                window_close_norm = (window_close - np.min(window_close)) / window_range
                window_ema_norm = (window_ema - np.min(window_ema)) / (np.max(window_ema) - np.min(window_ema))

                # Calculate cosine similarity
                close_similarity = np.dot(pattern_close_norm, window_close_norm) / (
                    np.linalg.norm(pattern_close_norm) * np.linalg.norm(window_close_norm)
                )
                
                ema_similarity = np.dot(pattern_ema_norm, window_ema_norm) / (
                    np.linalg.norm(pattern_ema_norm) * np.linalg.norm(window_ema_norm)
                )
                
                # Combine similarities
                similarity = (close_similarity + ema_similarity) / 2
                
                if include_volume and hist_volume is not None:
                    window_volume = hist_volume[i:i + pattern_length]
                    window_volume_norm = (window_volume - np.min(window_volume)) / (np.max(window_volume) - np.min(window_volume))
                    volume_similarity = np.dot(pattern_volume_norm, window_volume_norm) / (
                        np.linalg.norm(pattern_volume_norm) * np.linalg.norm(window_volume_norm)
                    )
                    similarity = (similarity * 2 + volume_similarity) / 3

                # Threshold check with absolute similarity
                if similarity > 0.7:  # Adjustable threshold
                    window_data = historical_df.slice(i, pattern_length)
                    match_data = {
                        "start_date": window_data["timestamp"][0].strftime("%Y-%m-%dT%H:%M:%S"),
                        "end_date": window_data["timestamp"][-1].strftime("%Y-%m-%dT%H:%M:%S"),
                        "similarity": float(similarity),
                        "index": i,
                        "data": {
                            "timestamp": window_data["timestamp"].to_list(),
                            "open": window_data["open"].to_list(),
                            "high": window_data["high"].to_list(),
                            "low": window_data["low"].to_list(),
                            "close": window_data["close"].to_list(),
                            "volume": window_data["volume"].to_list()
                        }
                    }
                    chunk_matches.append(match_data)
                    i += pattern_length  # Skip overlapping patterns
                else:
                    i += 1

            all_matches.extend(chunk_matches)
            current_start = current_end

        if not all_matches:
            return {
                "pattern_info": {
                    "start_date": pattern_start.isoformat(),
                    "end_date": pattern_end.isoformat(),
                    "timeframe": timeframe,
                    "data": pattern_data
                },
                "matches": [],
                "message": "No similar patterns found in the specified date range"
            }

        # Sort matches by similarity and get top N
        all_matches.sort(key=lambda x: x["similarity"], reverse=True)
        top_matches = all_matches[:top_n]
        
        # Remove the index field from final results
        for match in top_matches:
            del match["index"]

        return {
            "pattern_info": {
                "start_date": pattern_start.isoformat(),
                "end_date": pattern_end.isoformat(),
                "timeframe": timeframe,
                "data": pattern_data
            },
            "matches": top_matches
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in pattern search: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during pattern search: {str(e)}"
        )

    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6027)
