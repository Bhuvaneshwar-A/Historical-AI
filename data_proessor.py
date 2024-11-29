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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, db_url: str):
        try:
            self.client = AsyncIOMotorClient(
                db_url,
                maxPoolSize=60,
                minPoolSize=30,
                serverSelectionTimeoutMS=2000,
                connectTimeoutMS=2000,
                maxIdleTimeMS=200000,
                retryWrites=True,
                w=1,
                journal=False
            )
            self.db = self.client["Historical_data"]
            self.categories_collection = self.db["categories"]
            self.chunk_size = 200000
            self.upload_chunk_size = 100000
            self.max_workers = 4
            self.bulk_write_size = 100000

            self.timeframes = {
                '1T': timedelta(minutes=1),
                '5T': timedelta(minutes=5),
                '15T': timedelta(minutes=15),
                '30T': timedelta(minutes=30),
                '1H': timedelta(hours=1),
                '4H': timedelta(hours=4),
                '1D': timedelta(days=1),
                '1W': timedelta(weeks=1),
                '1M': timedelta(days=30)
            }

            self.polars_timeframes = {
                '1T': '1m',
                '5T': '5m',
                '15T': '15m',
                '30T': '30m',
                '1H': '1h',
                '4H': '4h',
                '1D': '1d',
                '1W': '1w',
                '1M': '1mo'
            }
            self.index_cache = set()
        except Exception as e:
            logger.error(f"Error initializing DataProcessor: {str(e)}")
            raise

    async def initialize(self):
        """Initialize required indexes for categories"""
        try:
            await self.categories_collection.create_index([("category", ASCENDING)], unique=True)
            await self.categories_collection.create_index([
                ("category", ASCENDING),
                ("symbols.symbol", ASCENDING)
            ])
        except Exception as e:
            logger.error(f"Error initializing indexes: {str(e)}")
            raise

    def _chunk_file_content(self, content: bytes, chunk_size: int) -> Iterator[List[str]]:
        """Split file content into chunks efficiently"""
        try:
            buffer = io.StringIO(content.decode('utf-8'))
            while True:
                chunk = list(islice(buffer, chunk_size))
                if not chunk:
                    break
                yield chunk
        except Exception as e:
            logger.error(f"Error chunking file content: {str(e)}")
            raise

    def _parse_chunk(self, lines: List[str], symbol: str) -> List[Dict[str, Any]]:
        """Parse a chunk of lines into data records"""
        data = []
        for line in lines:
            if line.strip():
                try:
                    parts = line.split(';')
                    timestamp = datetime.strptime(
                        parts[0], '%Y%m%d %H%M%S'
                    ).replace(tzinfo=pytz.UTC)
                    data.append({
                        'timestamp': timestamp,
                        'open': float(parts[1]),
                        'high': float(parts[2]),
                        'low': float(parts[3]),
                        'close': float(parts[4]),
                        'volume': int(parts[5]),
                        'symbol': symbol
                    })
                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipping invalid line: {line.strip()} - Error: {str(e)}")
        return data

    @lru_cache(maxsize=100)
    def _get_collection_name(self, category: str, symbol: str, timeframe: str) -> str:
        """Cache collection names to avoid string operations"""
        return f"{category}_{symbol}_{timeframe}"

    async def _ensure_indexes(self, collection_name: str) -> None:
        """Ensure minimal required indexes exist"""
        try:
            if collection_name in self.index_cache:
                return
            await self.db[collection_name].create_index([
                ("timestamp", ASCENDING),
                ("symbol", ASCENDING)
            ], unique=True, background=True)
            self.index_cache.add(collection_name)
        except Exception as e:
            logger.error(f"Error ensuring indexes for collection {collection_name}: {str(e)}")
            raise

    def _create_polars_frame(self, data: List[Dict[str, Any]]) -> pl.DataFrame:
        """Create a Polars DataFrame with proper column types"""
        try:
            if not data:
                return pl.DataFrame()

            return pl.DataFrame(data).with_columns([
                pl.col('timestamp').cast(pl.Datetime),
                pl.col('open').cast(pl.Float64),
                pl.col('high').cast(pl.Float64),
                pl.col('low').cast(pl.Float64),
                pl.col('close').cast(pl.Float64),
                pl.col('volume').cast(pl.Int64),
                pl.col('symbol').cast(pl.Categorical)
            ])
        except Exception as e:
            logger.error(f"Error creating Polars DataFrame: {str(e)}")
            raise

    async def _store_raw_data(self, data: List[Dict[str, Any]], category: str, symbol: str) -> None:
        """Store raw 1-minute data without aggregation"""
        try:
            collection_name = self._get_collection_name(category, symbol, '1T')
            await self._ensure_indexes(collection_name)

            bulk_operations = [
                ReplaceOne(
                    {"timestamp": item["timestamp"], "symbol": symbol},
                    item,
                    upsert=True
                ) for item in data
            ]

            collection = self.db[collection_name]
            for i in range(0, len(bulk_operations), self.bulk_write_size):
                batch = bulk_operations[i:i + self.bulk_write_size]
                result = await collection.bulk_write(batch, ordered=False)
                logger.info(f"Raw data batch {i//self.bulk_write_size + 1}: Inserted/Updated {result.upserted_count + result.modified_count} documents for {collection_name}")
        except Exception as e:
            logger.error(f"Error storing raw data for {collection_name}: {str(e)}")
            raise

    def aggregate_dataframe(self, df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
        """Aggregate data into specified timeframe"""
        try:
            return df.group_by_dynamic(
                index_column="timestamp",
                every=timeframe,
                by="symbol"
            ).agg([
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume")
            ]).drop_nulls()
        except Exception as e:
            logger.error(f"Error aggregating dataframe: {str(e)}")
            raise

    async def _aggregate_and_store(
        self,
        df: pl.DataFrame,
        category: str,
        symbol: str,
        mongo_tf: str,
        polars_tf: str
    ) -> None:
        """Aggregate and store data for a specific timeframe"""
        try:
            collection_name = self._get_collection_name(category, symbol, mongo_tf)
            await self._ensure_indexes(collection_name)

            agg_df = self.aggregate_dataframe(df, polars_tf)
            if agg_df.is_empty():
                return

            agg_data = agg_df.to_dicts()

            for i in range(0, len(agg_data), self.bulk_write_size):
                batch = agg_data[i:i + self.bulk_write_size]
                bulk_operations = [
                    ReplaceOne(
                        {"timestamp": item["timestamp"], "symbol": symbol},
                        item,
                        upsert=True
                    ) for item in batch
                ]

                collection = self.db[collection_name]
                result = await collection.bulk_write(bulk_operations, ordered=False)
                logger.info(f"Batch {i//self.bulk_write_size + 1}: Inserted/Updated {result.upserted_count + result.modified_count} documents for {collection_name}")
        except Exception as e:
            logger.error(f"Error during bulk write for {collection_name}: {str(e)}")
            raise

    async def _process_chunk_and_aggregate(
        self,
        chunk_data: List[Dict[str, Any]],
        category: str,
        symbol: str,
        semaphore: asyncio.Semaphore
    ) -> None:
        """Process aggregations for timeframes larger than 1 minute"""
        try:
            if not chunk_data:
                return

            df = self._create_polars_frame(chunk_data)
            if df.is_empty():
                return

            async with semaphore:
                tasks = []
                # Skip 1T timeframe as it's already stored in raw form
                for mongo_tf, polars_tf in {k: v for k, v in self.polars_timeframes.items() if k != '1T'}.items():
                    task = self._aggregate_and_store(df, category, symbol, mongo_tf, polars_tf)
                    tasks.append(task)

                await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error processing chunk and aggregating: {str(e)}")
            raise

    async def process_file(self, file: Any, category: str, symbol: str) -> None:
        """Process a single file with chunking and parallel processing"""
        try:
            content = await file.read()

            semaphore = asyncio.Semaphore(self.max_workers)
            tasks = []

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for lines in self._chunk_file_content(content, self.upload_chunk_size):
                    chunk_data = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        self._parse_chunk,
                        lines,
                        symbol
                    )

                    if chunk_data:
                        # Store raw 1-minute data first
                        await self._store_raw_data(chunk_data, category, symbol)

                        # Then process aggregations
                        task = asyncio.create_task(
                            self._process_chunk_and_aggregate(chunk_data, category, symbol, semaphore)
                        )
                        tasks.append(task)

                    if len(tasks) >= self.max_workers * 2:
                        completed, tasks = await asyncio.wait(
                            tasks,
                            return_when=asyncio.FIRST_COMPLETED
                        )

            if tasks:
                await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise

    # Category management methods
    async def add_category(self, category: str) -> Dict[str, Any]:
        """Add a new category"""
        try:
            await self.categories_collection.insert_one({
                "category": category,
                "symbols": [],
                "created_at": datetime.utcnow()
            })
            return {"status": "success", "message": f"Category {category} created successfully"}
        except Exception as e:
            if "duplicate key error" in str(e).lower():
                return {"status": "error", "message": f"Category {category} already exists"}
            logger.error(f"Error adding category {category}: {str(e)}")
            raise

    async def delete_category(self, category: str) -> Dict[str, Any]:
        """Delete a category and all its associated data"""
        try:
            category_doc = await self.categories_collection.find_one({"category": category})
            if not category_doc:
                return {"status": "error", "message": f"Category {category} not found"}

            # Delete all collections for this category's symbols
            for symbol_info in category_doc["symbols"]:
                symbol = symbol_info["symbol"]
                for timeframe in self.timeframes.keys():
                    collection_name = self._get_collection_name(category, symbol, timeframe)
                    await self.db[collection_name].drop()

            # Delete the category itself
            await self.categories_collection.delete_one({"category": category})
            return {"status": "success", "message": f"Category {category} and all its data deleted"}
        except Exception as e:
            logger.error(f"Error deleting category {category}: {str(e)}")
            raise

    async def add_symbol_to_category(self, category: str, symbol: str) -> Dict[str, Any]:
        """Add a symbol to a category"""
        try:
            result = await self.categories_collection.update_one(
                {"category": category},
                {"$addToSet": {"symbols": {
                    "symbol": symbol,
                    "added_at": datetime.utcnow()
                }}}
            )
            if result.matched_count == 0:
                return {"status": "error", "message": f"Category {category} not found"}
            return {"status": "success", "message": f"Symbol {symbol} added to category {category}"}
        except Exception as e:
            logger.error(f"Error adding symbol {symbol} to category {category}: {str(e)}")
            raise

    async def remove_symbol_from_category(self, category: str, symbol: str) -> Dict[str, Any]:
        """Remove a symbol from a category"""
        try:
            result = await self.categories_collection.update_one(
                {"category": category},
                {"$pull": {"symbols": {"symbol": symbol}}}
            )
            if result.matched_count == 0:
                return {"status": "error", "message": f"Category {category} not found"}

            # Drop all collections for this symbol
            for timeframe in self.timeframes.keys():
                collection_name = self._get_collection_name(category, symbol, timeframe)
                await self.db[collection_name].drop()

            return {"status": "success", "message": f"Symbol {symbol} removed from category {category}"}
        except Exception as e:
            logger.error(f"Error removing symbol {symbol} from category {category}: {str(e)}")
            raise

    async def list_categories(self, query: str = "") -> List[Dict[str, Any]]:
        """List all categories with optional search"""
        try:
            filter_query = {}
            if query:
                filter_query = {"category": {"$regex": query, "$options": "i"}}

            categories = await self.categories_collection.find(
                filter_query,
                {"_id": 0}
            ).to_list(None)

            return categories
        except Exception as e:
            logger.error(f"Error listing categories: {str(e)}")
            raise

    async def search_symbols_in_category(self, category: str, query: str = "") -> List[str]:
        """Search symbols within a specific category"""
        try:
            category_doc = await self.categories_collection.find_one(
                {"category": category},
                {"_id": 0, "symbols": 1}
            )

            if not category_doc:
                return []

            symbols = [
                s["symbol"] for s in category_doc["symbols"]
                if not query or query.lower() in s["symbol"].lower()
            ]
            return sorted(symbols)
        except Exception as e:
            logger.error(f"Error searching symbols in category {category}: {str(e)}")
            raise

    async def process_multiple_files(
        self,
        files: List[Any],
        category: str,
        symbol: str
    ) -> Dict[str, Any]:
        """Process multiple files with progress tracking"""
        try:
            # First, ensure the category exists and the symbol is added
            await self.add_category(category)
            await self.add_symbol_to_category(category, symbol)

            start_time = datetime.now()
            total_files = len(files)
            processed_files = 0
            errors = []

            async def process_with_progress(file):
                try:
                    await self.process_file(file, category, symbol)
                    return True, None
                except Exception as e:
                    return False, str(e)

            tasks = [process_with_progress(file) for file in files]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, (success, error) in enumerate(results):
                if success:
                    processed_files += 1
                elif error:
                    errors.append(f"Error processing file {i + 1}: {error}")

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            return {
                "total_files": total_files,
                "processed_files": processed_files,
                "failed_files": total_files - processed_files,
                "processing_time_seconds": processing_time,
                "errors": errors
            }
        except Exception as e:
            logger.error(f"Error processing multiple files: {str(e)}")
            raise

    async def get_data(
        self,
        category: str,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> Dict[str, List]:
        """Ultra-fast data retrieval with optimized parallel processing"""
        try:
            # Verify category and symbol exist
            category_doc = await self.categories_collection.find_one({
                "category": category,
                "symbols.symbol": symbol
            })

            if not category_doc:
                raise ValueError(f"Symbol {symbol} not found in category {category}")

            collection_name = self._get_collection_name(category, symbol, timeframe)
            await self._ensure_indexes(collection_name)

            # Calculate optimal chunk size
            time_diff = (end_date - start_date).total_seconds()
            chunk_duration = min(time_diff / 10, 86400)  # Max 1 day per chunk
            num_chunks = max(1, min(10, int(time_diff / chunk_duration)))

            # Create time chunks
            time_chunks = []
            chunk_size = time_diff / num_chunks
            for i in range(num_chunks):
                chunk_start = start_date + timedelta(seconds=i * chunk_size)
                chunk_end = start_date + timedelta(seconds=(i + 1) * chunk_size)
                if i == num_chunks - 1:
                    chunk_end = end_date  # Ensure we cover the entire range
                time_chunks.append((chunk_start, chunk_end))

            # Process chunks in parallel using TaskGroup
            try:
                async with asyncio.TaskGroup() as tg:
                    tasks = [
                        tg.create_task(self._fetch_chunk_fast(collection_name, start, end))
                        for start, end in time_chunks
                    ]
                chunks = [task.result() for task in tasks]
            except AttributeError:
                # Fallback for Python versions without TaskGroup
                tasks = [
                    self._fetch_chunk_fast(collection_name, start, end)
                    for start, end in time_chunks
                ]
                chunks = await asyncio.gather(*tasks)

            return self._combine_chunks_fast(chunks)
        except Exception as e:
            logger.error(f"Error retrieving data: {str(e)}")
            raise

    async def _fetch_chunk_fast(
        self,
        collection_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, List]:
        """Optimized chunk retrieval with pre-allocated memory"""
        try:
            collection = self.db[collection_name]

            projection = {
                "_id": 0,
                "timestamp": 1,
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
                "volume": 1
            }

            # Pre-allocate arrays with estimated size
            estimated_size = int((end_date - start_date).total_seconds() / 60) + 100
            data = {
                "timestamp": [None] * estimated_size,
                "open": [None] * estimated_size,
                "high": [None] * estimated_size,
                "low": [None] * estimated_size,
                "close": [None] * estimated_size,
                "volume": [None] * estimated_size
            }

            # Use raw command for better performance
            pipeline = [
                {
                    "$match": {
                        "timestamp": {
                            "$gte": start_date,
                            "$lt": end_date
                        }
                    }
                },
                {
                    "$sort": {"timestamp": 1}
                },
                {
                    "$project": projection
                }
            ]

            idx = 0
            async for doc in collection.aggregate(pipeline, batchSize=10000):
                for field in data:
                    if field == "timestamp":
                        data[field][idx] = doc[field].isoformat()
                    elif field == "volume":
                        data[field][idx] = int(doc[field])
                    else:
                        data[field][idx] = float(doc[field])
                idx += 1

            # Trim arrays to actual size
            for field in data:
                data[field] = data[field][:idx]

            return data
        except Exception as e:
            logger.error(f"Error fetching chunk: {str(e)}")
            raise

    def _combine_chunks_fast(self, chunks: List[Dict[str, List]]) -> Dict[str, List]:
        """Fast chunk combination with pre-allocation"""
        try:
            if not chunks:
                return {
                    "timestamp": [],
                    "open": [],
                    "high": [],
                    "low": [],
                    "close": [],
                    "volume": []
                }

            # Calculate total length for pre-allocation
            total_length = sum(len(chunk["timestamp"]) for chunk in chunks)

            # Pre-allocate result arrays
            result = {
                "timestamp": [None] * total_length,
                "open": [None] * total_length,
                "high": [None] * total_length,
                "low": [None] * total_length,
                "close": [None] * total_length,
                "volume": [None] * total_length
            }

            # Fast chunk combination
            position = 0
            for chunk in chunks:
                chunk_length = len(chunk["timestamp"])
                for field in result:
                    result[field][position:position + chunk_length] = chunk[field]
                position += chunk_length

            return result
        except Exception as e:
            logger.error(f"Error combining chunks: {str(e)}")
            raise

    async def get_data_chunk(self, category: str, symbol: str, start: datetime, end: datetime, timeframe: str) -> dict:
        try:
            collection_name = f"{category}_{symbol}_{timeframe}"
            pipeline = [
                {
                    "$match": {
                        "timestamp": {"$gte": start, "$lt": end}
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "timestamp": 1,
                        "open": 1,
                        "high": 1,
                        "low": 1,
                        "close": 1,
                        "volume": 1
                    }
                },
                {
                    "$sort": {"timestamp": 1}
                }
            ]

            cursor = self.db[collection_name].aggregate(
                pipeline,
                allowDiskUse=True,
                batchSize=10000
            )

            chunk_data = {
                "timestamp": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": []
            }

            async for doc in cursor:
                chunk_data["timestamp"].append(doc["timestamp"].isoformat())
                chunk_data["open"].append(float(doc["open"]))
                chunk_data["high"].append(float(doc["high"]))
                chunk_data["low"].append(float(doc["low"]))
                chunk_data["close"].append(float(doc["close"]))
                chunk_data["volume"].append(int(doc["volume"]))

            return chunk_data
        except Exception as e:
            logger.error(f"Error getting data chunk: {str(e)}")
            raise

    async def search_symbols(self, query: str = "") -> List[str]:
        """Fast symbol search with caching"""
        try:
            collections = await self.db.list_collection_names()
            symbols = {
                coll.rsplit('_', 1)[0]
                for coll in collections
                if coll.endswith("_1T") and (not query or query.lower() in coll.lower())
            }
            return sorted(list(symbols))
        except Exception as e:
            logger.error(f"Error searching symbols: {str(e)}")
            raise

    async def get_symbol_date_range(self, category: str, symbol: str, timeframe: str) -> Tuple[datetime, datetime]:
        """Get the first and last available dates for a symbol"""
        try:
            collection_name = self._get_collection_name(category, symbol, timeframe)
            first_doc = await self.db[collection_name].find_one(
                sort=[("timestamp", ASCENDING)]
            )
            last_doc = await self.db[collection_name].find_one(
                sort=[("timestamp", DESCENDING)]
            )
            
            if not first_doc or not last_doc:
                raise ValueError(f"No data found for symbol {symbol} in category {category}")
                
            return first_doc["timestamp"], last_doc["timestamp"]
        except Exception as e:
            logger.error(f"Error getting symbol date range: {str(e)}")
            raise