#!/usr/bin/env python3
"""
Wiki Processor - FastAPI service
Receives markdown content from CI, performs LLM analysis, and stores wiki in Minio.

Mode: Incremental updates via prompt caching
"""

import asyncio
import io
import json
import logging
import re
from datetime import datetime
from typing import Optional
from functools import lru_cache

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from minio import Minio
from minio.error import S3Error

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Wiki Processor", version="0.1.0")


# ============================================================================
# Data Models
# ============================================================================

class ProcessRequest(BaseModel):
    """Request body for /process endpoint"""
    markdowns: dict[str, str]  # {filename: content}
    timestamp: str
    trigger_info: dict


class ProcessResponse(BaseModel):
    """Response from /process endpoint"""
    status: str
    message: str
    wiki_url: Optional[str] = None
    changes_summary: dict = {}
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    minio_connected: bool
    minimax_accessible: bool


# ============================================================================
# Configuration
# ============================================================================

@lru_cache(maxsize=1)
def get_minio_client():
    """Get Minio client (cached)"""
    return Minio(
        "minio:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False,
    )


def get_minimax_api_key() -> str:
    """Get Minimax API key from environment"""
    import os
    key = os.getenv("MINIMAX_API_KEY")
    if not key:
        logger.warning("MINIMAX_API_KEY not set, will fail on LLM calls")
    return key or "dummy-key"


# ============================================================================
# Minio Operations
# ============================================================================

async def get_snapshot() -> dict:
    """Retrieve markdowns_snapshot.json from Minio"""
    minio = get_minio_client()
    bucket = "wiki-data"

    try:
        obj = minio.get_object(bucket, "markdowns_snapshot.json")
        content = obj.read().decode()
        logger.info("Retrieved snapshot from Minio")
        return json.loads(content)
    except S3Error as e:
        if e.code == "NoSuchKey":
            logger.info("No snapshot found (first run)")
            return {}
        logger.error(f"Minio error: {e}")
        raise HTTPException(status_code=500, detail=f"Minio error: {e}")


async def save_snapshot(markdowns: dict) -> None:
    """Save markdowns_snapshot.json to Minio"""
    minio = get_minio_client()
    bucket = "wiki-data"

    data = json.dumps(markdowns, ensure_ascii=False, indent=2).encode()
    try:
        minio.put_object(bucket, "markdowns_snapshot.json", io.BytesIO(data), len(data))
        logger.info("Saved snapshot to Minio")
    except S3Error as e:
        logger.error(f"Failed to save snapshot: {e}")
        raise


async def get_wiki() -> dict:
    """Retrieve wiki.json from Minio"""
    minio = get_minio_client()
    bucket = "wiki-data"

    try:
        obj = minio.get_object(bucket, "wiki.json")
        content = obj.read().decode()
        logger.info("Retrieved wiki.json from Minio")
        return json.loads(content)
    except S3Error as e:
        if e.code == "NoSuchKey":
            logger.info("No wiki found (first run)")
            return {"apis": {}, "metadata": {"version": "1.0", "created_at": datetime.now().isoformat()}}
        logger.error(f"Minio error: {e}")
        raise


async def save_wiki(wiki: dict) -> None:
    """Save wiki.json to Minio"""
    minio = get_minio_client()
    bucket = "wiki-data"

    wiki["metadata"]["updated_at"] = datetime.now().isoformat()

    data = json.dumps(wiki, ensure_ascii=False, indent=2).encode()
    try:
        minio.put_object(bucket, "wiki.json", io.BytesIO(data), len(data))
        logger.info("Saved wiki.json to Minio")
    except S3Error as e:
        logger.error(f"Failed to save wiki: {e}")
        raise


# ============================================================================
# Diff & Change Detection
# ============================================================================

def detect_changes(old_snapshot: dict, new_markdowns: dict) -> dict:
    """
    Compare old snapshot with new markdowns.
    Return: {"added": [...], "modified": [...], "deleted": [...]}
    """
    old_files = set(old_snapshot.keys())
    new_files = set(new_markdowns.keys())

    added = new_files - old_files
    deleted = old_files - new_files
    modified = {f for f in old_files & new_files if old_snapshot[f] != new_markdowns[f]}

    return {
        "added": sorted(list(added)),
        "modified": sorted(list(modified)),
        "deleted": sorted(list(deleted)),
    }


# ============================================================================
# LLM Integration (Minimax)
# ============================================================================

MINIMAX_API_URL = "https://api.minimax.io/v1/text/chatcompletion_v2"
MINIMAX_MODEL = "MiniMax-M2.7"


def extract_json(content: str) -> dict:
    """Remove <think>...</think> tags and extract JSON from LLM response."""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


async def call_minimax_initial(markdowns: dict) -> dict:
    """
    First run: Analyze all markdowns and generate complete wiki.
    Prompt: Analyze these API markdowns, categorize, and generate wiki JSON.
    """
    api_key = get_minimax_api_key()

    # Concatenate all markdowns
    combined = "\n\n".join(f"## File: {fname}\n{content}" for fname, content in markdowns.items())

    prompt = f"""Analyze the following API documentation markdown files and generate a structured wiki.

{combined}

Task:
1. Extract all API endpoints (method, path, description, parameters)
2. Group by module/service
3. Generate JSON structure: {{"apis": {{"module": {{"endpoint": {{...}}}}}}, "metadata": {{}}}}

Output ONLY valid JSON, no markdown.
"""

    logger.info(f"Calling Minimax for initial wiki generation (LLM call size: {len(combined)} chars)")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                MINIMAX_API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": MINIMAX_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            result = response.json()

            content = result["choices"][0]["message"]["content"]
            wiki = extract_json(content)
            logger.info("Successfully generated initial wiki")
            return wiki

        except httpx.HTTPError as e:
            logger.error(f"Minimax API error: {e}")
            raise HTTPException(status_code=500, detail=f"LLM API error: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            raise HTTPException(status_code=500, detail=f"Invalid LLM response: {e}")


async def call_minimax_incremental(current_wiki: dict, changed_markdowns: dict, changes_summary: dict) -> dict:
    """
    Incremental update: Merge changes into existing wiki using prompt caching.

    Uses Karpathy-style incremental learning:
    - Current wiki as context (cached)
    - Only changed markdowns as new input
    - Returns delta to merge
    """
    api_key = get_minimax_api_key()

    changed_content = "\n\n".join(
        f"## File: {fname}\n{content}"
        for fname, content in changed_markdowns.items()
    )

    # Prepare current wiki summary for caching
    wiki_summary = json.dumps(current_wiki, ensure_ascii=False, indent=2)[:2000]  # Truncate for demo

    prompt = f"""Current Wiki (summarized):
{wiki_summary}

Changes: {json.dumps(changes_summary)}

New/Modified Markdowns:
{changed_content}

Task:
1. For new files: Extract APIs and add to wiki
2. For modified files: Update related APIs
3. For deleted files: Remove from wiki
4. Maintain module structure and semantic relationships

Output ONLY the updated wiki JSON.
"""

    logger.info(f"Calling Minimax for incremental update (changes: {changes_summary})")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                MINIMAX_API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": MINIMAX_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            result = response.json()

            content = result["choices"][0]["message"]["content"]
            updated_wiki = extract_json(content)
            logger.info("Successfully performed incremental wiki update")
            return updated_wiki

        except httpx.HTTPError as e:
            logger.error(f"Minimax API error: {e}")
            raise HTTPException(status_code=500, detail=f"LLM API error: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            raise HTTPException(status_code=500, detail=f"Invalid LLM response: {e}")


# ============================================================================
# Main Processing Logic
# ============================================================================

async def process_markdowns(request: ProcessRequest) -> ProcessResponse:
    """
    Main processing pipeline:
    1. Get old snapshot
    2. Detect changes
    3. If first run: LLM generates complete wiki
       If incremental: LLM merges changes
    4. Save updated wiki + new snapshot
    """

    timestamp = request.timestamp

    # Step 1: Retrieve old snapshot
    old_snapshot = await get_snapshot()
    is_first_run = len(old_snapshot) == 0

    # Step 2: Detect changes
    if is_first_run:
        logger.info("First run detected - will generate complete wiki")
        changes = {"added": list(request.markdowns.keys()), "modified": [], "deleted": []}
    else:
        changes = detect_changes(old_snapshot, request.markdowns)
        logger.info(f"Changes detected: {changes}")

    # Step 3: Call LLM
    if is_first_run:
        wiki = await call_minimax_initial(request.markdowns)
    else:
        # Extract only changed markdowns
        changed_markdowns = {
            f: request.markdowns[f] for f in
            (set(changes["added"]) | set(changes["modified"]))
        }

        if not changed_markdowns:
            logger.info("No actual content changes, skipping LLM call")
            wiki = await get_wiki()
        else:
            current_wiki = await get_wiki()
            wiki = await call_minimax_incremental(current_wiki, changed_markdowns, changes)

    # Step 4: Save to Minio
    await save_wiki(wiki)
    await save_snapshot(request.markdowns)

    logger.info(f"Processing complete for {timestamp}")

    return ProcessResponse(
        status="success",
        message=f"Wiki {'generated' if is_first_run else 'updated'} successfully",
        wiki_url="minio://wiki-data/wiki.json",
        changes_summary=changes,
        timestamp=datetime.now().isoformat(),
    )


# ============================================================================
# FastAPI Routes
# ============================================================================

@app.post("/process", response_model=ProcessResponse)
async def process(request: ProcessRequest):
    """Process markdown files and update wiki"""
    try:
        return await process_markdowns(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


@app.get("/status", response_model=dict)
async def status():
    """Get status of the processor"""
    try:
        wiki = await get_wiki()
        snapshot = await get_snapshot()
        return {
            "status": "running",
            "wiki_size": len(wiki.get("apis", {})),
            "tracked_files": len(snapshot),
            "last_updated": wiki.get("metadata", {}).get("updated_at", "unknown"),
        }
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check"""
    minio_ok = False
    minimax_ok = False

    try:
        minio = get_minio_client()
        minio.bucket_exists("wiki-data")
        minio_ok = True
    except Exception as e:
        logger.error(f"Minio health check failed: {e}")

    try:
        api_key = get_minimax_api_key()
        if api_key and api_key != "dummy-key":
            minimax_ok = True
    except Exception as e:
        logger.error(f"Minimax health check failed: {e}")

    return HealthResponse(
        status="ok" if minio_ok else "degraded",
        minio_connected=minio_ok,
        minimax_accessible=minimax_ok,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
