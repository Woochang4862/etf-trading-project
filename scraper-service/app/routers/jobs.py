"""Job management router."""
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from app.models.task_info import task_info_manager, JobStatus, SymbolStatus
from app.services.scraper import scraper, STOCK_LIST

router = APIRouter()
logger = logging.getLogger(__name__)


class JobResponse(BaseModel):
    """Response for job creation."""
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: Optional[str] = None
    status: str
    progress: dict
    current_symbol: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None


class RetryRequest(BaseModel):
    """Request model for retrying failed symbols."""
    symbols: List[str]


async def run_scraping_job(symbols: List[str], is_retry: bool = False):
    """Background task to run scraping job."""
    try:
        async with scraper:
            await scraper.process_all_stocks(symbols, is_retry=is_retry)
    except Exception as e:
        logger.error(f"Scraping job failed: {e}")
        await task_info_manager.update_job_status(JobStatus.ERROR)


@router.post("/full", response_model=JobResponse)
async def start_full_job(background_tasks: BackgroundTasks):
    """Start a full scraping job for all symbols."""
    job_info = await task_info_manager.get_job_info()

    if job_info.status == JobStatus.RUNNING:
        raise HTTPException(
            status_code=409,
            detail=f"Job already running: {job_info.job_id}"
        )

    from datetime import datetime
    job_id = f"full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # job 초기화는 process_all_stocks 내부에서 수행됨
    background_tasks.add_task(run_scraping_job, STOCK_LIST)

    return JobResponse(
        job_id=job_id,
        status="pending",
        message=f"Started full scraping job for {len(STOCK_LIST)} symbols"
    )


@router.post("/retry", response_model=JobResponse)
async def retry_symbols(request: RetryRequest, background_tasks: BackgroundTasks):
    """Retry scraping for specific symbols. Preserves main job state."""
    job_info = await task_info_manager.get_job_info()

    if job_info.status == JobStatus.RUNNING:
        raise HTTPException(
            status_code=409,
            detail=f"Job already running: {job_info.job_id}"
        )

    # Validate symbols
    valid_symbols = [s for s in request.symbols if s in STOCK_LIST]
    if not valid_symbols:
        raise HTTPException(
            status_code=400,
            detail="No valid symbols provided"
        )

    # Start retry task (preserves main job state)
    background_tasks.add_task(run_scraping_job, valid_symbols, True)

    return JobResponse(
        job_id=f"retry_{len(valid_symbols)}_symbols",
        status="pending",
        message=f"Started retry job for {len(valid_symbols)} symbols: {', '.join(valid_symbols)}"
    )


@router.get("/status", response_model=JobStatusResponse)
async def get_status():
    """Get current job status."""
    job_info = await task_info_manager.get_job_info()

    completed = sum(1 for s in job_info.symbols.values() if s.status == SymbolStatus.COMPLETED)
    failed = sum(1 for s in job_info.symbols.values() if s.status == SymbolStatus.FAILED)
    total = len(job_info.symbols)

    return JobStatusResponse(
        job_id=job_info.job_id if job_info.job_id != "initial" else None,
        status=job_info.status.value,
        progress={
            "current": completed,
            "total": total,
            "current_symbol": job_info.current_symbol,
            "errors": [s.symbol for s in job_info.symbols.values() if s.status == SymbolStatus.FAILED]
        },
        current_symbol=job_info.current_symbol,
        start_time=job_info.start_time,
        end_time=job_info.end_time
    )


@router.post("/cancel")
async def cancel_job():
    """Cancel the current running job."""
    job_info = await task_info_manager.get_job_info()

    if job_info.status != JobStatus.RUNNING:
        raise HTTPException(
            status_code=400,
            detail="No job is currently running"
        )

    await task_info_manager.update_job_status(JobStatus.STOPPED)

    return {"message": "Job cancelled", "job_id": job_info.job_id}


@router.get("/logs")
async def get_logs(limit: int = 100):
    """Get recent log entries."""
    from pathlib import Path
    from app.config import settings

    log_dir = Path(settings.log_dir)
    logs = []

    # Find most recent log file
    log_files = sorted(log_dir.glob("scraper_*.log"), reverse=True)

    if log_files:
        try:
            with open(log_files[0], "r") as f:
                lines = f.readlines()
                logs = lines[-limit:]
        except Exception as e:
            logger.error(f"Failed to read logs: {e}")

    return {"logs": logs, "log_file": str(log_files[0]) if log_files else None}
