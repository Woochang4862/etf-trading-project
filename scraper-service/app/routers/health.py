"""Health check router."""
from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

from app.models.task_info import task_info_manager, JobStatus

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    service: str
    current_job: Optional[str] = None


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    job_info = await task_info_manager.get_job_info()
    current_job = None

    if job_info.status == JobStatus.RUNNING:
        current_job = job_info.job_id

    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        service="scraper-service",
        current_job=current_job
    )
