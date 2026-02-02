"""FastAPI application for scraper service."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import health, jobs

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="ETF Scraper Service",
    description="TradingView data scraping service with async job management",
    version="1.0.0",
    root_path="/api/scraper"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(jobs.router, prefix="/jobs", tags=["Jobs"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logging.info("Scraper service starting up...")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logging.info("Scraper service shutting down...")
