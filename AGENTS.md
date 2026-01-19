# AGENTS.md - Guidelines for Coding Agents

This document provides build commands and code style guidelines for the ETF Trading Project.

## Build, Lint, and Test Commands

### Next.js Web Dashboard (`web-dashboard/`)
```bash
npm run dev          # Start dev server (http://localhost:3000)
npm run build        # Production build
npm run start        # Start production server
npm run lint         # Run ESLint
```

### Python ML Service (`ml-service/`)
```bash
# Start service
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# Or via Docker: docker-compose up -d ml-service

# Run tests (add pytest to requirements.txt first)
pytest tests/                          # All tests
pytest tests/test_specific.py -v       # Single file
pytest tests/test_specific.py::test_function  # Single function
```

### ETF Model (`etf-model/`)
```bash
source .venv/bin/activate  # Activate venv first
python -m src.experiment_pipeline --model lightgbm --features 100 --years 2024
python run_tuning_experiment.py  # Hyperparameter tuning
pytest tests/                    # If tests exist
```

### Docker Services
```bash
docker-compose up -d                # Start all services
docker-compose down                 # Stop all services
docker-compose logs -f ml-service   # View specific logs
docker-compose restart ml-service    # Restart specific service
```

## Code Style Guidelines

### Python (ml-service, etf-model)

**Imports**: standard library → third-party → local
```python
import logging
from typing import Optional
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends
from app.models import Prediction
```

**Naming Conventions**
- Classes: PascalCase (`PredictionService`, `BaseRankingModel`)
- Functions/variables: snake_case (`predict`, `feature_cols`)
- Constants: UPPER_SNAKE_CASE (`DEFAULT_TIMEOUT`, `API_BASE_URL`)
- Private members: `_prefix` (`_prepare_input`, `is_fitted_`)

**Type Hints**
- Use `list[T]` (Python 3.9+) over `List[T]` for collections
- Use `Optional[T]` for nullable values
- Always type function signatures
```python
def predict(self, symbol: str, timeframe: str = "D") -> Prediction:
def batch_predict(self, symbols: Optional[list[str]] = None) -> list[Prediction]:
```

**Error Handling/Logging**: Specific exceptions, log context before re-raising

**Docstrings**: Google/NumPy-style with Args, Returns, Raises sections

**FastAPI Specific**
- Route order matters: `/batch` before `/{symbol}` to avoid conflicts
- Use `Depends()` for dependency injection, `Query()` for parameters
- Response models with Pydantic v2 schemas
- Use `model_validate()` and `model_dump()` for Pydantic v2

**SQLAlchemy**
- Use context managers when possible
- Commit/refresh after write operations

**etf-model Specific**
- Inherit from `BaseRankingModel` for all ranking models
- Use factory pattern: `create_model(model_name, **kwargs)`
- Data leakage prevention: 95-day cutoff before prediction year
- Feature selection by correlation with target
- Competition rules: single model per year, same preprocessing

### TypeScript/Next.js (web-dashboard)

**Imports**: React/Next.js → third-party → UI components → local lib

**Component Structure**
- `"use client"` directive for client components
- Functional components with hooks
- Use shadcn/ui components from `@/components/ui`

**Naming**
- Components: PascalCase (`PredictionsPage`, `Button`)
- Functions/variables: camelCase (`loadPredictions`, `data`)
- Types: PascalCase (`Prediction`, `APIPrediction`)
- Constants: UPPER_SNAKE_CASE at module level

**TypeScript**
- Use `interface` for object shapes, `type` for unions
- Use `import type { ... }` for types-only imports
- Explicit typing for props and state

**Error Handling/Styling**: try/catch with user-friendly messages; Tailwind + `cn()` utility

**API Calls**
- Use `getApiBaseUrl()` for dynamic URL resolution
- Always add `{ cache: "no-store" }` to fetch calls for fresh data

## Project-Specific Conventions

**Database Access**
- Remote MySQL via SSH tunnel: `host.docker.internal:3306`
- Local SQLite: `./ml-service/data/predictions.db`
- Use SQLAlchemy ORM, no raw SQL queries

**Docker**
- Services: ml-service (8000), web-dashboard (3000), nginx (80)
- Use `host.docker.internal` for host access from containers
- Healthcheck on `/health` endpoint for ml-service

**Environment Variables**
- Store in `.env` (not committed)
- Python: use `python-dotenv`, Next.js: `NEXT_PUBLIC_*` for client access
- See `.env.example` for required variables

**Testing**
- Currently no formal tests
- Add pytest for Python with `--cov` for coverage
- Add Jest/Vitest for Next.js component tests
