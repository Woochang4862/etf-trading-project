"""
Factsheet Router - Snowballing AI ETF 팩트시트 API

월별 팩트시트 조회/생성 엔드포인트 제공
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional

from app.database import get_local_db
from app.services.factsheet_service import FactsheetService
from app.schemas import (
    FactsheetListResponse,
    FactsheetListItem,
    FactsheetGenerateRequest,
    FactsheetGenerateResponse,
    ETFMonthlySnapshotResponse,
    ETFCompositionResponse
)

router = APIRouter()


def get_factsheet_service(
    local_db: Session = Depends(get_local_db)
) -> FactsheetService:
    return FactsheetService(local_db)


@router.get("", response_model=FactsheetListResponse)
def list_factsheets(
    service: FactsheetService = Depends(get_factsheet_service)
):
    """
    모든 팩트시트 목록 조회 (최신순)

    Returns:
        - count: 총 팩트시트 수
        - factsheets: 팩트시트 목록 (year, month, snapshot_date)
    """
    factsheets = service.list_factsheets()
    return FactsheetListResponse(
        count=len(factsheets),
        factsheets=[FactsheetListItem(**f) for f in factsheets]
    )


@router.get("/latest")
def get_latest_factsheet(
    service: FactsheetService = Depends(get_factsheet_service)
):
    """
    가장 최신 팩트시트 조회

    Returns:
        - 최신 팩트시트 (구성 종목 포함)
    """
    factsheets = service.list_factsheets()
    if not factsheets:
        raise HTTPException(status_code=404, detail="No factsheets available")

    latest = factsheets[0]  # 이미 최신순 정렬됨
    factsheet = service.get_factsheet_with_compositions(latest["year"], latest["month"])

    if not factsheet:
        raise HTTPException(status_code=404, detail="Factsheet not found")

    return factsheet


@router.get("/{year}/{month}")
def get_factsheet(
    year: int,
    month: int,
    service: FactsheetService = Depends(get_factsheet_service)
):
    """
    특정 월 팩트시트 조회

    Args:
        - year: 연도 (2020-2024)
        - month: 월 (1-12)

    Returns:
        - 팩트시트 상세 정보 (구성 종목 포함)
    """
    if month < 1 or month > 12:
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12")

    factsheet = service.get_factsheet_with_compositions(year, month)

    if not factsheet:
        raise HTTPException(
            status_code=404,
            detail=f"Factsheet not found for {year}-{month:02d}"
        )

    return factsheet


@router.post("/generate", response_model=FactsheetGenerateResponse)
def generate_factsheet(
    request: FactsheetGenerateRequest,
    service: FactsheetService = Depends(get_factsheet_service)
):
    """
    팩트시트 생성

    Args:
        - year: 연도
        - month: 월

    Returns:
        - success: 성공 여부
        - message: 결과 메시지
        - snapshot: 생성된 팩트시트
    """
    try:
        snapshot = service.generate_factsheet(request.year, request.month)

        # Response 모델로 변환
        compositions = [
            ETFCompositionResponse(
                rank=c.rank,
                ticker=c.ticker,
                weight=c.weight,
                stock_name=c.stock_name,
                sector=c.sector
            )
            for c in snapshot.compositions
        ]

        snapshot_response = ETFMonthlySnapshotResponse(
            id=snapshot.id,
            year=snapshot.year,
            month=snapshot.month,
            snapshot_date=snapshot.snapshot_date,
            nav=snapshot.nav,
            monthly_return=snapshot.monthly_return,
            ytd_return=snapshot.ytd_return,
            volatility=snapshot.volatility,
            sharpe_ratio=snapshot.sharpe_ratio,
            max_drawdown=snapshot.max_drawdown,
            compositions=compositions
        )

        return FactsheetGenerateResponse(
            success=True,
            message=f"Factsheet generated for {request.year}-{request.month:02d}",
            snapshot=snapshot_response
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-all")
def generate_all_factsheets(
    years: Optional[list[int]] = Query(default=None, description="연도 목록 (기본: 2020-2024)"),
    service: FactsheetService = Depends(get_factsheet_service)
):
    """
    모든 과거 팩트시트 일괄 생성 (2020-2024)

    Args:
        - years: 생성할 연도 목록 (기본: [2020, 2021, 2022, 2023, 2024])

    Returns:
        - total: 총 시도 수
        - success: 성공 수
        - failed: 실패 수
        - results: 개별 결과 목록
    """
    if years is None:
        years = [2020, 2021, 2022, 2023, 2024]

    results = service.generate_all_historical(years)
    success_count = sum(1 for r in results if r.get('success'))

    return {
        "total": len(results),
        "success": success_count,
        "failed": len(results) - success_count,
        "results": results
    }


@router.delete("/{year}/{month}")
def delete_factsheet(
    year: int,
    month: int,
    service: FactsheetService = Depends(get_factsheet_service)
):
    """
    팩트시트 삭제

    Args:
        - year: 연도
        - month: 월

    Returns:
        - success: 삭제 성공 여부
        - message: 결과 메시지
    """
    deleted = service.delete_factsheet(year, month)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Factsheet not found for {year}-{month:02d}"
        )

    return {
        "success": True,
        "message": f"Factsheet deleted for {year}-{month:02d}"
    }
