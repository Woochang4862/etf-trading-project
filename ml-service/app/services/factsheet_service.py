"""
Factsheet Service - Snowballing AI ETF 팩트시트 생성 및 조회

submission CSV에서 월말 Top-10 종목을 추출하여 팩트시트를 생성합니다.
"""

import pandas as pd
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional
import os
import logging

from app.models import ETFMonthlySnapshot, ETFComposition

logger = logging.getLogger(__name__)

# Docker 환경에서는 /app/submissions, 로컬에서는 상대 경로
SUBMISSIONS_DIR = os.environ.get("SUBMISSIONS_DIR", "/app/submissions")


class FactsheetService:
    """팩트시트 생성 및 조회 서비스"""

    def __init__(self, local_db: Session):
        self.local_db = local_db

    def load_submission_csv(self, year: int) -> pd.DataFrame:
        """제출 CSV 파일 로드 (tabpfn_v2 우선)"""
        # 파일 패턴 우선순위
        patterns = [
            f"{year}.tabpfn_v2.submission.csv",
            f"{year}.tabpfn.submission.csv",
            f"{year}.submission.csv",
        ]

        for pattern in patterns:
            filepath = os.path.join(SUBMISSIONS_DIR, pattern)
            if os.path.exists(filepath):
                logger.info(f"Loading submission file: {filepath}")
                df = pd.read_csv(filepath)
                df['date'] = pd.to_datetime(df['date'])
                return df

        raise FileNotFoundError(f"No submission file found for year {year} in {SUBMISSIONS_DIR}")

    def get_last_trading_day(self, year: int, month: int, df: pd.DataFrame) -> Optional[str]:
        """해당 월의 마지막 거래일 반환"""
        month_data = df[(df['date'].dt.year == year) & (df['date'].dt.month == month)]
        if month_data.empty:
            return None
        return month_data['date'].max().strftime('%Y-%m-%d')

    def extract_monthly_top10(self, year: int, month: int) -> list[dict]:
        """월말 Top-10 종목 추출"""
        df = self.load_submission_csv(year)

        # 해당 월 마지막 거래일
        last_day = self.get_last_trading_day(year, month, df)
        if last_day is None:
            logger.warning(f"No trading data for {year}-{month:02d}")
            return []

        # 마지막 거래일의 Top-10 (rank 1-10)
        top10 = df[df['date'] == last_day].sort_values('rank').head(10)

        return [
            {"rank": int(row['rank']), "ticker": row['ticker'], "weight": 10.0}
            for _, row in top10.iterrows()
        ]

    def generate_factsheet(self, year: int, month: int) -> ETFMonthlySnapshot:
        """월별 팩트시트 생성 및 저장"""
        # 이미 존재하는지 확인
        existing = self.get_factsheet(year, month)
        if existing:
            logger.info(f"Factsheet already exists for {year}-{month:02d}")
            return existing

        # Top-10 구성 추출
        compositions = self.extract_monthly_top10(year, month)

        if not compositions:
            raise ValueError(f"No data found for {year}-{month:02d}")

        # 스냅샷 날짜 가져오기
        df = self.load_submission_csv(year)
        last_day = self.get_last_trading_day(year, month, df)
        snapshot_date = datetime.strptime(last_day, '%Y-%m-%d')

        # 스냅샷 생성
        snapshot = ETFMonthlySnapshot(
            year=year,
            month=month,
            snapshot_date=snapshot_date,
            # Performance metrics는 별도 계산 필요 (향후 확장)
        )

        self.local_db.add(snapshot)
        self.local_db.flush()  # ID 생성을 위해 flush

        # 구성 종목 추가
        for comp in compositions:
            composition = ETFComposition(
                snapshot_id=snapshot.id,
                rank=comp['rank'],
                ticker=comp['ticker'],
                weight=comp['weight']
            )
            self.local_db.add(composition)

        self.local_db.commit()
        self.local_db.refresh(snapshot)

        logger.info(f"Generated factsheet for {year}-{month:02d} with {len(compositions)} holdings")
        return snapshot

    def generate_all_historical(self, years: list[int] = None) -> list[dict]:
        """모든 과거 팩트시트 일괄 생성"""
        if years is None:
            years = [2020, 2021, 2022, 2023, 2024]

        results = []
        for year in years:
            for month in range(1, 13):
                try:
                    snapshot = self.generate_factsheet(year, month)
                    results.append({
                        "year": year,
                        "month": month,
                        "success": True,
                        "snapshot_id": snapshot.id
                    })
                except FileNotFoundError as e:
                    results.append({
                        "year": year,
                        "month": month,
                        "success": False,
                        "error": str(e)
                    })
                except ValueError as e:
                    # 해당 월에 데이터가 없는 경우 (예: 미래 월)
                    results.append({
                        "year": year,
                        "month": month,
                        "success": False,
                        "error": str(e)
                    })
                except Exception as e:
                    logger.error(f"Error generating factsheet for {year}-{month:02d}: {e}")
                    results.append({
                        "year": year,
                        "month": month,
                        "success": False,
                        "error": str(e)
                    })

        return results

    def list_factsheets(self) -> list[dict]:
        """모든 팩트시트 목록 조회 (최신순)"""
        snapshots = (
            self.local_db.query(ETFMonthlySnapshot)
            .order_by(ETFMonthlySnapshot.year.desc(), ETFMonthlySnapshot.month.desc())
            .all()
        )

        return [
            {
                "id": s.id,
                "year": s.year,
                "month": s.month,
                "snapshot_date": s.snapshot_date
            }
            for s in snapshots
        ]

    def get_factsheet(self, year: int, month: int) -> Optional[ETFMonthlySnapshot]:
        """특정 월 팩트시트 조회"""
        return (
            self.local_db.query(ETFMonthlySnapshot)
            .filter(
                ETFMonthlySnapshot.year == year,
                ETFMonthlySnapshot.month == month
            )
            .first()
        )

    def get_factsheet_with_compositions(self, year: int, month: int) -> Optional[dict]:
        """팩트시트 + 구성 종목 조회"""
        snapshot = self.get_factsheet(year, month)
        if not snapshot:
            return None

        compositions = (
            self.local_db.query(ETFComposition)
            .filter(ETFComposition.snapshot_id == snapshot.id)
            .order_by(ETFComposition.rank)
            .all()
        )

        return {
            "id": snapshot.id,
            "year": snapshot.year,
            "month": snapshot.month,
            "snapshot_date": snapshot.snapshot_date.isoformat(),
            "nav": snapshot.nav,
            "monthly_return": snapshot.monthly_return,
            "ytd_return": snapshot.ytd_return,
            "volatility": snapshot.volatility,
            "sharpe_ratio": snapshot.sharpe_ratio,
            "max_drawdown": snapshot.max_drawdown,
            "compositions": [
                {
                    "rank": c.rank,
                    "ticker": c.ticker,
                    "weight": c.weight,
                    "stock_name": c.stock_name,
                    "sector": c.sector
                }
                for c in compositions
            ]
        }

    def delete_factsheet(self, year: int, month: int) -> bool:
        """팩트시트 삭제"""
        snapshot = self.get_factsheet(year, month)
        if not snapshot:
            return False

        self.local_db.delete(snapshot)
        self.local_db.commit()
        logger.info(f"Deleted factsheet for {year}-{month:02d}")
        return True
