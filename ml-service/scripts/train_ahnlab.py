#!/usr/bin/env python3
"""
Train AhnLab LightGBM LambdaRank model from etf2_db_processed data.

Reads features from the processed database, trains a 2-fold rolling CV
ensemble, and saves the model files to ml-service/data/models/ahnlab_lgbm/.

Usage:
    # From ml-service directory (or Docker container)
    python scripts/train_ahnlab.py

    # With custom pred_year
    python scripts/train_ahnlab.py --pred-year 2025

    # With GPU
    python scripts/train_ahnlab.py --device gpu
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add parent directory to path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import lightgbm as lgb
except ImportError:
    logger.error("lightgbm is required. Install with: pip install lightgbm")
    sys.exit(1)

# ── Feature columns (from ahnlab_constants.py) ──

BASE_FEATURE_COLS = [
    "open", "high", "low", "close", "volume", "dividends", "stock_splits",
    "ret_1d", "ret_5d", "ret_20d", "ret_63d",
    "macd", "macd_signal", "macd_hist",
    "rsi_14", "rsi_28",
    "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_position",
    "atr_14", "obv",
    "ema_10", "ema_20", "ema_50", "ema_200",
    "sma_10", "sma_20", "sma_50",
    "stoch_k", "stoch_d", "adx", "cci", "willr", "mfi", "vwap",
    "volume_sma_20", "volume_ratio",
    "vix", "fed_funds_rate", "unemployment_rate", "cpi",
    "treasury_10y", "treasury_2y", "yield_curve",
    "oil_price", "usd_eur", "high_yield_spread",
]

ENGINEERED_FEATURE_COLS = [
    "ret_10d", "ret_30d", "vol_20d", "vol_63d",
    "price_to_sma_50", "price_to_ema_200", "volume_trend",
    "close_to_high_52w", "ret_5d_20d_ratio", "momentum_strength",
    "volume_surge", "ret_vol_ratio_20d", "ret_vol_ratio_63d",
    "trend_acceleration", "close_to_high_20d", "close_to_high_63d",
    "close_to_high_126d",
    "ema_5", "ema_100", "price_to_ema_10", "price_to_ema_50",
    "ema_cross_short", "ema_cross_long", "ema_slope_20",
]

ZS_BASE_COLS = [
    "vol_63d", "volume_sma_20", "obv", "vwap",
    "ema_200", "price_to_ema_200", "close_to_high_52w",
]
ZS_FEATURE_COLS = [f"{col}_zs" for col in ZS_BASE_COLS]

RANK_BASE_COLS = [
    "ret_20d", "ret_63d", "vol_20d", "momentum_strength", "volume_surge",
]
RANK_FEATURE_COLS = [f"{col}_rank" for col in RANK_BASE_COLS]

ALL_FEATURE_COLS = BASE_FEATURE_COLS + ENGINEERED_FEATURE_COLS + ZS_FEATURE_COLS + RANK_FEATURE_COLS

# ── LightGBM hyperparameters ──

LGB_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "label_gain": [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 30, 33, 36, 39, 43, 47, 52, 56, 61, 66,
        71, 77, 83, 89, 95, 101, 107, 114, 121, 128, 135, 142,
    ],
    "learning_rate": 0.05,
    "max_depth": -1,
    "num_leaves": 45,
    "min_child_samples": 60,
    "subsample": 0.7,
    "subsample_freq": 5,
    "colsample_bytree": 0.65,
    "min_split_gain": 0.00,
    "reg_alpha": 0.8,
    "reg_lambda": 1.2,
    "verbosity": 1,
}

NUM_BOOST_ROUND = 5000
EARLY_STOPPING_ROUNDS = 150
TOP_K = 100
RELEVANCE_BINS = 50
SEED = 42
MIN_HISTORY_DAYS = 126


def get_db_url() -> str:
    """Build processed DB URL."""
    url = os.getenv("PROCESSED_DB_URL")
    if url:
        return url
    host = os.getenv("MYSQL_HOST", "host.docker.internal")
    port = os.getenv("MYSQL_PORT", "3306")
    user = os.getenv("MYSQL_USER", "ahnbi2")
    password = os.getenv("MYSQL_PASSWORD", "bigdata")
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/etf2_db_processed"


def load_panel_from_db(engine) -> pd.DataFrame:
    """Load all symbol data from etf2_db_processed into a panel DataFrame."""
    # Get list of daily tables
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
                "WHERE TABLE_SCHEMA = 'etf2_db_processed' "
                "AND TABLE_NAME LIKE '%_D'"
            )
        )
        tables = [row[0] for row in result]

    logger.info(f"Found {len(tables)} daily tables")

    cols_str = ", ".join(f"`{c}`" for c in ALL_FEATURE_COLS)
    all_dfs = []

    for table_name in tables:
        symbol = table_name.replace("_D", "")
        try:
            query = text(
                f"SELECT `time`, `target_3m`, {cols_str} "
                f"FROM `{table_name}` ORDER BY `time`"
            )
            df = pd.read_sql(query, engine)
            if df.empty:
                continue
            df["ticker"] = symbol
            df.rename(columns={"time": "date"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            all_dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {table_name}: {e}")
            continue

    if not all_dfs:
        raise RuntimeError("No data loaded from processed DB")

    panel = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Panel loaded: {len(panel)} rows, {panel['ticker'].nunique()} symbols")
    return panel


def add_relevance_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add quantile-binned relevance labels for LambdaRank."""
    df = df.copy()

    def _label(series: pd.Series) -> pd.Series:
        q = min(RELEVANCE_BINS, series.shape[0])
        if q <= 1:
            return pd.Series(0, index=series.index, dtype=int)
        ranks = series.rank(method="first")
        labels = pd.qcut(ranks, q=q, labels=False, duplicates="drop")
        return labels.fillna(0).astype(int)

    df["relevance"] = df.groupby("date")["target_3m"].transform(_label)
    df["relevance"] = df["relevance"].fillna(0).astype(int)
    return df


def get_rolling_folds(pred_year: int):
    """Get rolling 2-fold CV configuration."""
    valid_year_1 = pred_year - 3
    valid_year_2 = pred_year - 1
    return [
        {
            "train_end": pd.Timestamp(f"{valid_year_1 - 1}-12-31"),
            "valid_start": pd.Timestamp(f"{valid_year_1}-01-01"),
            "valid_end": pd.Timestamp(f"{valid_year_1}-12-31"),
        },
        {
            "train_end": pd.Timestamp(f"{valid_year_2 - 1}-12-31"),
            "valid_start": pd.Timestamp(f"{valid_year_2}-01-01"),
            "valid_end": pd.Timestamp(f"{valid_year_2}-12-31"),
        },
    ]


def prepare_window(panel, start, end, train_end, feature_cols):
    """Prepare a data window with relevance labels."""
    mask = (panel["date"] >= start) & (panel["date"] <= end)
    df = panel.loc[mask].copy()

    if df.empty:
        return df

    # Clean features
    avail_cols = [c for c in feature_cols if c in df.columns]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["target_3m"], inplace=True)
    df[avail_cols] = df[avail_cols].fillna(0)

    # Add relevance labels
    df = add_relevance_labels(df)
    return df


def train_single_ranker(train_df, valid_df, feature_cols, params, device="cpu"):
    """Train a single LGBMRanker model."""
    avail_cols = [c for c in feature_cols if c in train_df.columns]

    train_sorted = train_df.sort_values(["date", "ticker"]).reset_index(drop=True)
    valid_sorted = valid_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    X_train = train_sorted[avail_cols]
    y_train = train_sorted["relevance"]
    train_groups = train_sorted.groupby("date").size().astype(int).tolist()

    X_valid = valid_sorted[avail_cols]
    y_valid = valid_sorted["relevance"]
    valid_groups = valid_sorted.groupby("date").size().astype(int).tolist()

    full_params = {**params, "device": device}
    if device == "gpu":
        full_params["gpu_platform_id"] = 0
        full_params["gpu_device_id"] = 0

    evals_result = {}
    model = lgb.LGBMRanker(
        **full_params,
        n_estimators=NUM_BOOST_ROUND,
        random_state=SEED,
    )

    model.fit(
        X_train, y_train,
        group=train_groups,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_group=[train_groups, valid_groups],
        eval_names=["train", "valid"],
        eval_at=[TOP_K],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=50),
            lgb.record_evaluation(evals_result),
        ],
    )

    return model, evals_result


def main():
    parser = argparse.ArgumentParser(description="Train AhnLab LightGBM ranking model")
    parser.add_argument("--pred-year", type=int, default=datetime.now().year,
                        help="Prediction year (default: current year)")
    parser.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu",
                        help="Compute device")
    parser.add_argument("--train-start", default="2010-01-01",
                        help="Training data start date")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: ml-service/data/models/ahnlab_lgbm/)")
    args = parser.parse_args()

    pred_year = args.pred_year
    device = args.device
    if device == "auto":
        try:
            import subprocess
            result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
            device = "gpu" if result.returncode == 0 else "cpu"
        except Exception:
            device = "cpu"

    logger.info(f"Training AhnLab LightGBM for pred_year={pred_year}, device={device}")

    # Output directory
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = Path(__file__).parent.parent / "data" / "models" / "ahnlab_lgbm"

    version_name = f"v{datetime.now().strftime('%Y%m%d')}"
    version_dir = output_base / version_name
    version_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    db_url = get_db_url()
    logger.info(f"Connecting to: {db_url.split('@')[0]}@...")
    engine = create_engine(db_url, pool_pre_ping=True)

    panel = load_panel_from_db(engine)

    # Shift features by 1 day to prevent leakage
    logger.info("Shifting features by 1 day...")
    grouped = panel.groupby("ticker")
    for col in ALL_FEATURE_COLS:
        if col in panel.columns:
            panel[col] = grouped[col].shift(1)

    # Train rolling 2-fold CV
    folds = get_rolling_folds(pred_year)
    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(f"{pred_year - 1}-12-31")

    models = []
    for idx, fold in enumerate(folds, start=1):
        logger.info(
            f"Fold {idx}: valid {fold['valid_start'].date()} ~ {fold['valid_end'].date()}"
        )

        train_df = prepare_window(
            panel, train_start, fold["train_end"], train_end, ALL_FEATURE_COLS
        )
        valid_df = prepare_window(
            panel, fold["valid_start"], fold["valid_end"], train_end, ALL_FEATURE_COLS
        )

        if train_df.empty or valid_df.empty:
            logger.warning(f"Skipping fold {idx}: insufficient data")
            continue

        logger.info(f"Fold {idx}: {len(train_df)} train rows, {len(valid_df)} valid rows")

        model, evals_result = train_single_ranker(
            train_df, valid_df, ALL_FEATURE_COLS, LGB_PARAMS, device
        )
        models.append(model)

        best_iter = getattr(model, "best_iteration_", None)
        logger.info(f"Fold {idx} done, best_iteration={best_iter}")

    if not models:
        logger.error("No models trained!")
        sys.exit(1)

    # Save fold files
    for i, model in enumerate(models):
        fold_path = version_dir / f"ahnlab_lgbm_fold{i}.txt"
        model.booster_.save_model(str(fold_path))
        logger.info(f"Saved: {fold_path}")

    # Save metadata
    metadata = {
        "version": version_name,
        "trained_at": datetime.now().isoformat(),
        "train_data_end": str(train_end.date()),
        "pred_year": pred_year,
        "n_symbols": panel["ticker"].nunique(),
        "n_features": len(ALL_FEATURE_COLS),
        "n_folds": len(models),
        "n_train_rows": len(panel),
        "lgb_params": LGB_PARAMS,
        "device": device,
        "format": "txt",
    }
    with open(version_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Update current symlink
    current_link = output_base / "current"
    if current_link.is_symlink():
        current_link.unlink()
    elif current_link.exists():
        import shutil
        shutil.rmtree(current_link)
    current_link.symlink_to(version_name)
    logger.info(f"Updated current -> {version_name}")

    # Update versions.json
    versions_file = output_base / "versions.json"
    if versions_file.exists():
        with open(versions_file) as f:
            versions_data = json.load(f)
    else:
        versions_data = {"current": None, "versions": []}

    # Mark previous active as archived
    for v in versions_data["versions"]:
        if v.get("status") == "active":
            v["status"] = "archived"

    versions_data["current"] = version_name
    versions_data["versions"].insert(0, {
        "version": version_name,
        "trained_at": datetime.now().isoformat(),
        "status": "active",
    })

    with open(versions_file, "w") as f:
        json.dump(versions_data, f, indent=2)

    logger.info(f"Training complete! Model saved to {version_dir}")
    logger.info(f"  Folds: {len(models)}")
    logger.info(f"  Symbols: {panel['ticker'].nunique()}")
    logger.info(f"  Features: {len(ALL_FEATURE_COLS)}")


if __name__ == "__main__":
    main()
