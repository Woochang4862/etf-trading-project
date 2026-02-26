"""
Experiment Logger for tracking and comparing ML model experiments

Usage:
    from src.utils.experiment_logger import ExperimentLogger

    logger = ExperimentLogger()
    logger.log_experiment(
        model_name='xgboost',
        params={'max_depth': 8},
        results={2020: 0.15, 2021: 0.12},
        duration=3600
    )
    logger.print_comparison_table()
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class ExperimentLogger:
    """
    Logger for experiment tracking and comparison

    Saves experiments to JSONL file for easy querying.
    """

    def __init__(self, log_dir: Path = None):
        """
        Initialize logger

        Args:
            log_dir: Directory for experiment logs
        """
        if log_dir is None:
            log_dir = Path(__file__).parent.parent.parent / "experiments"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.log_file = self.log_dir / "experiments.jsonl"
        self.results_dir = self.log_dir / "results"
        self.results_dir.mkdir(exist_ok=True)

    def log_experiment(
        self,
        model_name: str,
        params: Dict[str, Any],
        results: Dict[int, float],
        feature_count: int = None,
        train_samples: int = None,
        duration: float = None,
        notes: str = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Log an experiment result

        Args:
            model_name: Name of the model
            params: Hyperparameters used
            results: Dict mapping year -> score
            feature_count: Number of features used
            train_samples: Number of training samples
            duration: Training duration in seconds
            notes: Optional notes about the experiment
            metadata: Additional metadata

        Returns:
            Experiment ID
        """
        timestamp = datetime.now()
        exp_id = f"{model_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        # Calculate average score
        if results:
            avg_score = np.mean(list(results.values()))
        else:
            avg_score = None

        entry = {
            'id': exp_id,
            'timestamp': timestamp.isoformat(),
            'model': model_name,
            'params': params,
            'results': {str(k): v for k, v in results.items()},  # JSON keys must be strings
            'avg_score': avg_score,
            'feature_count': feature_count,
            'train_samples': train_samples,
            'duration_seconds': duration,
            'duration_minutes': duration / 60 if duration else None,
            'notes': notes,
            'metadata': metadata or {}
        }

        # Append to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        # Also save as individual result file
        result_file = self.results_dir / f"{exp_id}.json"
        with open(result_file, 'w') as f:
            json.dump(entry, f, indent=2)

        return exp_id

    def load_experiments(self) -> pd.DataFrame:
        """
        Load all experiments as DataFrame

        Returns:
            DataFrame with all experiment records
        """
        if not self.log_file.exists():
            return pd.DataFrame()

        records = []
        with open(self.log_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        # Flatten results for easier access
                        for year in ['2020', '2021', '2022', '2023', '2024']:
                            record[f'score_{year}'] = record.get('results', {}).get(year)
                        records.append(record)
                    except json.JSONDecodeError:
                        continue

        return pd.DataFrame(records)

    def get_best_model(self, metric: str = 'avg_score') -> Optional[Dict]:
        """
        Get best performing model configuration

        Args:
            metric: Metric to optimize ('avg_score' or 'score_YYYY')

        Returns:
            Best experiment record or None
        """
        df = self.load_experiments()
        if df.empty:
            return None

        best_idx = df[metric].idxmax()
        return df.loc[best_idx].to_dict()

    def get_model_history(self, model_name: str) -> pd.DataFrame:
        """
        Get history of experiments for a specific model

        Args:
            model_name: Name of the model

        Returns:
            DataFrame with model's experiment history
        """
        df = self.load_experiments()
        if df.empty:
            return df

        return df[df['model'] == model_name].sort_values('timestamp', ascending=False)

    def print_comparison_table(
        self,
        top_n: int = None,
        models: List[str] = None
    ):
        """
        Print comparison table of all experiments

        Args:
            top_n: Show only top N experiments by avg_score
            models: Filter to specific models
        """
        df = self.load_experiments()
        if df.empty:
            print("No experiments logged yet.")
            return

        # Filter models
        if models:
            df = df[df['model'].isin(models)]

        # Sort by avg_score
        df = df.sort_values('avg_score', ascending=False)

        # Limit to top_n
        if top_n:
            df = df.head(top_n)

        # Print table
        print("\n" + "="*90)
        print("EXPERIMENT RESULTS")
        print("="*90)

        header = f"{'Model':<20} {'2020':>7} {'2021':>7} {'2022':>7} {'2023':>7} {'2024':>7} {'Avg':>7} {'Time':>8}"
        print(header)
        print("-"*90)

        for _, row in df.iterrows():
            model = row['model'][:18]
            s2020 = f"{row.get('score_2020', '-'):.3f}" if row.get('score_2020') else '-'
            s2021 = f"{row.get('score_2021', '-'):.3f}" if row.get('score_2021') else '-'
            s2022 = f"{row.get('score_2022', '-'):.3f}" if row.get('score_2022') else '-'
            s2023 = f"{row.get('score_2023', '-'):.3f}" if row.get('score_2023') else '-'
            s2024 = f"{row.get('score_2024', '-'):.3f}" if row.get('score_2024') else '-'
            avg = f"{row.get('avg_score', 0):.3f}" if row.get('avg_score') else '-'
            time_min = f"{row.get('duration_minutes', 0):.1f}m" if row.get('duration_minutes') else '-'

            print(f"{model:<20} {s2020:>7} {s2021:>7} {s2022:>7} {s2023:>7} {s2024:>7} {avg:>7} {time_min:>8}")

        print("="*90)

    def print_model_params(self, model_name: str = None):
        """
        Print parameter comparison for models

        Args:
            model_name: Specific model to show (None for all)
        """
        df = self.load_experiments()
        if df.empty:
            print("No experiments logged yet.")
            return

        if model_name:
            df = df[df['model'] == model_name]

        print("\n" + "="*60)
        print("MODEL PARAMETERS")
        print("="*60)

        for _, row in df.iterrows():
            print(f"\n{row['model']} ({row.get('id', 'N/A')})")
            print(f"  Score: {row.get('avg_score', 'N/A'):.4f}")
            print(f"  Features: {row.get('feature_count', 'N/A')}")
            print(f"  Samples: {row.get('train_samples', 'N/A')}")
            if row.get('params'):
                print(f"  Params: {json.dumps(row['params'], indent=4)}")
            if row.get('notes'):
                print(f"  Notes: {row['notes']}")

    def export_to_csv(self, filepath: Path = None):
        """
        Export experiments to CSV

        Args:
            filepath: Output path (default: experiments/summary.csv)
        """
        df = self.load_experiments()
        if df.empty:
            print("No experiments to export.")
            return

        if filepath is None:
            filepath = self.log_dir / "summary.csv"

        # Select relevant columns
        columns = [
            'id', 'timestamp', 'model', 'avg_score',
            'score_2020', 'score_2021', 'score_2022', 'score_2023', 'score_2024',
            'feature_count', 'train_samples', 'duration_minutes', 'notes'
        ]
        available = [c for c in columns if c in df.columns]

        df[available].to_csv(filepath, index=False)
        print(f"Exported to {filepath}")

    def clear_experiments(self, confirm: bool = False):
        """
        Clear all experiment logs

        Args:
            confirm: Must be True to actually delete
        """
        if not confirm:
            print("Pass confirm=True to actually delete experiment logs.")
            return

        if self.log_file.exists():
            self.log_file.unlink()
            print("Cleared experiments.jsonl")

        for f in self.results_dir.glob("*.json"):
            f.unlink()
        print("Cleared results directory")


def add_baseline_results():
    """Add baseline results for comparison"""
    logger = ExperimentLogger()

    # LightGBM LambdaRank baseline
    logger.log_experiment(
        model_name='lightgbm_lambdarank',
        params={'objective': 'lambdarank', 'num_leaves': 63, 'learning_rate': 0.05},
        results={2020: 0.168, 2021: 0.113, 2022: 0.112, 2023: 0.135, 2024: 0.168},
        feature_count=242,
        notes='Baseline LambdaRank model'
    )

    # TabPFN V2 baseline
    logger.log_experiment(
        model_name='tabpfn_v2',
        params={'n_estimators': 8, 'features': 150},
        results={2020: 0.163, 2021: 0.151, 2022: 0.154, 2023: 0.190, 2024: 0.185},
        feature_count=150,
        notes='TabPFN V2 - current best'
    )

    print("Added baseline results.")
    logger.print_comparison_table()


if __name__ == "__main__":
    # Add baselines when run directly
    add_baseline_results()
