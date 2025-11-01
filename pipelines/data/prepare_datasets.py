#!/usr/bin/env python3
"""Prepare classification datasets with stratified splits."""
import sys
import csv
import logging
from pathlib import Path
from collections import Counter
from typing import List, Tuple
from sklearn.model_selection import train_test_split

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Now we can import from src
from src.application.config.loader import ConfigLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetPreparator:
    """Prepare classification datasets."""

    def __init__(self, config: dict):
        self.config = config
        self.small_cluster_threshold = config.get("small_cluster_threshold", 30)
        self.merge_small = config.get("merge_small_clusters", True)
        self.train_ratio = config["data_split"]["train_ratio"]
        self.val_ratio = config["data_split"]["val_ratio"]
        self.test_ratio = config["data_split"]["test_ratio"]
        self.random_state = config["data_split"]["random_state"]
        self.stratify = config["data_split"]["stratify"]

    def prepare(self, input_file: Path, output_dir: Path):
        """
        Prepare datasets with splits.

        Args:
            input_file: Path to questions_with_clusters.csv
            output_dir: Output directory for splits
        """
        logger.info(f"Preparing datasets from {input_file}")

        # Load data
        questions, clusters = self._load_csv(input_file)

        # Merge small clusters if enabled
        if self.merge_small:
            clusters = self._merge_small_clusters(clusters)

        # Check class distribution
        self._print_distribution(clusters)

        # Create stratified splits
        train_q, train_c, val_q, val_c, test_q, test_c = self._create_splits(
            questions, clusters
        )

        # Save splits
        output_dir.mkdir(parents=True, exist_ok=True)

        self._save_csv(output_dir / "train.csv", train_q, train_c)
        self._save_csv(output_dir / "val.csv", val_q, val_c)
        self._save_csv(output_dir / "test.csv", test_q, test_c)

        logger.info(f"Datasets saved to {output_dir}")
        logger.info(f"  Train: {len(train_q)} samples")
        logger.info(f"  Val: {len(val_q)} samples")
        logger.info(f"  Test: {len(test_q)} samples")

    def _load_csv(self, filepath: Path) -> Tuple[List[str], List[str]]:
        """Load questions and clusters from CSV."""
        questions, clusters = [], []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    questions.append(row['question'])
                    clusters.append(row['cluster_name'])

            logger.info(f"Loaded {len(questions)} samples")
            return questions, clusters

        except Exception as e:
            raise RuntimeError(f"Failed to load {filepath}: {str(e)}")

    def _merge_small_clusters(self, clusters: List[str]) -> List[str]:
        """Merge clusters with < threshold samples into 'Miscellaneous'."""
        cluster_counts = Counter(clusters)
        small_clusters = [
            c for c, count in cluster_counts.items()
            if count < self.small_cluster_threshold
        ]

        if not small_clusters:
            logger.info("No small clusters to merge")
            return clusters

        # Merge
        merged = [
            "Miscellaneous" if c in small_clusters else c
            for c in clusters
        ]

        # Report
        logger.info(f"Merged {len(small_clusters)} small clusters into 'Miscellaneous':")
        for cluster in sorted(small_clusters):
            logger.info(f"  - {cluster} ({cluster_counts[cluster]} samples)")

        misc_count = sum(cluster_counts[c] for c in small_clusters)
        logger.info(f"Total samples in 'Miscellaneous': {misc_count}")

        return merged

    def _print_distribution(self, clusters: List[str]):
        """Print cluster distribution."""
        cluster_counts = Counter(clusters)
        logger.info(f"\nCluster distribution ({len(cluster_counts)} unique clusters):")

        for cluster, count in cluster_counts.most_common():
            logger.info(f"  {cluster}: {count}")

    def _create_splits(
        self,
        questions: List[str],
        clusters: List[str]
    ) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
        """Create stratified train/val/test splits."""
        # First split: train vs (val + test)
        val_test_ratio = self.val_ratio + self.test_ratio

        train_q, temp_q, train_c, temp_c = train_test_split(
            questions,
            clusters,
            test_size=val_test_ratio,
            stratify=clusters if self.stratify else None,
            random_state=self.random_state
        )

        # Second split: val vs test
        val_ratio_adjusted = self.val_ratio / val_test_ratio

        val_q, test_q, val_c, test_c = train_test_split(
            temp_q,
            temp_c,
            test_size=1 - val_ratio_adjusted,
            stratify=temp_c if self.stratify else None,
            random_state=self.random_state
        )

        return train_q, train_c, val_q, val_c, test_q, test_c

    def _save_csv(self, filepath: Path, questions: List[str], clusters: List[str]):
        """Save questions and clusters to CSV."""
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['question', 'cluster_name'])
            for q, c in zip(questions, clusters):
                writer.writerow([q, c])

        logger.info(f"Saved {len(questions)} samples to {filepath}")


def main():
    """Main execution."""
    print("=" * 70)
    print("Dataset Preparation for Classification")
    print("=" * 70)

    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load("classification")

    # Prepare datasets
    preparator = DatasetPreparator(config)

    input_file = PROJECT_ROOT / "datasets/raw/questions_with_clusters.csv"
    output_dir = PROJECT_ROOT / "datasets/processed/classification"

    preparator.prepare(input_file, output_dir)

    print("\n" + "=" * 70)
    print("✓ Dataset preparation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
