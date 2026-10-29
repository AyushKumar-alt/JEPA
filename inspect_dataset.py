from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path("data/bev-decision")

train_path = DATA_DIR / "train.parquet"
test_path = DATA_DIR / "test.parquet"

for path in (train_path, test_path):
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Dataset file is empty: {path}")

dataset = load_dataset(
    "parquet",
    data_files={
        "train": str(train_path),
        "test": str(test_path),
    },
)

print("\n========== DATASET SUMMARY ==========")
print(dataset)

for split in dataset:
    print(f"\n========== {split.upper()} ==========")
    print("Rows:", len(dataset[split]))
    print("Columns:", dataset[split].column_names)

print("\n========== FIRST TRAINING EXAMPLE ==========")
example = dataset["train"][0]

for key, value in example.items():
    print(f"\n{key}:")
    print(str(value)[:2000])

print("\n========== INSPECTION COMPLETE ==========")
