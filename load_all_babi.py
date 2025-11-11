from datasets import load_dataset
import sys

configs = (
    [f"en-qa{i}" for i in range(1, 21)]
    + [f"en-10k-qa{i}" for i in range(1, 21)]
    + [f"en-valid-qa{i}" for i in range(1, 21)]
    + [f"en-valid-10k-qa{i}" for i in range(1, 21)]
    + [f"hn-qa{i}" for i in range(1, 21)]
    + [f"hn-10k-qa{i}" for i in range(1, 21)]
    + [f"shuffled-qa{i}" for i in range(1, 21)]
    + [f"shuffled-10k-qa{i}" for i in range(1, 21)]
)

print(f"Loading {len(configs)} configs...")
all_datasets = {}

for cfg in configs:
    print(f"-> {cfg}")
    sys.stdout.flush()
    all_datasets[cfg] = load_dataset('./babi_qa', cfg, trust_remote_code=True)

print("\nLoaded configs:")
for cfg, ds in all_datasets.items():
    train_len = len(ds['train']) if 'train' in ds else 0
    test_len = len(ds['test']) if 'test' in ds else 0
    print(f"{cfg:20s} train={train_len:5d} test={test_len:5d}")
