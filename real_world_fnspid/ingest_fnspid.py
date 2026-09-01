import os
import pandas as pd
from datasets import load_dataset

# Setup local paths
data_dir = "./real_world_fnspid/data"
os.makedirs(data_dir, exist_ok=True)
output_path = os.path.join(data_dir, "fnspid_processed_slice.csv")

print("Initializing Hugging Face stream for Zihan1004/FNSPID...")
# Stream data dynamically without downloading the full 23+ GB database
dataset_stream = load_dataset("Zihan1004/FNSPID", streaming=True)

row_limit = 5000  # Extract a clean, sequential 5000-step time-series slice
sampled_rows = []

for i, row in enumerate(dataset_stream["train"]):
    sampled_rows.append(row)
    if i + 1 >= row_limit:
        break

df = pd.DataFrame(sampled_rows)
df.to_csv(output_path, index=False)
print(f"Extraction complete. Saved {len(df)} rows to {output_path}")
