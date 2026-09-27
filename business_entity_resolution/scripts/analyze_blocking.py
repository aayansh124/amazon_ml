import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import polars as pl
from src.data_loader import load_data, normalize, create_blocking_keys

def analyze_blocking():
    data_dir = ".."
    if not os.path.exists(os.path.join(data_dir, "train_source1.tsv")):
        data_dir = "."
        
    s1_raw, s2_raw, s3_raw, gt_raw = load_data(data_dir, "train")
    
    # We will sample for fast analysis
    s1_raw = s1_raw.head(20000)
    s2_raw = s2_raw.head(100000)
    s3_raw = s3_raw.head(100000)
    
    s1 = normalize(s1_raw.lazy())
    s2 = normalize(s2_raw.lazy())
    s3 = normalize(s3_raw.lazy())
    
    s1 = create_blocking_keys(s1)
    s2 = create_blocking_keys(s2)
    s3 = create_blocking_keys(s3)
    
    s23 = pl.concat([s2, s3])
    
    keys = ["bk_1", "bk_2", "bk_3", "bk_4"]
    
    s1_df = s1.collect()
    s23_df = s23.collect()
    
    for key in keys:
        print(f"--- Blocking Key: {key} ---")
        block_sizes = s23_df.group_by(key).agg(pl.len().alias("block_size"))
        # filter out empty keys
        block_sizes = block_sizes.filter(
            pl.col(key).is_not_null() & 
            (pl.col(key) != "") & 
            (pl.col(key) != "||") & 
            (pl.col(key) != "|")
        )
        
        desc = block_sizes["block_size"].describe()
        print("Unique Blocks:", len(block_sizes))
        try:
            print("Mean size:", block_sizes["block_size"].mean())
            print("Median size:", block_sizes["block_size"].median())
            print("Max size:", block_sizes["block_size"].max())
            print("95th percentile:", block_sizes["block_size"].quantile(0.95))
            print("99th percentile:", block_sizes["block_size"].quantile(0.99))
        except Exception as e:
            print("Error computing quantiles:", e)
        print()

if __name__ == "__main__":
    analyze_blocking()
