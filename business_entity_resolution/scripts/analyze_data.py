import sys
import os
import polars as pl
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def main():
    data_dir = ".."
    if not os.path.exists(os.path.join(data_dir, "train_source1.tsv")):
        data_dir = "."
        
    print("Loading data...")
    s1 = pl.read_csv(f"{data_dir}/train_source1.tsv", separator="\t")
    s2 = pl.read_csv(f"{data_dir}/train_source2.tsv", separator="\t")
    
    print("Source 1 Shape:", s1.shape)
    print("Source 2 Shape:", s2.shape)
    print("S1 Country Dist:")
    print(s1["country"].value_counts())
    
if __name__ == "__main__":
    main()
