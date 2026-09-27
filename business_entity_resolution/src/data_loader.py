import polars as pl
import argparse
from typing import List

def load_data(data_dir: str, mode: str):
    if mode == "train":
        s1 = pl.read_csv(f"{data_dir}/train_source1.tsv", separator="\t")
        s2 = pl.read_csv(f"{data_dir}/train_source2.tsv", separator="\t")
        s3 = pl.read_csv(f"{data_dir}/train_source3.tsv", separator="\t")
        gt = pl.read_csv(f"{data_dir}/train_ground_truth.tsv", separator="\t")
        return s1, s2, s3, gt
    else:
        s1 = pl.read_csv(f"{data_dir}/test_source1.tsv", separator="\t")
        s2 = pl.read_csv(f"{data_dir}/test_source2.tsv", separator="\t")
        s3 = pl.read_csv(f"{data_dir}/test_source3.tsv", separator="\t")
        return s1, s2, s3, None

def normalize(df: pl.LazyFrame) -> pl.LazyFrame:
    df = df.with_columns([
        pl.col("business_name").str.to_lowercase()
            .str.replace_all(r"[^a-z0-9\s]", " ")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
            .alias("norm_name"),
        pl.col("business_address").str.to_lowercase()
            .str.replace_all(r"[^a-z0-9\s]", " ")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
            .alias("norm_address"),
        pl.col("country").str.to_lowercase().str.strip_chars().alias("norm_country")
    ])
    
    df = df.with_columns([
        pl.col("norm_name").str.split(" ").list.first().alias("name_token_1"),
        pl.col("norm_name").str.split(" ").list.get(1).alias("name_token_2"),
        pl.col("norm_address").str.split(" ").list.first().alias("addr_token_1"),
        pl.col("norm_address").str.split(" ").list.get(1).alias("addr_token_2"),
        pl.col("norm_name").str.slice(0, 4).alias("name_prefix"),
        pl.col("norm_address").str.slice(0, 4).alias("addr_prefix"),
    ])
    
    df = df.with_columns([
        pl.col(c).fill_null("") for c in [
            "name_token_1", "name_token_2", "addr_token_1", "addr_token_2",
            "name_prefix", "addr_prefix"
        ]
    ])
    return df

def create_blocking_keys(df: pl.LazyFrame) -> pl.LazyFrame:
    df = df.with_columns([
        pl.concat_str([pl.col("norm_country"), pl.lit("|"), pl.col("name_prefix")]).alias("bk_1"),
        pl.concat_str([pl.col("norm_country"), pl.lit("|"), pl.col("name_token_1"), pl.lit("|"), pl.col("addr_token_1")]).alias("bk_2"),
        pl.concat_str([pl.col("norm_country"), pl.lit("|"), pl.col("name_token_2"), pl.lit("|"), pl.col("name_token_1")]).alias("bk_3"),
        pl.concat_str([pl.col("name_prefix"), pl.lit("|"), pl.col("addr_prefix")]).alias("bk_4"),
    ])
    return df
