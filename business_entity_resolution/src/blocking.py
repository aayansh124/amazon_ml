import polars as pl
from typing import List

def generate_candidates_for_key(s1_keys: pl.LazyFrame, s23_keys: pl.LazyFrame, key_name: str, max_block_size: int = 1000) -> pl.LazyFrame:
    # Count sizes in S23
    block_sizes = s23_keys.group_by(key_name).agg(pl.len().alias("block_size"))
    
    # Filter valid blocks
    valid_blocks = block_sizes.filter((pl.col("block_size") <= max_block_size) & (pl.col("block_size") > 0))
    
    # Keep only valid keys
    s1_filtered = s1_keys.join(valid_blocks, on=key_name, how="inner")
    s23_filtered = s23_keys.join(valid_blocks, on=key_name, how="inner")
    
    # Generate pairs
    pairs = s1_filtered.join(s23_filtered, on=key_name, how="inner")
    return pairs.select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ])

def generate_all_candidates(df_s1: pl.LazyFrame, df_s23: pl.LazyFrame, keys: List[str], max_block_size: int = 1000) -> pl.LazyFrame:
    all_pairs = []
    
    for bk in keys:
        s1_k = df_s1.select(["entity_id", bk]).filter(
            pl.col(bk).is_not_null() & 
            (pl.col(bk) != "") & 
            (pl.col(bk) != "||") & 
            (pl.col(bk) != "|")
        )
        s23_k = df_s23.select(["entity_id", bk]).filter(
            pl.col(bk).is_not_null() & 
            (pl.col(bk) != "") & 
            (pl.col(bk) != "||") & 
            (pl.col(bk) != "|")
        )
        
        # Rename S23 entity id to avoid collision on join
        s23_k = s23_k.rename({"entity_id": "entity_id_right"})
        
        pairs = generate_candidates_for_key(s1_k, s23_k, bk, max_block_size)
        all_pairs.append(pairs)
    
    # Union all
    if not all_pairs:
        return pl.DataFrame({"source1_entity_id": [], "candidate_entity_id": []}).lazy()
        
    final_candidates = pl.concat(all_pairs).unique()
    return final_candidates
