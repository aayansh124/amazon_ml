import polars as pl
from rapidfuzz.distance import JaroWinkler

def compute_features_pandas(df_pd):
    # Basic matches
    df_pd["country_match"] = (df_pd["norm_country"] == df_pd["norm_country_cand"]).astype(int)
    df_pd["name_t1_match"] = (df_pd["name_token_1"] == df_pd["name_token_1_cand"]).astype(int)
    df_pd["addr_t1_match"] = (df_pd["addr_token_1"] == df_pd["addr_token_1_cand"]).astype(int)
    
    # Fuzzy using JaroWinkler
    names1 = df_pd["norm_name"].fillna("").tolist()
    names2 = df_pd["norm_name_cand"].fillna("").tolist()
    addrs1 = df_pd["norm_address"].fillna("").tolist()
    addrs2 = df_pd["norm_address_cand"].fillna("").tolist()
    
    name_sims = [JaroWinkler.normalized_similarity(n1, n2) for n1, n2 in zip(names1, names2)]
    addr_sims = [JaroWinkler.normalized_similarity(a1, a2) for a1, a2 in zip(addrs1, addrs2)]
    
    df_pd["name_sim"] = name_sims
    df_pd["addr_sim"] = addr_sims
    return df_pd

def compute_similarity(df_pairs: pl.LazyFrame, df_s1: pl.LazyFrame, df_s23: pl.LazyFrame) -> pl.LazyFrame:
    df_feat = df_pairs.join(df_s1, left_on="source1_entity_id", right_on="entity_id", how="left")
    df_feat = df_feat.join(df_s23, left_on="candidate_entity_id", right_on="entity_id", how="left", suffix="_cand")
    
    # We will compute basic exact features in Polars for speed
    df_feat = df_feat.with_columns([
        (pl.col("norm_country") == pl.col("norm_country_cand")).cast(pl.Int32).alias("country_match"),
        (pl.col("name_token_1") == pl.col("name_token_1_cand")).cast(pl.Int32).alias("name_t1_match"),
        (pl.col("addr_token_1") == pl.col("addr_token_1_cand")).cast(pl.Int32).alias("addr_t1_match"),
        
        # We can also add length difference
        (pl.col("norm_name").str.len_chars() - pl.col("norm_name_cand").str.len_chars()).abs().alias("name_len_diff"),
        (pl.col("norm_address").str.len_chars() - pl.col("norm_address_cand").str.len_chars()).abs().alias("addr_len_diff"),
    ])
    
    return df_feat
