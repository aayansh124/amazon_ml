import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import polars as pl
import lightgbm as lgb
import pickle
from src.data_loader import load_data, normalize, create_blocking_keys
from src.blocking import generate_all_candidates
from src.features import compute_similarity, compute_features_pandas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="model.pkl")
    parser.add_argument("--output-matching", type=str, default="output/matching_results.tsv")
    parser.add_argument("--output-candidate", type=str, default="output/candidate_pairs.tsv")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    
    print("Loading test data...")
    data_dir = ".."
    if not os.path.exists(os.path.join(data_dir, "test_source1.tsv")):
        data_dir = "."
        
    s1_raw, s2_raw, s3_raw, _ = load_data(data_dir, "test")
    
    print("Normalizing...")
    s1 = normalize(s1_raw.lazy())
    s2 = normalize(s2_raw.lazy())
    s3 = normalize(s3_raw.lazy())
    
    print("Creating blocking keys...")
    s1 = create_blocking_keys(s1)
    s2 = create_blocking_keys(s2)
    s3 = create_blocking_keys(s3)
    
    s23 = pl.concat([s2, s3])
    
    keys = ["bk_1", "bk_2", "bk_3", "bk_4"]
    print("Generating candidates...")
    candidates = generate_all_candidates(s1, s23, keys, max_block_size=1000)
    
    candidates_df = candidates.collect()
    print(f"Generated {len(candidates_df)} candidates.")
    
    # Save candidate_pairs.tsv
    # S1 might have zero candidates, but candidate_pairs.tsv usually expects pairs.
    # The rules say: "every predicted match must exist in candidate_pairs.tsv"
    # Also "exactly one row per Source 1 test entity". Wait, the prompt says:
    # Columns: source1_entity_id, candidate_entity_ids
    # So we need to group by source1_entity_id.
    
    s1_ids = s1_raw.select("entity_id").rename({"entity_id": "source1_entity_id"})
    
    cand_grouped = candidates_df.group_by("source1_entity_id").agg(
        pl.col("candidate_entity_id").alias("candidate_entity_ids")
    )
    cand_grouped = cand_grouped.with_columns(
        pl.col("candidate_entity_ids").list.join(",")
    )
    
    # Join with all S1 to ensure every S1 is present
    cand_final = s1_ids.join(cand_grouped, on="source1_entity_id", how="left").fill_null("")
    
    os.makedirs(os.path.dirname(args.output_candidate), exist_ok=True)
    cand_final.write_csv(args.output_candidate, separator="\t")
    print(f"Saved candidate pairs to {args.output_candidate}")
    
    if len(candidates_df) == 0:
        print("No candidates, saving empty matching.")
        cand_final = cand_final.rename({"candidate_entity_ids": "matched_entity_ids"})
        cand_final.write_csv(args.output_matching, separator="\t")
        return

    print("Computing features...")
    features_lazy = compute_similarity(candidates_df.lazy(), s1, s23)
    features_df = features_lazy.collect()
    
    print("Computing fuzzy features...")
    features_pd = features_df.to_pandas()
    features_pd = compute_features_pandas(features_pd)
    
    feature_cols = [
        "country_match", "name_t1_match", "addr_t1_match", 
        "name_len_diff", "addr_len_diff",
        "name_sim", "addr_sim"
    ]
    
    X = features_pd[feature_cols].fillna(0)
    
    print("Loading model...")
    with open(args.model_path, "rb") as f:
        model = pickle.load(f)
        
    preds = model.predict_proba(X)[:, 1]
    features_pd["score"] = preds
    
    # Filter by threshold
    matches = features_pd[features_pd["score"] >= args.threshold]
    matches_df = pl.from_pandas(matches)
    
    if len(matches_df) > 0:
        matches_grouped = matches_df.group_by("source1_entity_id").agg(
            pl.col("candidate_entity_id").alias("matched_entity_ids")
        )
        matches_grouped = matches_grouped.with_columns(
            pl.col("matched_entity_ids").list.join(",")
        )
    else:
        matches_grouped = pl.DataFrame({"source1_entity_id": [], "matched_entity_ids": []})
        
    match_final = s1_ids.join(matches_grouped, on="source1_entity_id", how="left").fill_null("")
    match_final.write_csv(args.output_matching, separator="\t")
    
    print(f"Saved matching results to {args.output_matching}")

if __name__ == "__main__":
    main()
