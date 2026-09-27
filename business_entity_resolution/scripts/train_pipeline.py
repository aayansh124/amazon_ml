import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import polars as pl
import lightgbm as lgb
from sklearn.metrics import fbeta_score, precision_score, recall_score
import pickle
from src.data_loader import load_data, normalize, create_blocking_keys
from src.blocking import generate_all_candidates
from src.features import compute_similarity, compute_features_pandas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="sample", choices=["sample", "full"])
    args = parser.parse_args()
    
    print(f"Running pipeline in {args.mode} mode...")
    data_dir = ".." # Since we run from scripts/, the data is one level up
    
    # Check if files exist
    if not os.path.exists(os.path.join(data_dir, "train_source1.tsv")):
        # Maybe data is in current dir
        data_dir = "."
        
    s1_raw, s2_raw, s3_raw, gt_raw = load_data(data_dir, "train")
    
    if args.mode == "sample":
        s1_raw = s1_raw.head(10000)
        s2_raw = s2_raw.head(50000)
        s3_raw = s3_raw.head(50000)
        
    print("Normalizing...")
    s1 = normalize(s1_raw.lazy())
    s2 = normalize(s2_raw.lazy())
    s3 = normalize(s3_raw.lazy())
    
    print("Creating blocking keys...")
    s1 = create_blocking_keys(s1)
    s2 = create_blocking_keys(s2)
    s3 = create_blocking_keys(s3)
    
    s23 = pl.concat([s2, s3])
    
    # Blocking
    keys = ["bk_1", "bk_2", "bk_3", "bk_4"]
    print("Generating candidates...")
    candidates = generate_all_candidates(s1, s23, keys, max_block_size=1000)
    
    candidates_df = candidates.collect()
    print(f"Generated {len(candidates_df)} candidates.")
    
    # We need to create labels for train
    if gt_raw is not None:
        # Explode matched_entity_ids
        gt = gt_raw.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
        gt = gt.with_columns(pl.lit(1).alias("label"))
        
        # Join with candidates
        candidates_df = candidates_df.join(
            gt, 
            left_on=["source1_entity_id", "candidate_entity_id"],
            right_on=["source1_entity_id", "matched_entity_ids"],
            how="left"
        ).fill_null(0)
    else:
        candidates_df = candidates_df.with_columns(pl.lit(0).alias("label"))

    print("Computing features...")
    # Compute features in Polars
    features_lazy = compute_similarity(candidates_df.lazy(), s1, s23)
    features_df = features_lazy.collect()
    
    # Compute fuzzy features in Pandas
    print("Computing fuzzy features...")
    features_pd = features_df.to_pandas()
    features_pd = compute_features_pandas(features_pd)
    
    # Train LightGBM
    feature_cols = [
        "country_match", "name_t1_match", "addr_t1_match", 
        "name_len_diff", "addr_len_diff",
        "name_sim", "addr_sim"
    ]
    
    X = features_pd[feature_cols].fillna(0)
    y = features_pd["label"]
    
    print(f"Training LightGBM on {len(X)} pairs, {y.sum()} positive...")
    if len(X) == 0:
        print("No candidates found, exiting.")
        return
        
    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    
    preds = model.predict_proba(X)[:, 1]
    features_pd["score"] = preds
    
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Saved model to model.pkl")
    
    # Evaluate at threshold 0.5
    y_pred = (preds > 0.5).astype(int)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f05 = fbeta_score(y, y_pred, beta=0.5, zero_division=0)
    
    print(f"Metrics (Threshold 0.5) -> Precision: {precision:.4f} | Recall: {recall:.4f} | F0.5: {f05:.4f}")
    
    # Check max score for singletons (just a summary)
    print("Done pipeline.")

if __name__ == "__main__":
    main()
