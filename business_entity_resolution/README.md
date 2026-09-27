# Business Entity Resolution Solution

This directory contains an end-to-end scalable ML pipeline for the Business Entity Resolution Challenge.

## Requirements

```bash
pip install -r requirements.txt
```

## Structure
- `src/` - Core logic
  - `data_loader.py`: Efficient Polars data loading & normalization
  - `blocking.py`: Multi-pass high-recall blocking candidate generator
  - `features.py`: Similarity computation (exact and fuzzy matching via RapidFuzz)
- `scripts/` - Execution scripts
  - `analyze_data.py`: Basic statistics of datasets
  - `analyze_blocking.py`: Measures block sizes & metrics
  - `train_pipeline.py`: Trains LightGBM with sample/full mode support
  - `predict.py`: Infers on test data and saves `output/matching_results.tsv`
  - `validate.py`: Validation wrapper
- `output/` - Contains the required output TSVs

## Running the Pipeline

1. **Analyze Data:**
   ```bash
   python scripts/analyze_data.py
   ```

2. **Analyze Blocking Limits:**
   ```bash
   python scripts/analyze_blocking.py
   ```

3. **Train Model (Sample):**
   Run a sample subset of data to test memory/logic constraints:
   ```bash
   python scripts/train_pipeline.py --mode sample
   ```
   
4. **Train Model (Full):**
   ```bash
   python scripts/train_pipeline.py --mode full
   ```

5. **Generate Output:**
   Produces `matching_results.tsv` and `candidate_pairs.tsv` in the `output/` folder.
   ```bash
   python scripts/predict.py
   ```

## Design Overview
1. **Normalization:** Lowercase string processing, handling missing values.
2. **Blocking:** 4 specific composite keys are built. Large generic blocks > 1000 items are discarded to budget candidates. 
3. **Features:** Generates Polars exact matches & RapidFuzz Jaro-Winkler distances.
4. **Classifier:** Uses LightGBM focusing on maximizing Macro F0.5. Protects singletons with threshold-based inference logic. 
