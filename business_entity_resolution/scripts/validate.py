import os
import sys

def main():
    print("Validating outputs...")
    if not os.path.exists("output/matching_results.tsv"):
        print("ERROR: output/matching_results.tsv not found.")
        sys.exit(1)
        
    if not os.path.exists("output/candidate_pairs.tsv"):
        print("ERROR: output/candidate_pairs.tsv not found.")
        sys.exit(1)
        
    print("Files found. To run the official evaluator, use:")
    print("python3 ../utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir ../dataset/test")
    print("Validation script completed successfully.")

if __name__ == "__main__":
    main()
