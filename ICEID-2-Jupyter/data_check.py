#!/usr/bin/env python3
# inspect_data.py
# -----------------------------------------------------------------------------
# DESCRIPTION:
#   This script loads all raw and processed data artifacts to provide a
#   detailed profile of their content and structure. It does not perform any
#   training or linking. The goal is to understand the data's types, null
#   counts, and value distributions to inform the creation of a better
#   training script.
#
# INPUT:
#   - `raw_data/people.csv`
#   - `raw_data/manntol_einstaklingar_new.csv`
#   - `artifacts/iceid_ml_ready.npz`
#   - `artifacts/row_labels.csv`
#
# OUTPUT:
#   - A detailed printout to the console summarizing the data.
# -----------------------------------------------------------------------------
import pandas as pd
from pathlib import Path
from scipy import sparse
import numpy as np

# --- Configuration ---
DATA_DIR = Path("raw_data")
ART_DIR = Path("artifacts")

def profile_dataframe(df, name="DataFrame"):
    """Prints a detailed profile of a pandas DataFrame."""
    print("\n" + "="*50)
    print(f"Profiling: {name}")
    print("="*50)
    
    print(f"\n--- Shape ---\n{df.shape}")
    
    print("\n--- First 5 Rows ---")
    print(df.head().to_string())
    
    print("\n--- Data Types and Null Counts ---")
    info_df = pd.DataFrame({
        'Dtype': df.dtypes,
        'Nulls': df.isnull().sum(),
        'Null %': (df.isnull().sum() / len(df) * 100).round(2)
    })
    print(info_df.to_string())
    
    print("\n--- Column Value Analysis ---")
    # Columns to analyze in detail
    analysis_cols = [
        'sex', 'status', 'marriagestatus', 'hjuskapur', # Categorical
        'first_name', 'patronym', 'surname', # Name fields for blocking
        'birthyear', 'heimild', # Numeric for blocking/features
        'bi_sokn', 'bi_hreppur', 'bi_sysla' # Geographic for blocking
    ]
    
    for col in analysis_cols:
        if col in df.columns:
            print(f"\n- Analysis for column: '{col}'")
            # Check for empty strings in object columns, as they aren't caught by isnull()
            if df[col].dtype == 'object':
                empty_strings = (df[col] == '').sum()
                if empty_strings > 0:
                    print(f"  Found {empty_strings} empty strings ('').")

            # Show value counts for categorical data
            if df[col].nunique() < 20: # Only for low-cardinality columns
                print("  Value Counts:")
                print(df[col].value_counts(dropna=False).to_string())
            else:
                print(f"  Unique values: {df[col].nunique()}")

def main():
    print("Starting Data Inspection Script...")

    # --- 1. Profile raw_data/people.csv ---
    people_path = DATA_DIR / "people.csv"
    if people_path.exists():
        people_df = pd.read_csv(people_path, low_memory=False, dtype={
            "first_name": str, "middle_name": str, "patronym": str, "surname": str
        })
        profile_dataframe(people_df, name="raw_data/people.csv")
    else:
        print(f"\nERROR: Could not find {people_path}")

    # --- 2. Profile raw_data/manntol_einstaklingar_new.csv ---
    mann_path = DATA_DIR / "manntol_einstaklingar_new.csv"
    if mann_path.exists():
        mann_df = pd.read_csv(mann_path, low_memory=False, dtype=str)
        profile_dataframe(mann_df, name="raw_data/manntol_einstaklingar_new.csv")
    else:
        print(f"\nERROR: Could not find {mann_path}")
        
    # --- 3. Profile artifacts/row_labels.csv ---
    labels_path = ART_DIR / "row_labels.csv"
    if labels_path.exists():
        labels_df = pd.read_csv(labels_path)
        profile_dataframe(labels_df, name="artifacts/row_labels.csv")
    else:
        print(f"\nERROR: Could not find {labels_path}")

    # --- 4. Profile artifacts/iceid_ml_ready.npz ---
    features_path = ART_DIR / "iceid_ml_ready.npz"
    print("\n" + "="*50)
    print("Profiling: artifacts/iceid_ml_ready.npz")
    print("="*50)
    if features_path.exists():
        X = sparse.load_npz(features_path)
        print(f"\n--- Sparse Feature Matrix Info ---")
        print(f"Shape: {X.shape}")
        print(f"Format: {X.format}")
        print(f"Data Type: {X.dtype}")
        print(f"Stored elements: {X.nnz}")
    else:
        print(f"\nERROR: Could not find {features_path}")

    print("\n--- Data Inspection Finished ---")

if __name__ == "__main__":
    main()