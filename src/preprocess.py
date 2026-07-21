"""
preprocess.py

CIC-IDS2017 dataset preprocessing script.
Merges all *WorkingHours* raw CSVs, cleans them, and produces a
processed dataset with both a binary Label and the original AttackType.
"""

import glob

import numpy as np
import pandas as pd

RAW_GLOB = "data/raw/*WorkingHours*.csv"
OUTPUT_PATH = "data/processed/cicids2017_processed.csv"


def main():
    # ==========================================
    # 1. Read all raw CSV files
    # ==========================================
    files = glob.glob(RAW_GLOB)
    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  - {f}")

    if not files:
        raise FileNotFoundError(
            f"No files matched pattern '{RAW_GLOB}'. "
            "Check your raw data folder and filenames."
        )

    dfs = []
    for file in files:
        print(f"Reading: {file}")
        tmp = pd.read_csv(file, low_memory=False)
        # Track which source file each row came from — useful later for
        # a day-based (temporal) train/test split instead of a random one.
        tmp["SourceFile"] = file
        dfs.append(tmp)

    # ==========================================
    # 2. Merge all files
    # ==========================================
    df = pd.concat(dfs, ignore_index=True)
    print(f"\nMerged shape: {df.shape}")

    # Remove spaces from column names (CIC-IDS2017 has inconsistent
    # leading spaces like " Label" in some files)
    df.columns = df.columns.str.strip()

    # ==========================================
    # 3. Missing values
    # ==========================================
    print("\nMissing values per column (before drop):")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    before = len(df)
    df = df.dropna()
    print(f"Dropped {before - len(df)} rows with missing values "
          f"({before} -> {len(df)})")

    # ==========================================
    # 4. Infinite values
    # ==========================================
    before = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    print(f"Dropped {before - len(df)} rows with infinite values "
          f"({before} -> {len(df)})")

    # ==========================================
    # 5. Duplicates
    # ==========================================
    n_dupes = df.duplicated().sum()
    print(f"\nExact duplicate rows found: {n_dupes}")

    if n_dupes > 0:
        print("Sample of duplicate groups (all occurrences shown):")
        print(df[df.duplicated(keep=False)].head(10))

    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows "
          f"({before} -> {len(df)})")

    # ==========================================
    # 6. Create AttackType and binary Label
    # ==========================================
    df["Label"] = df["Label"].str.strip()

    # Save original attack names before converting to binary
    df["AttackType"] = df["Label"]

    # Convert Label to binary: 0 = benign, 1 = any attack
    df["Label"] = df["AttackType"].apply(lambda x: 0 if x == "BENIGN" else 1)
    
    df = df.drop(columns=["AttackType", "SourceFile"]) 
    print("\nData types:")
    print(df.dtypes)

    # ==========================================
    # 7. Shuffle dataset
    # ==========================================
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ==========================================
    # 8. Distribution summaries
    # ==========================================
    print("\nBinary Label Distribution:")
    print(df["Label"].value_counts())

  

    # ==========================================
    # 9. Save processed dataset
    # ==========================================
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved successfully to {OUTPUT_PATH}")
    print("Final Shape:", df.shape)


if __name__ == "__main__":
    main()