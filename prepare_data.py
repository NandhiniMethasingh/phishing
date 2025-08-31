import pandas as pd

# ---------- CONFIG ----------
CSV_FILE = "dataset_full.csv"      # already extracted
OUTPUT_CSV = "dataset_cleaned.csv"
# ----------------------------

# 1) Load dataset
df = pd.read_csv(CSV_FILE, low_memory=False)
print("Loaded:", CSV_FILE)
print("Initial shape:", df.shape)

# 2) Ensure target encoding (1=phishing, 0=legitimate)
if df["phishing"].dtype == "O":  # object/string
    df["phishing"] = (
        df["phishing"]
        .str.strip()
        .str.lower()
        .map({"phishing": 1, "legitimate": 0})
    )

if df["phishing"].dtype != "int64" and df["phishing"].dtype != "int32":
    df["phishing"] = pd.to_numeric(df["phishing"], errors="coerce")

if set(df["phishing"].dropna().unique()) - {0, 1}:
    raise ValueError("Target column 'phishing' is not cleanly encoded as 0/1.")

# 3) Handle missing values (fill numeric NaN with median)
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# 4) Drop duplicate rows
before = df.shape[0]
df = df.drop_duplicates()
dupes_dropped = before - df.shape[0]
print(f"Duplicates dropped: {dupes_dropped}")

# 5) Drop constant columns
constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
df = df.drop(columns=constant_cols)
print(f"Constant columns dropped ({len(constant_cols)}): {constant_cols}")

# 6) Save cleaned dataset
print("Final shape:", df.shape)
print("Target distribution:\n", df["phishing"].value_counts(dropna=False))
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved cleaned data to: {OUTPUT_CSV}")
