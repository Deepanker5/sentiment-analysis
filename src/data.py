# src/data.py
import pandas as pd
from sklearn.model_selection import train_test_split

# Consistent order across the whole project
LABEL_ORDER = ["negative", "neutral", "positive"]

def load_dataset(path="data/all-data.csv"):
    # This dataset has a funky header; latin-1 is robust here
    df = pd.read_csv(path, encoding="latin-1", engine="python", on_bad_lines="skip")
    # First col is label, second col is text (long header name in file)
    df = df.rename(columns={df.columns[0]: "label", df.columns[1]: "text"})
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()
    # Keep only known labels
    df = df[df["label"].isin(LABEL_ORDER)].reset_index(drop=True)
    return df

def split_stratified(df, seed=42, train_size=0.7, val_size=0.15):
    train_df, temp_df = train_test_split(
        df, test_size=1-train_size, random_state=seed, stratify=df["label"]
    )
    val_rel = val_size / (1 - train_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=1-val_rel, random_state=seed, stratify=temp_df["label"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def label_to_id(label: str) -> int:
    return LABEL_ORDER.index(label)

def id_to_label(idx: int) -> str:
    return LABEL_ORDER[idx]
