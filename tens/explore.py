
import pandas as pd

url = (
    "https://huggingface.co/datasets/"
    "Tobi-Bueck/customer-support-tickets/resolve/main/"
    "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
)

df = pd.read_csv(url)

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

print("\nTypes:")
print(df["type"].value_counts())

print("\nPriorities:")
print(df["priority"].value_counts())

print("\nQueues:")
print(df["queue"].value_counts())

print("\nLanguages:")
print(df["language"].value_counts())

print("\nMissing values:")
print(df.isna().sum())

print("\nExample ticket:")
print(df.iloc[0].to_string())
