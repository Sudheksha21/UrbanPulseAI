import pandas as pd

file_path = "data/Metro_Interstate_Traffic_Volume.csv.gz"

df = pd.read_csv(file_path)

print("===== FIRST 5 RECORDS =====")
print(df.head())

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== DATASET SIZE =====")
print(df.shape)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())