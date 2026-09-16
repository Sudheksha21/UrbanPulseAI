import pandas as pd

# Load preprocessed traffic data
df = pd.read_csv("data/preprocessed_traffic.csv")

print("===== ORIGINAL DATASET =====")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Convert date_time to datetime
df["date_time"] = pd.to_datetime(df["date_time"])

# Extract additional time features
df["day_of_month"] = df["date_time"].dt.day
df["week_of_year"] = df["date_time"].dt.isocalendar().week.astype(int)

# Create rush-hour feature
df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

# Create morning/evening period features
df["is_morning_peak"] = df["hour"].isin([7, 8, 9]).astype(int)
df["is_evening_peak"] = df["hour"].isin([16, 17, 18, 19]).astype(int)

# Remove date_time because we extracted useful information from it
df = df.drop(columns=["date_time"])

# Save feature-engineered dataset
df.to_csv("data/traffic_features.csv", index=False)

print("\n===== FEATURE ENGINEERING COMPLETED =====")
print("Final shape:", df.shape)
print("New columns:")
print(df.columns.tolist())
print("\nSaved to: data/traffic_features.csv")