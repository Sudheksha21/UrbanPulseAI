import pandas as pd

# Load cleaned traffic data
df = pd.read_csv("data/cleaned_traffic.csv")

print("===== BEFORE PREPROCESSING =====")
print("Dataset shape:", df.shape)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATES =====")
print(df.duplicated().sum())

# Handle suspicious extreme rainfall values
extreme_rain = df["rain_1h"] > 100

print("\n===== EXTREME RAINFALL VALUES =====")
print(df.loc[extreme_rain, ["date_time", "rain_1h", "weather_main", "traffic_volume"]])

# Replace extreme rainfall values with 0
df.loc[extreme_rain, "rain_1h"] = 0.0

print("\nExtreme rainfall values corrected:", extreme_rain.sum())

# Save preprocessed dataset
df.to_csv("data/preprocessed_traffic.csv", index=False)

print("\n===== PREPROCESSING COMPLETED =====")
print("Final dataset shape:", df.shape)
print("Saved to: data/preprocessed_traffic.csv")