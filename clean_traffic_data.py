import pandas as pd

# 1. Load the original dataset
file_path = "data/Metro_Interstate_Traffic_Volume.csv"

data = pd.read_csv(file_path)

print("Original dataset shape:", data.shape)

# 2. Check duplicate records
duplicates = data.duplicated().sum()
print("Duplicate records:", duplicates)

# 3. Remove duplicate records
data = data.drop_duplicates()

# 4. Convert date_time into datetime format
data["date_time"] = pd.to_datetime(data["date_time"])

# 5. Create useful time-based features
data["hour"] = data["date_time"].dt.hour
data["day_of_week"] = data["date_time"].dt.dayofweek
data["month"] = data["date_time"].dt.month
data["year"] = data["date_time"].dt.year

# 6. Create weekend indicator
data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)

# 7. Remove the holiday column
# It contains too many missing values in this dataset.
data = data.drop(columns=["holiday"])

# 8. Save the cleaned dataset
output_file = "data/cleaned_traffic.csv"

data.to_csv(output_file, index=False)

print("\nCleaning completed successfully!")
print("Cleaned dataset shape:", data.shape)
print("Saved as:", output_file)

print("\nNew columns:")
print(data.columns.tolist())                                            