import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD FEATURE-ENGINEERED DATA
# ============================================================

df = pd.read_csv("data/traffic_features.csv")

print("===== DATASET =====")
print("Shape:", df.shape)


# ============================================================
# 2. SORT BY TIME
# ============================================================

# traffic_features.csv no longer contains date_time,
# so we use the original preprocessed data to recover
# the chronological order.

original = pd.read_csv("data/preprocessed_traffic.csv")
original["date_time"] = pd.to_datetime(original["date_time"])

df["date_time"] = original["date_time"]

df = df.sort_values("date_time").reset_index(drop=True)

# Remove date_time after sorting
df = df.drop(columns=["date_time"])


# ============================================================
# 3. DEFINE TARGET
# ============================================================

target = "traffic_volume"

X = df.drop(columns=[target])
y = df[target]


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

# Use the first 80% for training
# and the final 20% for testing.
# This respects the time order.

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\n===== TRAIN / TEST SPLIT =====")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 5. IDENTIFY CATEGORICAL FEATURES
# ============================================================

categorical_features = [
    "weather_main",
    "weather_description"
]

numeric_features = [
    column for column in X.columns
    if column not in categorical_features
]


# ============================================================
# 6. ENCODE CATEGORICAL FEATURES
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# ============================================================
# 7. CREATE RANDOM FOREST MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    max_depth=20
)


# ============================================================
# 8. TRANSFORM DATA
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# ============================================================
# 9. TRAIN MODEL
# ============================================================

print("\n===== TRAINING MODEL =====")

model.fit(X_train_processed, y_train)

print("Model training completed!")


# ============================================================
# 10. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test_processed)


# ============================================================
# 11. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 4))


# ============================================================
# 12. SAVE MODEL
# ============================================================

model_package = {
    "model": model,
    "preprocessor": preprocessor,
    "features": X.columns.tolist()
}

joblib.dump(
    model_package,
    "models/traffic_model.pkl"
)

print("\n===== MODEL SAVED =====")
print("Saved to: models/traffic_model.pkl")