import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Define column names
# -----------------------------

columns = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21"
]


# -----------------------------
# 2. Load training data
# -----------------------------

train_path = "data/raw/train_FD001.txt"

df = pd.read_csv(
    train_path,
    sep=r"\s+",
    header=None,
    names=columns
)


# -----------------------------
# 3. Calculate final cycle
# -----------------------------

final_cycles = df.groupby("engine_id")["cycle"].transform("max")


# -----------------------------
# 4. Calculate Remaining Useful Life
# -----------------------------

df["RUL"] = final_cycles - df["cycle"]


# -----------------------------
# 5. Create failure target
# -----------------------------

failure_horizon = 30

df["failure"] = (
    df["RUL"] <= failure_horizon
).astype(int)


# -----------------------------
# 6. Check results
# -----------------------------

print("First 10 rows:")
print(
    df[
        [
            "engine_id",
            "cycle",
            "RUL",
            "failure"
        ]
    ].head(10)
)

print("\nLast 10 rows of Engine 1:")
print(
    df[df["engine_id"] == 1][
        [
            "engine_id",
            "cycle",
            "RUL",
            "failure"
        ]
    ].tail(10)
)

print("\nFailure distribution:")
print(df["failure"].value_counts())

print("\nFailure percentage:")
print(
    df["failure"].value_counts(
        normalize=True
    ) * 100
)

# -----------------------------
# 7. Check feature variation
# -----------------------------

feature_columns = [
    "setting_1",
    "setting_2",
    "setting_3"
] + [
    f"sensor_{i}"
    for i in range(1, 22)
]

print("\nFeature Standard Deviations:")
print(
    df[feature_columns]
    .std()
    .sort_values()
)


# -----------------------------
# 8. Remove constant features
# -----------------------------

constant_features = [
    "sensor_1",
    "setting_3",
    "sensor_10",
    "sensor_19",
    "sensor_18",
    "sensor_16",
    "sensor_5"
]

df = df.drop(columns=constant_features)

print("\nDataset Shape After Removing Constant Features:")
print(df.shape)

print("\nRemaining Features:")
print(df.columns.tolist())

from sklearn.model_selection import train_test_split


# -----------------------------
# 9. Split data by engine
# -----------------------------

engine_ids = df["engine_id"].unique()

# First: 70% training, 30% temporary
train_engines, temp_engines = train_test_split(
    engine_ids,
    test_size=0.30,
    random_state=42
)

# Split remaining 30% into validation and test
val_engines, test_engines = train_test_split(
    temp_engines,
    test_size=0.50,
    random_state=42
)


# Create separate DataFrames
train_df = df[df["engine_id"].isin(train_engines)].copy()

val_df = df[df["engine_id"].isin(val_engines)].copy()

test_df = df[df["engine_id"].isin(test_engines)].copy()


# -----------------------------
# 10. Verify the split
# -----------------------------

print("\nEngine-level split:")

print(
    f"Training engines: "
    f"{train_df['engine_id'].nunique()}"
)

print(
    f"Validation engines: "
    f"{val_df['engine_id'].nunique()}"
)

print(
    f"Test engines: "
    f"{test_df['engine_id'].nunique()}"
)


print("\nRow-level distribution:")

print(f"Training rows: {len(train_df)}")
print(f"Validation rows: {len(val_df)}")
print(f"Test rows: {len(test_df)}")

# -----------------------------
# 11. Define input features
# -----------------------------

feature_columns = [
    column
    for column in df.columns
    if column not in ["engine_id", "cycle", "RUL", "failure"]
]

print("\nNumber of input features:")
print(len(feature_columns))

print("\nInput features:")
print(feature_columns)


# -----------------------------
# 12. Scale features
# -----------------------------

scaler = StandardScaler()

# Fit ONLY on training data
scaler.fit(train_df[feature_columns])

# Transform all datasets
train_df[feature_columns] = scaler.transform(
    train_df[feature_columns]
)

val_df[feature_columns] = scaler.transform(
    val_df[feature_columns]
)

test_df[feature_columns] = scaler.transform(
    test_df[feature_columns]
)


# -----------------------------
# 13. Verify scaling
# -----------------------------

print("\nTraining feature means:")
print(
    train_df[feature_columns]
    .mean()
    .round(3)
)

print("\nTraining feature standard deviations:")
print(
    train_df[feature_columns]
    .std()
    .round(3)
)


from create_sequences import create_sequences


# -----------------------------
# 14. Create RNN sequences
# -----------------------------

sequence_length = 30

X_train, y_train = create_sequences(
    train_df,
    feature_columns,
    sequence_length
)

X_val, y_val = create_sequences(
    val_df,
    feature_columns,
    sequence_length
)

X_test, y_test = create_sequences(
    test_df,
    feature_columns,
    sequence_length
)


print("\nSequence Shapes:")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nX_val:", X_val.shape)
print("y_val:", y_val.shape)

print("\nX_test:", X_test.shape)
print("y_test:", y_test.shape)

# -----------------------------
# 15. Check class distribution
# -----------------------------

print("\nClass Distribution After Sequence Creation:")

print("\nTraining:")
print(np.bincount(y_train))
print(np.bincount(y_train) / len(y_train) * 100)

print("\nValidation:")
print(np.bincount(y_val))
print(np.bincount(y_val) / len(y_val) * 100)

print("\nTest:")
print(np.bincount(y_test))
print(np.bincount(y_test) / len(y_test) * 100)

# -----------------------------
# 16. Save processed datasets
# -----------------------------

import os

os.makedirs("data/processed", exist_ok=True)

np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/y_train.npy", y_train)

np.save("data/processed/X_val.npy", X_val)
np.save("data/processed/y_val.npy", y_val)

np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/y_test.npy", y_test)

print("\nProcessed datasets saved successfully!")