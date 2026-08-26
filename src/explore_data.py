import pandas as pd

train_path = "data/raw/train_FD001.txt"

# Column names
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

train_df = pd.read_csv(
    train_path,
    sep=r"\s+",
    header=None,
    names=columns
)

print("Dataset Shape:")
print(train_df.shape)

print("\nFirst 5 Rows:")
print(train_df.head())

print("\nNumber of Engines:")
print(train_df["engine_id"].nunique())

print("\nCycles per Engine:")
print(train_df.groupby("engine_id")["cycle"].max().describe())