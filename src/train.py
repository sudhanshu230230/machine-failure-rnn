import os
import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from model import build_lstm_model


# -----------------------------
# 1. Load processed datasets
# -----------------------------

X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy")

X_val = np.load("data/processed/X_val.npy")
y_val = np.load("data/processed/y_val.npy")


print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)


# -----------------------------
# 2. Build the LSTM model
# -----------------------------

input_shape = (
    X_train.shape[1],
    X_train.shape[2]
)

model = build_lstm_model(input_shape)


# -----------------------------
# 3. Compile the model
# -----------------------------

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# -----------------------------
# 4. Create required folders
# -----------------------------

os.makedirs("models", exist_ok=True)


# -----------------------------
# 5. Callbacks
# -----------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    filepath="models/best_lstm_model.keras",
    monitor="val_loss",
    save_best_only=True
)


# -----------------------------
# 6. Train the model
# -----------------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    callbacks=[
        early_stopping,
        model_checkpoint
    ]
)


# -----------------------------
# 7. Save final model
# -----------------------------

model.save("models/final_lstm_model.keras")

print("\nTraining completed successfully!")