from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_lstm_model(input_shape):
    
    model = Sequential([
        
        LSTM(
            units=64,
            input_shape=input_shape
        ),
        
        Dropout(0.2),
        
        Dense(
            units=32,
            activation="relu"
        ),
        
        Dropout(0.2),
        
        Dense(
            units=1,
            activation="sigmoid"
        )
    ])
    
    return model