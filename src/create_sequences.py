import numpy as np


def create_sequences(df, feature_columns, sequence_length=30):
    """
    Convert engine sensor data into sequences suitable for RNN models.

    Each sequence has shape:
        (sequence_length, number_of_features)

    The target is the failure label at the final cycle
    of each sequence.
    """

    X = []
    y = []

    # Process each engine separately
    for engine_id in df["engine_id"].unique():

        engine_data = df[
            df["engine_id"] == engine_id
        ].sort_values("cycle")

        features = engine_data[
            feature_columns
        ].values

        targets = engine_data["failure"].values

        # Create sliding windows
        for i in range(
            len(engine_data) - sequence_length + 1
        ):

            sequence = features[
                i:i + sequence_length
            ]

            target = targets[
                i + sequence_length - 1
            ]

            X.append(sequence)
            y.append(target)

    return np.array(X), np.array(y)