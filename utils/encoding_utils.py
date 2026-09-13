"""
Encoding/decoding utilities: converts a single raw user-entered row into the integer-coded
feature matrix expected by the base models, and converts predicted integer class codes back
into their original string labels.
"""

import pandas as pd


class UnseenLabelError(ValueError):
    """
    Raised when a value entered by the user was never observed while fitting the corresponding
    LabelEncoder, and can therefore not be encoded for the base models.
    """
    pass


def encode_feature_row(input_row, feature_encoders, feature_columns):
    """
    INPUT: input_row (dict[str, str]): raw feature values entered by the user, keyed by column
               name; must contain an entry for every name in feature_columns.
           feature_encoders (dict[str, sklearn.preprocessing.LabelEncoder]): fitted per-column
               encoders.
           feature_columns (list[str]): ordered feature column names expected by the base
               models; also fixes the column order of the returned DataFrame.
    OUTPUT: pandas.DataFrame: single-row DataFrame with feature_columns as columns, containing
        the integer code produced by each column's LabelEncoder.
    RAISES:
        UnseenLabelError: if a value was not present among the classes seen while fitting its
            encoder.
    """
    encoded_values = {}
    for column in feature_columns:
        raw_value = input_row[column]
        encoder = feature_encoders[column]
        try:
            encoded_values[column] = encoder.transform([raw_value])[0]
        except ValueError as exc:
            raise UnseenLabelError(
                f"Value '{raw_value}' for column '{column}' was not seen during training."
            ) from exc
    return pd.DataFrame([encoded_values], columns=feature_columns)


def decode_target_labels(encoded_values, target_encoder):
    """
    INPUT: encoded_values (list[int] | numpy.ndarray): integer class codes to decode.
           target_encoder (sklearn.preprocessing.LabelEncoder): fitted encoder for the target
               column.
    OUTPUT: list[str]: decoded class labels, in the same order as encoded_values.
    """
    return list(target_encoder.inverse_transform(list(encoded_values)))
