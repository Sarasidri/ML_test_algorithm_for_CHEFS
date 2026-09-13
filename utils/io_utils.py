"""
Loading utilities: reads every trained artefact (base models, meta-model, encoders, and the
lookup tables used for GUI autocompletion) from disk into memory, ready for inference.
"""

import os

import joblib
import lightgbm as lgb
import pandas as pd


def load_feature_encoders(path):
    """
    INPUT: path (str): path to the pickled dictionary of fitted per-column LabelEncoder objects.
    OUTPUT: dict[str, sklearn.preprocessing.LabelEncoder]: the feature encoders, keyed by column
        name.
    """
    return joblib.load(path)


def load_target_encoder(path):
    """
    INPUT: path (str): path to the pickled LabelEncoder fitted on the target column.
    OUTPUT: sklearn.preprocessing.LabelEncoder: the target encoder.
    """
    return joblib.load(path)


def load_base_models(models_dir, model_names):
    """
    INPUT: models_dir (str): directory containing the individual base-model files, named
               'model{i}.pkl' for i = 1, 2, ...
           model_names (list[str]): ordered model identifiers as used during meta-model training
               (e.g. ['Model 1', 'Model 2', ...]). The returned dictionary is built strictly in
               this order, so that it stays aligned with the column order expected by the
               meta-model.
    OUTPUT: dict[str, object]: fitted base classifiers, keyed by model name.
    """
    base_models = {}
    for name in model_names:
        model_index = name.split(' ')[-1]
        model_path = os.path.join(models_dir, f"model{model_index}.pkl")
        base_models[name] = joblib.load(model_path)
    return base_models


def load_meta_model(path):
    """
    INPUT: path (str): path to the trained meta-model, either in LightGBM's native text format
               (.txt) or pickled with joblib (.pkl).
    OUTPUT: lightgbm.Booster: the trained meta-model.
    """
    if path.endswith('.txt'):
        return lgb.Booster(model_file=path)
    return joblib.load(path)


def load_meta_metadata(path):
    """
    INPUT: path (str): path to the pickled metadata dictionary saved alongside the meta-model.
    OUTPUT: dict: must contain 'all_classes', 'class_to_index', 'model_names', and
        'feature_columns', as produced at meta-model training time.
    """
    return joblib.load(path)


def load_autocomplete_values(columns_dir, columns):
    """
    INPUT: columns_dir (str): directory containing one '{column}.txt' file per column, listing
               its unique observed values (one per line).
           columns (list[str]): columns for which an autocomplete list should be loaded.
    OUTPUT: dict[str, list[str]]: mapping from column name to its list of known values.
    """
    autocomplete_values = {}
    for column in columns:
        file_path = os.path.join(columns_dir, f"{column}.txt")
        with open(file_path, 'r', encoding='utf-8') as f:
            autocomplete_values[column] = [line.strip() for line in f if line.strip()]
    return autocomplete_values


def load_test_samples(path):
    """
    INPUT: path (str): path to the Excel file holding a random sample of real (feature +
               target) rows drawn from the held-out test set, stored with their original string
               values (already decoded, not label-encoded).
    OUTPUT: pandas.DataFrame | None: the loaded rows, used by the GUI's 'Test real data' button
        to draw one row at a time; None if the file does not exist, so the GUI can start up
        without this optional feature.
    """
    if not os.path.exists(path):
        return None
    return pd.read_excel(path)


def load_completion_table(path, key_column='productcomp'):
    """
    INPUT: path (str): path to the completion lookup CSV, containing one row per productcomp
               with its majority-frequency (producttype, productspec) pair.
           key_column (str): column to use as the lookup key.
    OUTPUT: pandas.DataFrame: the completion table indexed by key_column, for constant-time row
        lookup.
    """
    completion_df = pd.read_csv(path)
    return completion_df.set_index(key_column)


def load_all_resources(models_dir, encoders_dir, columns_dir, autocomplete_columns,
                        test_data_path=None):
    """
    Convenience loader that gathers every artefact required to run the inference pipeline from a
    fresh Python process (called once at GUI start-up).

    INPUT: models_dir (str): directory containing 'model1.pkl' ... 'modelN.pkl', the trained
               meta-model, and 'meta_model_metadata.pkl'.
           encoders_dir (str): directory containing 'features_encoder.pkl' and
               'target_encoder.pkl'.
           columns_dir (str): directory containing the per-column autocomplete '.txt' files and
               'completion.csv'.
           autocomplete_columns (list[str]): feature columns for which an autocomplete list
               should be loaded (typically the user-editable columns).
           test_data_path (str | None): optional path to an Excel file of real sample rows used
               by the GUI's 'Test real data' button; the corresponding resource is None when not
               provided or when the file is missing.
    OUTPUT: dict: bundle with keys 'feature_encoders', 'target_encoder', 'base_models',
        'meta_model', 'meta_metadata', 'autocomplete_values', 'completion_table', and
        'test_samples'.
    """
    meta_metadata = load_meta_metadata(os.path.join(models_dir, 'meta_model_metadata.pkl'))

    meta_model_path = os.path.join(models_dir, 'meta_model.pkl')
    if not os.path.exists(meta_model_path):
        meta_model_path = os.path.join(models_dir, 'meta_model.txt')

    return {
        'feature_encoders': load_feature_encoders(
            os.path.join(encoders_dir, 'features_encoder.pkl')),
        'target_encoder': load_target_encoder(
            os.path.join(encoders_dir, 'target_encoder.pkl')),
        'base_models': load_base_models(models_dir, meta_metadata['model_names']),
        'meta_model': load_meta_model(meta_model_path),
        'meta_metadata': meta_metadata,
        'autocomplete_values': load_autocomplete_values(columns_dir, autocomplete_columns),
        'completion_table': load_completion_table(
            os.path.join(columns_dir, 'completion.csv')),
        'test_samples': load_test_samples(test_data_path) if test_data_path else None,
    }
