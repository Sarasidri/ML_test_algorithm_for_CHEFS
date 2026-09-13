"""
Core inference pipeline: base-model predictions, meta-model feature construction, and top-k
ranking of candidate target classes for a single input sample.

These functions mirror, at inference time, the exact computations used when training the
meta-model, so that a sample processed through this module receives predictions consistent
with the reported evaluation metrics.
"""

import numpy as np

from utils.encoding_utils import decode_target_labels, encode_feature_row


def predict_all(models, X):
    """
    Runs every base model on the same input matrix.

    INPUT: models (dict[str, object]): fitted base classifiers, keyed by model name; each must
               expose .predict(X) and .predict_proba(X), as well as a .classes_ attribute.
           X (pandas.DataFrame): encoded feature matrix, one row per sample.
    OUTPUT: tuple:
        - numpy.ndarray of shape (n_samples, n_models): the class predicted by each base model.
        - list[tuple[numpy.ndarray, numpy.ndarray]]: for each model, a pair (predicted
          probabilities of shape (n_samples, n_classes_model), the model's classes_).
    """
    preds, probas = [], []
    for model in models.values():
        preds.append(model.predict(X))
        probas.append((model.predict_proba(X), model.classes_))
    return np.column_stack(preds), probas


def build_meta_features(probas_list, all_classes, class_to_index):
    """
    Builds the meta-model's input features from the base models' predicted probabilities.

    The feature set summarises, for each base model, how confident and how "peaked" its
    probability distribution is (maximum probability, gap between the top-2 classes, mean
    probability, predicted-class rank, entropy, mass concentrated in the top-3 classes, and the
    ratio between the top-1 and top-2 probabilities), plus a single ensemble-level entropy
    computed from the probabilities of all base models pooled together.

    INPUT: probas_list (list[tuple[numpy.ndarray, numpy.ndarray]]): per-model (probabilities,
               classes_) pairs, as returned by predict_all.
           all_classes (list): sorted list of every target class observed across all base
               models at training time.
           class_to_index (dict): mapping from class label to its column index in the pooled
               ensemble probability matrix.
    OUTPUT: numpy.ndarray of shape (n_samples, n_features_meta): the feature matrix consumed by
        the meta-model.
    """
    n = probas_list[0][0].shape[0]
    n_models = len(probas_list)

    max_prob = np.column_stack([p.max(axis=1) for p, _ in probas_list])
    diff_top1_top2 = np.zeros((n, n_models))
    mean_prob = np.zeros((n, n_models))
    rank_prob = np.zeros((n, n_models))
    entropy = np.zeros((n, n_models))
    top3_sum = np.zeros((n, n_models))
    ratio_top2 = np.zeros((n, n_models))
    ensemble_matrix = np.zeros((n, len(all_classes)))

    for i, (probas, cls) in enumerate(probas_list):
        idx = [class_to_index[c] for c in cls]
        ensemble_matrix[:, idx] += probas

        top2 = np.partition(probas, -2, axis=1)[:, -2:]
        diff_top1_top2[:, i] = top2[:, 1] - top2[:, 0]
        mean_prob[:, i] = probas.mean(axis=1)
        rank_prob[:, i] = (-probas).argsort(axis=1)[:, 0]
        entropy[:, i] = -np.sum(probas * np.log(probas + 1e-9), axis=1)

        sorted_probs = np.sort(probas, axis=1)
        top3_sum[:, i] = sorted_probs[:, -3:].sum(axis=1)
        ratio_top2[:, i] = sorted_probs[:, -1] / (sorted_probs[:, -2] + 1e-9)

    ens_norm = ensemble_matrix / ensemble_matrix.sum(axis=1, keepdims=True)
    ensemble_entropy = -np.sum(ens_norm * np.log(ens_norm + 1e-9), axis=1).reshape(-1, 1)

    return np.hstack([max_prob, diff_top1_top2, mean_prob, rank_prob,
                       entropy, top3_sum, ratio_top2, ensemble_entropy])


def get_topk_unique_predictions(base_preds_row, meta_probs_row, k):
    """
    Ranks base models by the meta-model's confidence and walks down that ranking to collect the
    top-k *distinct* predicted classes for a single sample.

    INPUT: base_preds_row (numpy.ndarray of shape (n_models,)): each base model's predicted
               (encoded) class for one sample, in the same order as the meta-model's classes.
           meta_probs_row (numpy.ndarray of shape (n_models,)): meta-model probability assigned
               to each base model for that same sample, i.e. how much the meta-model trusts each
               base model's prediction on this specific sample.
           k (int): maximum number of distinct classes to return.
    OUTPUT: list: up to k distinct encoded class labels, ordered from most to least confident.
        Fewer than k values are returned if the base models agree enough that scanning down the
        full model ranking produces fewer than k distinct classes.
    """
    ranked_model_idx = np.argsort(meta_probs_row)[::-1]
    ranked_predictions = base_preds_row[ranked_model_idx]

    unique_predictions = []
    for prediction in ranked_predictions:
        if prediction not in unique_predictions:
            unique_predictions.append(prediction)
        if len(unique_predictions) == k:
            break
    return unique_predictions


def predict_top_k(input_row, feature_encoders, base_models, meta_model, meta_metadata,
                   target_encoder, k=3):
    """
    Full single-sample inference pipeline: encodes the raw input, runs all base models, scores
    them with the meta-model, and decodes the top-k most likely target classes.

    INPUT: input_row (dict[str, str]): raw feature values entered by the user, keyed by column
               name (must cover every column in meta_metadata['feature_columns']).
           feature_encoders (dict): fitted per-column LabelEncoder objects.
           base_models (dict): fitted base classifiers, keyed by model name.
           meta_model (lightgbm.Booster): trained meta-model.
           meta_metadata (dict): must contain 'feature_columns', 'model_names', 'all_classes',
               and 'class_to_index', as saved at meta-model training time.
           target_encoder (sklearn.preprocessing.LabelEncoder): fitted encoder for the target
               column.
           k (int): number of distinct predicted classes to return.
    OUTPUT: list[str]: up to k decoded target class labels, ranked from most to least confident.
    RAISES:
        UnseenLabelError: propagated from encode_feature_row if a provided value is not among
            those seen while fitting its encoder.
    """
    feature_columns = meta_metadata['feature_columns']
    all_classes = meta_metadata['all_classes']
    class_to_index = meta_metadata['class_to_index']

    X = encode_feature_row(input_row, feature_encoders, feature_columns)

    base_preds, base_probas = predict_all(base_models, X)
    meta_X = build_meta_features(base_probas, all_classes, class_to_index)
    meta_probs = meta_model.predict(meta_X)

    top_k_encoded = get_topk_unique_predictions(base_preds[0], meta_probs[0], k=k)
    return decode_target_labels(top_k_encoded, target_encoder)
