"""Shared configuration and utilities for pattern identifier ensembles."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

CONFIG = {
    "target_column": "verified_pattern",
    "predict_mode": False,
    "labeled_data_path": "/datasets/labeled_data.csv",
    "embeddings_path": "/datasets/embeddings.csv",
    "min_samples_per_class": 20,
    "n_splits": 5,
    "random_state": 42,
    "none_label": "none",
    "model_dir": "artifacts/models",
    "output_dir": "artifacts/outputs",
}

MODEL_DEFAULTS = {
    "LogReg": {"C": 10, "class_weight": "balanced", "solver": "lbfgs"},
    "SVC": {
        "C": 10,
        "kernel": "rbf",
        "class_weight": "balanced",
        "probability": True,
        "gamma": "scale",
    },
    "KNN": {"n_neighbors": 15, "weights": "distance", "p": 2},
}

MODEL_BUILDERS = {
    "LogReg": lambda params, seed: LogisticRegression(**params, random_state=seed, max_iter=2000),
    "SVC": lambda params, seed: make_pipeline(StandardScaler(), SVC(**params, random_state=seed)),
    "KNN": lambda params, _seed: make_pipeline(StandardScaler(), KNeighborsClassifier(**params)),
}

# Calibrate weights via averaged model F1 scores or researcher priors if one model should dominate
VOTE_WEIGHTS = {"LogReg": 1.06, "SVC": 1.01, "KNN": 0.93}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_labels(y):
    label_encoder = LabelEncoder()
    y_values = y.values if isinstance(y, pd.Series) else y
    encoded = label_encoder.fit_transform(y_values)
    return label_encoder, encoded


def get_model_classes(fitted_model):
    if hasattr(fitted_model, "classes_"):
        return fitted_model.classes_
    if hasattr(fitted_model, "steps") and fitted_model.steps:
        return fitted_model.steps[-1][1].classes_
    raise AttributeError("Model does not expose classes_.")


def build_model(name, params, seed):
    if name not in MODEL_BUILDERS:
        raise KeyError(f"Unknown model name: {name}")
    return MODEL_BUILDERS[name](params, seed)


def align_probabilities(prob_matrix, model_classes, global_classes):
    aligned = np.zeros((prob_matrix.shape[0], len(global_classes)))
    for class_id, class_label in enumerate(model_classes):
        global_index = np.where(global_classes == class_label)[0][0]
        aligned[:, global_index] = prob_matrix[:, class_id]
    return aligned


def resolve_model_params(custom_params=None):
    if not custom_params:
        return MODEL_DEFAULTS
    merged = {}
    for name, defaults in MODEL_DEFAULTS.items():
        override = custom_params.get(name, {}) if custom_params else {}
        merged[name] = {**defaults, **override}
    return merged


def resolve_weights(custom_weights=None):
    if custom_weights is None:
        return VOTE_WEIGHTS
    return {**VOTE_WEIGHTS, **custom_weights}


def load_and_merge_data(config):
    labeled = pd.read_csv(config["labeled_data_path"])
    embeddings = pd.read_csv(config["embeddings_path"])
    merged = pd.merge(labeled, embeddings, on="file")
    return merged


def split_verified_sets(data, target_column, min_samples):
    verified = data[~data[target_column].isna()]
    counts = verified[target_column].value_counts()
    keep_labels = counts[counts >= min_samples].index
    verified_filtered = verified[verified[target_column].isin(keep_labels)]
    unverified = data[data[target_column].isna()]
    return verified_filtered, unverified


def select_embedding_columns(df):
    return [col for col in df.columns if col.startswith("dim_")]


def dump_json(obj, path: Path):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
