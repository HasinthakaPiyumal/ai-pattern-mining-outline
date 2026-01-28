"""Inference pipeline: load saved models and score unverified data."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from ensemble_common import (
    CONFIG,
    prepare_labels,
    get_model_classes,
    align_probabilities,
    resolve_weights,
    load_and_merge_data,
    split_verified_sets,
    ensure_dir,
    load_json,
)


def weighted_vote_predict(prob_results, label_encoder, weights):
    weight_map = resolve_weights(weights)
    predictions = []
    scores = []

    total_rows = len(next(iter(prob_results.values()))) if prob_results else 0
    for row_index in range(total_rows):
        score_vector = None
        for name, probs_df in prob_results.items():
            weighted = weight_map.get(name, 0) * probs_df.iloc[row_index].astype(float).values
            score_vector = weighted if score_vector is None else score_vector + weighted
        best_index = int(np.argmax(score_vector))
        predictions.append(label_encoder.inverse_transform([best_index])[0])
        scores.append(float(score_vector[best_index]))

    return predictions, scores


def load_artifacts(model_dir: Path):
    models = {}
    for name in ("LogReg", "SVC", "KNN"):
        model_path = model_dir / f"{name.lower()}_model.joblib"
        if model_path.exists():
            models[name] = joblib.load(model_path)
    label_encoder = joblib.load(model_dir / "label_encoder.joblib")
    global_classes = np.array(load_json(model_dir / "classes.json"))
    feature_list = load_json(model_dir / "feature_list.json")
    return models, label_encoder, global_classes, feature_list


def run_prediction_pipeline(config=CONFIG):
    model_dir = Path(config["model_dir"])
    output_dir = ensure_dir(Path(config["output_dir"]))
    models, label_encoder, global_classes, feature_list = load_artifacts(model_dir)

    data = load_and_merge_data(config)
    _, unverified_df = split_verified_sets(
        data,
        target_column=config["target_column"],
        min_samples=config["min_samples_per_class"],
    )
    if unverified_df.empty:
        print("No unverified data to predict.")
        return

    X_predict = unverified_df[feature_list]
    prob_map = {}
    for name, model in models.items():
        probs = model.predict_proba(X_predict.values)
        model_classes = get_model_classes(model)
        aligned = align_probabilities(probs, model_classes, global_classes)
        prob_map[name] = pd.DataFrame(aligned, columns=global_classes)

    predictions, scores = weighted_vote_predict(prob_map, label_encoder, weights=None)

    pd.DataFrame(
        {
            "file": unverified_df["file"],
            "weighted_pred": predictions,
            "weighted_score": scores,
            "code_summary": unverified_df.get("code_summary"),
        }
    ).to_csv(output_dir / "unverified_weighted_predictions.csv", index=False)
    print(f"Saved predictions to {output_dir / 'unverified_weighted_predictions.csv'}")


if __name__ == "__main__":
    run_prediction_pipeline()
