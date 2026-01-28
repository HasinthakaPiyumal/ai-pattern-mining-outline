"""Training pipeline: cross-validation reporting and model persistence."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

from ensemble_common import (
    CONFIG,
    MODEL_BUILDERS,
    prepare_labels,
    get_model_classes,
    build_model,
    align_probabilities,
    resolve_model_params,
    resolve_weights,
    load_and_merge_data,
    split_verified_sets,
    select_embedding_columns,
    ensure_dir,
    dump_json,
)


def majority_vote(prob_results, y_true, label_encoder, none_label):
    y_true_values = y_true.values if isinstance(y_true, pd.Series) else y_true
    bucket_names = ("all_three_agree", "two_agree", "all_disagree")
    bucket_totals = {bucket: 0 for bucket in bucket_names}
    bucket_correct = {bucket: 0 for bucket in bucket_names}

    predictions = []
    for row_index in range(len(y_true_values)):
        votes = []
        for probs_df in prob_results.values():
            encoded_class = int(probs_df.iloc[row_index].astype(float).idxmax())
            votes.append(label_encoder.inverse_transform([encoded_class])[0])

        counts = pd.Series(votes).value_counts()
        top_label = counts.idxmax()
        top_count = counts.iloc[0]

        if top_count == 3:
            bucket = "all_three_agree"
            final_label = top_label
        elif top_count == 2:
            bucket = "two_agree"
            final_label = top_label
        else:
            bucket = "all_disagree"
            final_label = none_label

        bucket_totals[bucket] += 1
        bucket_correct[bucket] += int(final_label == y_true_values[row_index])
        predictions.append(final_label)

    labels = sorted(set(y_true_values) | {none_label})
    report = classification_report(y_true_values, predictions, labels=labels)

    summary_rows = []
    for bucket in bucket_names:
        total = bucket_totals[bucket]
        correct = bucket_correct[bucket]
        accuracy = correct / total if total else float("nan")
        summary_rows.append({"bucket": bucket, "total": total, "correct": correct, "accuracy": accuracy})
    summary_df = pd.DataFrame(summary_rows)

    return predictions, report, summary_df


def weighted_vote(prob_results, y_true, label_encoder, weights, none_label):
    y_true_values = y_true.values if isinstance(y_true, pd.Series) else y_true
    weight_map = resolve_weights(weights)

    predictions = []
    winning_scores = []
    for row_index in range(len(y_true_values)):
        score_vector = None
        for name, probs_df in prob_results.items():
            weighted = weight_map.get(name, 0) * probs_df.iloc[row_index].astype(float).values
            score_vector = weighted if score_vector is None else score_vector + weighted
        best_index = int(np.argmax(score_vector))
        predictions.append(label_encoder.inverse_transform([best_index])[0])
        winning_scores.append(float(score_vector[best_index]))

    labels = sorted(set(y_true_values) | {none_label})
    report = classification_report(y_true_values, predictions, labels=labels)

    total = len(predictions)
    correct = int(np.sum(np.array(predictions) == np.array(y_true_values)))
    accuracy = correct / total if total else float("nan")
    summary_df = pd.DataFrame(
        [
            {
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "score_min": float(np.min(winning_scores)) if winning_scores else float("nan"),
                "score_mean": float(np.mean(winning_scores)) if winning_scores else float("nan"),
                "score_max": float(np.max(winning_scores)) if winning_scores else float("nan"),
            }
        ]
    )

    return predictions, report, summary_df


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


def run_all_ensembles(X, y, n_splits, model_params, random_state):
    X_values = X.values if isinstance(X, pd.DataFrame) else X
    label_encoder, y_encoded = prepare_labels(y)
    global_classes = np.unique(y_encoded)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    params = resolve_model_params(model_params)
    prob_store = {name: np.zeros((len(y_encoded), len(global_classes))) for name in MODEL_BUILDERS}

    for fold_index, (train_idx, val_idx) in enumerate(skf.split(X_values, y_encoded), 1):
        X_train, X_val = X_values[train_idx], X_values[val_idx]
        y_train = y_encoded[train_idx]

        for name in MODEL_BUILDERS:
            model = build_model(name, params[name], random_state)
            model.fit(X_train, y_train)

            val_probs = model.predict_proba(X_val)
            model_classes = get_model_classes(model)
            aligned_probs = align_probabilities(val_probs, model_classes, global_classes)
            prob_store[name][val_idx] = aligned_probs

        print(f"Completed fold {fold_index}/{n_splits}.")

    results = {"__meta__": {"label_encoder": label_encoder, "classes": label_encoder.classes_}}
    for name, probs in prob_store.items():
        probs_df = pd.DataFrame(probs, columns=global_classes)
        pred_indices = probs_df.idxmax(axis=1).values
        y_pred = label_encoder.inverse_transform(pred_indices)
        report = classification_report(label_encoder.inverse_transform(y_encoded), y_pred)
        results[name] = (probs_df, report)
        print(f"\nClassification Report (OOF - {name}):\n{report}")

    return results


def plot_confusion_matrix(cm, labels, output_dir: Path):
    ensure_dir(output_dir)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Weighted Vote Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color=color)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_dir / "weighted_confusion.png", dpi=150)
    plt.close()


def train_and_save_models(verified_df, feature_columns, target_column, config, params=None):
    model_dir = ensure_dir(Path(config["model_dir"]))
    X_train = verified_df[feature_columns]
    y_train = verified_df[target_column]
    label_encoder, y_encoded = prepare_labels(y_train)
    global_classes = np.unique(y_encoded)
    resolved_params = resolve_model_params(params)

    for name in MODEL_BUILDERS:
        model = build_model(name, resolved_params[name], config["random_state"])
        model.fit(X_train.values, y_encoded)
        joblib.dump(model, model_dir / f"{name.lower()}_model.joblib")

    joblib.dump(label_encoder, model_dir / "label_encoder.joblib")
    dump_json(list(global_classes), model_dir / "classes.json")
    dump_json(feature_columns, model_dir / "feature_list.json")
    dump_json(config, model_dir / "config_snapshot.json")
    print(f"Saved models and metadata to {model_dir}.")


def run_training_pipeline(config=CONFIG):
    data = load_and_merge_data(config)
    verified_df, _ = split_verified_sets(
        data,
        target_column=config["target_column"],
        min_samples=config["min_samples_per_class"],
    )
    feature_columns = select_embedding_columns(verified_df)
    X = verified_df[feature_columns]
    y = verified_df[config["target_column"]]

    results = run_all_ensembles(
        X,
        y,
        n_splits=config["n_splits"],
        model_params=None,
        random_state=config["random_state"],
    )

    lr_probs, _ = results["LogReg"]
    svc_probs, _ = results["SVC"]
    knn_probs, _ = results["KNN"]

    output_dir = ensure_dir(Path(config["output_dir"]))
    lr_probs.to_csv(output_dir / "probs_lr.csv", index=False)
    svc_probs.to_csv(output_dir / "probs_svc.csv", index=False)
    knn_probs.to_csv(output_dir / "probs_knn.csv", index=False)

    label_encoder = results["__meta__"]["label_encoder"]
    prob_map = {"LogReg": lr_probs, "SVC": svc_probs, "KNN": knn_probs}

    _, majority_report, majority_summary = majority_vote(
        prob_map, y, label_encoder, none_label=config["none_label"]
    )
    print("\nClassification Report (Majority Vote):")
    print(majority_report)
    print("Vote breakdown (totals, correct, accuracy):")
    print(majority_summary.to_string(index=False))

    weighted_predictions, weighted_report, weighted_summary = weighted_vote(
        prob_map,
        y,
        label_encoder,
        weights=None,
        none_label=config["none_label"],
    )
    print("\nClassification Report (Weighted Vote):")
    print(weighted_report)
    print("Weighted vote summary (totals, correct, accuracy, score stats):")
    print(weighted_summary.to_string(index=False))

    labels = list(label_encoder.classes_)
    if config["none_label"] not in labels:
        labels.append(config["none_label"])
    cm = confusion_matrix(y, weighted_predictions, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("\nConfusion Matrix (Weighted Vote):")
    print(cm_df.to_string())

    pd.DataFrame(
        {
            "file": verified_df["file"],
            "true_label": y,
            "weighted_pred": weighted_predictions,
        }
    ).to_csv(output_dir / "ensemble_classification_predictions.csv", index=False)

    plot_confusion_matrix(cm, labels, output_dir)
    train_and_save_models(verified_df, feature_columns, config["target_column"], config)


if __name__ == "__main__":
    run_training_pipeline()
