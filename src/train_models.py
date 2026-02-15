import os
import time
import pickle
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix,
)

RANDOM_STATE = 42


# Utility Functions

def ensure_dirs(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)

def normalize_labels(series):
    return series.str.strip().str.capitalize()

# computing multiple evaluation matrics
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


# define feature representations
def get_feature_configs():
    return {
        "bow": CountVectorizer(stop_words="english", max_features=12000, ngram_range=(1, 1)),
        "tfidf": TfidfVectorizer(stop_words="english", max_features=12000, ngram_range=(1, 1), sublinear_tf=True),
        "ngram": TfidfVectorizer(stop_words="english", max_features=18000, ngram_range=(1, 2), min_df=2),
    }

# defining ML models and hypwrparameters
def get_model_configs():
    return {
        "multinomial_nb": {
            "estimator": MultinomialNB(),
            "param_grid": {"model__alpha": [0.2, 0.4, 0.6, 0.8, 1.0]},
        },
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE),
            "param_grid": {
                "model__C": [0.5, 1.0, 2.0, 4.0],
                "model__class_weight": [None, "balanced"],
            },
        },
        "linear_svm": {
            "estimator": LinearSVC(random_state=RANDOM_STATE),
            "param_grid": {
                "model__C": [0.5, 1.0, 2.0],
                "model__class_weight": [None, "balanced"],
            },
        },
    }


# Training all the models

def run_experiments(project_root):

    data_dir = os.path.join(project_root, "data", "processed")
    models_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")

    ensure_dirs(models_dir, results_dir)

    # Load data
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    full_df = pd.read_csv(os.path.join(data_dir, "complete_dataset.csv"))

    train_df["label"] = normalize_labels(train_df["label"])
    test_df["label"] = normalize_labels(test_df["label"])
    full_df["label"] = normalize_labels(full_df["label"])

    X_train = train_df["text"]
    y_train = train_df["label"]
    X_test = test_df["text"]
    y_test = test_df["label"]

    X_full = full_df["text"]
    y_full = full_df["label"]

    feature_configs = get_feature_configs()
    model_configs = get_model_configs()

    # cross validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    overall_rows = []
    per_class_rows = []
    confusion_rows = []
    cv_rows = []
    full_rows = []

    # run all the models
    for feature_name, vectorizer in feature_configs.items():
        for model_name, config in model_configs.items():

            experiment = f"{model_name}__{feature_name}"
            print(f"\nRunning {experiment}")

            pipeline = Pipeline([
                ("vectorizer", vectorizer),
                ("model", config["estimator"]),
            ])

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=config["param_grid"],
                scoring="f1_macro",
                cv=cv,
                n_jobs=-1,
                refit=True,
            )

            start_time = time.time()
            grid.fit(X_train, y_train)
            training_time = time.time() - start_time

            best_pipeline = grid.best_estimator_
            best_params = grid.best_params_
            best_cv_score = grid.best_score_

            cv_rows.append({
                "experiment": experiment,
                "model": model_name,
                "feature": feature_name,
                "best_cv_f1_macro": best_cv_score,
                "best_params": str(best_params),
                "cv_training_time_sec": training_time,
            })

            # test evaluations
            predictions = best_pipeline.predict(X_test)
            metrics = compute_metrics(y_test, predictions)

            overall_rows.append({
                "experiment": experiment,
                "model": model_name,
                "feature": feature_name,
                "training_time_sec": training_time,
                **metrics,
            })

            # Per-class metrics
            report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

            for class_label in report:
                if class_label in ["Politics", "Sport"]:
                    per_class_rows.append({
                        "experiment": experiment,
                        "model": model_name,
                        "feature": feature_name,
                        "class_label": class_label,
                        "precision": report[class_label]["precision"],
                        "recall": report[class_label]["recall"],
                        "f1_score": report[class_label]["f1-score"],
                        "support": int(report[class_label]["support"]),
                    })

            # Confusion matrix
            cm = confusion_matrix(y_test, predictions, labels=["Politics", "Sport"])

            confusion_rows.append({
                "experiment": experiment,
                "model": model_name,
                "feature": feature_name,
                "politics_pred_politics": int(cm[0, 0]),
                "politics_pred_sport": int(cm[0, 1]),
                "sport_pred_politics": int(cm[1, 0]),
                "sport_pred_sport": int(cm[1, 1]),
            })

            # Retrain best on full dataset
            best_pipeline.fit(X_full, y_full)

            model_path = os.path.join(models_dir, f"{experiment}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(best_pipeline, f)

            full_rows.append({
                "experiment": experiment,
                "model": model_name,
                "feature": feature_name,
                "trained_on": "complete_dataset",
                "full_dataset_size": len(full_df),
                "model_path": model_path,
            })

    # Saveing all the results
    pd.DataFrame(overall_rows).sort_values("f1_macro", ascending=False).to_csv(
        os.path.join(results_dir, "all_results.csv"), index=False
    )

    pd.DataFrame(per_class_rows).to_csv(
        os.path.join(results_dir, "per_class_results.csv"), index=False
    )

    pd.DataFrame(confusion_rows).to_csv(
        os.path.join(results_dir, "confusion_matrices.csv"), index=False
    )

    pd.DataFrame(cv_rows).sort_values("best_cv_f1_macro", ascending=False).to_csv(
        os.path.join(results_dir, "cross_val_results.csv"), index=False
    )

    pd.DataFrame(full_rows).to_csv(
        os.path.join(results_dir, "full_data_models.csv"), index=False
    )

    print("\nAll experiments completed.")
    print("All CSV files saved in results/")
    print("All models saved in models/")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_experiments(root)
