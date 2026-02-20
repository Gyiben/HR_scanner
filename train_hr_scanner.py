import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def train_risk_model() -> Tuple[Pipeline, bool]:
    kaggle_zip_path = "train.csv.zip"

    if os.path.exists(kaggle_zip_path):
        df = pd.read_csv(kaggle_zip_path)

        label_cols = [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ]
        df["is_risky"] = (df[label_cols].sum(axis=1) > 0).astype(int)
        df = df[["comment_text", "is_risky"]].dropna()

        texts = df["comment_text"].astype(str).tolist()
        labels = df["is_risky"].values
        using_kaggle = True

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=33, stratify=labels
    )

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=20000)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Print evaluation metrics
    print("\n--- Model Evaluation ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["safe", "risky"], zero_division=0))
    print("----------------------------------------------\n")
    return model, using_kaggle


def save_model(model: Pipeline, using_kaggle: bool, path: str = "hr_scanner_model.pkl") -> None:
    with open(path, "wb") as f:
        pickle.dump((model, using_kaggle), f)


if __name__ == "__main__":
    print("Training Predictive HR Scanner model...")
    model, used_kaggle = train_risk_model()
    save_model(model, used_kaggle)
    source = "Kaggle Toxic Comment dataset (train.csv.zip)" if used_kaggle else "synthetic demo dataset"
    print(f"Model trained using: {source}")
    print("Saved model to hr_scanner_model.pkl")

