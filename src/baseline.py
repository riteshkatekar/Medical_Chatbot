"""
baseline.py

TF-IDF + LogisticRegression baseline training.
Saves symptom -> intent map into system_metadata.pkl for use at inference.
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

from preprocessing import read_csv_fuzzy, detect_columns, split_symptoms_from_message, synthesize_utterances_from_symptoms, build_tfidf, build_symptom_intent_map
from preprocessing import normalize_symptom_token
from typing import List, Dict
from collections import Counter

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_pickle(obj, path: str):
    ensure_dir(os.path.dirname(path) or ".")
    joblib.dump(obj, path)

def prepare_dataset(df: pd.DataFrame, intent_col: str, message_col: str, response_col: str = None, synth_per_row: int = 4):
    texts: List[str] = []
    labels: List[str] = []
    meta: List[Dict] = []
    for _, row in df.iterrows():
        intent = str(row[intent_col]).strip()
        message = ""
        if message_col in df.columns:
            message = str(row[message_col]).strip()
        # Heuristic to detect token lists
        if ("," in message or "_" in message or ";" in message or "|" in message) and len(message.split()) <= 15:
            symptoms = split_symptoms_from_message(message)
            synths = synthesize_utterances_from_symptoms(symptoms, n_variations=synth_per_row)
            if not synths:
                synths = [", ".join(symptoms)] if symptoms else []
            for s in synths:
                texts.append(s)
                labels.append(intent)
                meta.append({"intent": intent, "symptoms": symptoms, "source": "synth"})
        else:
            if message:
                texts.append(message)
                labels.append(intent)
                meta.append({"intent": intent, "symptoms": split_symptoms_from_message(message), "source": "natural"})
            else:
                continue
    return texts, labels, meta

DEFAULT_TFIDF_MAX = 20000

def train_baseline(data_path: str, desc_path: str, prec_path: str, save_dir: str,
                   tfidf_max_features: int = DEFAULT_TFIDF_MAX, ngram_min: int = 1, ngram_max: int = 2,
                   test_size: float = 0.15, random_state: int = 42):
    df = read_csv_fuzzy(data_path)
    df_desc = read_csv_fuzzy(desc_path) if desc_path else None
    df_prec = read_csv_fuzzy(prec_path) if prec_path else None

    intent_col, message_col, response_col = detect_columns(df)

    texts, labels, meta = prepare_dataset(df, intent_col, message_col, response_col, synth_per_row=4)

    if not texts:
        print("No training texts found after preprocessing. Aborting.", file=sys.stderr)
        sys.exit(1)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    tfidf = build_tfidf(texts, max_features=tfidf_max_features, ngram_range=(ngram_min, ngram_max))
    X = tfidf.transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="saga", C=1.0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # target_names for classification_report:
    inv_labels = le.inverse_transform(sorted(np.unique(y)))
    report = classification_report(y_test, y_pred, target_names=inv_labels)
    print("=== Baseline training complete ===")
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"Test accuracy: {acc:.4f}")
    print("Classification report (test):")
    print(report)

    ensure_dir(save_dir)
    tfidf_path = os.path.join(save_dir, "tfidf.pkl")
    model_path = os.path.join(save_dir, "sklearn_baseline.pkl")
    le_path = os.path.join(save_dir, "label_encoder.pkl")

    save_pickle(tfidf, tfidf_path)
    save_pickle(clf, model_path)
    save_pickle(le, le_path)

    print(f"Saved: {tfidf_path}, {model_path}, {le_path}")

    # Build metadata: descriptions and precautions
    metadata = {
        "descriptions": {},
        "precautions": {},
        "symptom_map": {}
    }

    def df_to_map(dframe):
        m = {}
        if dframe is None:
            return m
        if dframe.shape[1] >= 2:
            kcol = dframe.columns[0]
            vcol = dframe.columns[1]
            for _, r in dframe.iterrows():
                key = str(r[kcol]).strip()
                val = str(r[vcol]).strip()
                if key and key.lower() not in ("nan", "na"):
                    m[key] = val
        else:
            for _, r in dframe.iterrows():
                raw = str(r[dframe.columns[0]])
                if "\t" in raw:
                    k, v = raw.split("\t", 1)
                    m[k.strip()] = v.strip()
        return m

    metadata["descriptions"] = df_to_map(df_desc)
    # precautions: first col key, remaining columns as items
    prec_map = {}
    if df_prec is not None:
        for _, r in df_prec.iterrows():
            key = str(r[df_prec.columns[0]]).strip()
            vals = []
            for c in df_prec.columns[1:]:
                v = str(r[c]).strip()
                if v and v.lower() not in ("nan", "na"):
                    vals.append(v)
            if key:
                prec_map[key] = vals
    metadata["precautions"] = prec_map

    # Build symptom -> intent frequency map using original dataframe tokens
    symptom_map = build_symptom_intent_map(df, intent_col, message_col)
    metadata["symptom_map"] = symptom_map

    meta_path = os.path.join(save_dir, "system_metadata.pkl")
    save_pickle(metadata, meta_path)
    print(f"Saved metadata: {meta_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--desc_path", required=False, default=None)
    p.add_argument("--prec_path", required=False, default=None)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--tfidf_max_features", type=int, default=20000)
    p.add_argument("--ngram_min", type=int, default=1)
    p.add_argument("--ngram_max", type=int, default=2)
    args = p.parse_args()

    train_baseline(args.data_path, args.desc_path, args.prec_path, args.save_dir, tfidf_max_features=args.tfidf_max_features, ngram_min=args.ngram_min, ngram_max=args.ngram_max)

if __name__ == "__main__":
    main()
