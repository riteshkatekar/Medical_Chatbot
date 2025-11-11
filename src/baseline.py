


#!/usr/bin/env python3


"""
src/baseline.py

TF-IDF + Keras baseline training with robust headerless / single-column CSV handling.

This version:
 - Detects headerless / single-column CSVs and reconstructs intent + __synth_message.
 - If detect_columns returns suspicious values, prefers safe defaults:
     intent_col = first dataframe column
     message_col = '__synth_message' (created from remaining columns or by splitting single-col text)
 - Filters out unseen test labels (-1) before passing validation_data to Keras.
 - Clear diagnostics when no training texts would be generated.
"""

import argparse
import sys
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
import joblib
from typing import List, Dict
from collections import Counter

# local helpers
from preprocessing import (
    read_csv_fuzzy,
    detect_columns,
    split_symptoms_from_message,
    synthesize_utterances_from_symptoms,
    build_tfidf,
    build_symptom_intent_map,
)

# Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks, regularizers
except Exception as e:
    raise RuntimeError("TensorFlow / Keras required for this script. Install tensorflow. Error: " + str(e))


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_pickle(obj, path: str):
    ensure_dir(os.path.dirname(path) or ".")
    joblib.dump(obj, path)


def _synthesize_for_df(df: pd.DataFrame, intent_col: str, message_col: str, synth_per_row: int = 4):
    texts: List[str] = []
    labels: List[str] = []
    for _, row in df.iterrows():
        intent = str(row[intent_col]).strip()
        message = str(row[message_col]).strip() if message_col in df.columns else ""
        if ("," in message or "_" in message or ";" in message or "|" in message) and len(message.split()) <= 15:
            symptoms = split_symptoms_from_message(message)
            synths = synthesize_utterances_from_symptoms(symptoms, n_variations=synth_per_row)
            if not synths:
                synths = [", ".join(symptoms)] if symptoms else []
            for s in synths:
                texts.append(s)
                labels.append(intent)
        else:
            if message:
                texts.append(message)
                labels.append(intent)
    return texts, labels


def _natural_texts_from_df(df: pd.DataFrame, intent_col: str, message_col: str):
    texts: List[str] = []
    labels: List[str] = []
    for _, row in df.iterrows():
        intent = str(row[intent_col]).strip()
        message = ""
        if message_col in df.columns:
            message = str(row[message_col]).strip()
        if message:
            texts.append(message)
            labels.append(intent)
    return texts, labels


DEFAULT_TFIDF_MAX = 20000


def build_keras_model(input_dim: int, num_classes: int, hidden_units: int = 128, dropout: float = 0.5) -> keras.Model:
    inp = keras.Input(shape=(input_dim,), name="tfidf_input")
    x = layers.Dense(hidden_units, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(max(64, hidden_units // 2), activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(dropout * 0.6)(x)
    out = layers.Dense(num_classes, activation="softmax", name="class_probs")(x)
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def _group_key_from_message(msg: str) -> str:
    if not isinstance(msg, str):
        return ""
    s = msg.strip().lower()
    s = " ".join(s.split())
    # normalize characters that don't matter for the canonical message
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"[\.\!\?;:]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _maybe_fix_headerless_or_singlecol_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make robust repairs when:
     - CSV had no header and pandas used first row as header (column names contain commas),
     - OR CSV parsed into a single column where each row is "intent, symptom, symptom, ...".
    Produces a new column '__synth_message' containing joined symptoms (or returns original df).
    """
    cols = list(df.columns)

    # Case A: suspicious column names (pandas used first row as header => column names contain commas)
    if any("," in str(c) for c in cols):
        # We'll create '__synth_message' by joining all other fields (if any) per row
        print("Detected suspicious columns (likely headerless CSV). Rebuilding message column from remaining columns.")
        # if there's more than 1 column, join remaining; otherwise fall through
        if len(cols) > 1:
            intent_col = cols[0]
            def join_row_parts(row):
                parts = []
                for c in cols[1:]:
                    v = str(row[c]).strip()
                    if v and v.lower() not in ("nan", "na", ""):
                        parts.append(v)
                return ", ".join(parts)
            df = df.copy()
            df["__synth_message"] = df.apply(join_row_parts, axis=1)
            return df
        # else: single-column (will be handled below)

    # Case B: single-column case where each row holds comma-separated tokens
    if len(cols) == 1:
        col0 = cols[0]
        # check if many rows contain commas
        n_with_commas = int(df[col0].astype(str).str.contains(",").sum())
        if n_with_commas > 0:
            print("Detected single-column CSV with comma-separated rows -> parsing into intent + __synth_message.")
            # split each row on commas: first token -> intent, rest -> joined message
            intents = []
            messages = []
            for raw in df[col0].astype(str).tolist():
                parts = [p.strip() for p in raw.split(",") if p.strip() and p.strip().lower() not in ("nan", "na")]
                if not parts:
                    intents.append("")
                    messages.append("")
                elif len(parts) == 1:
                    intents.append(parts[0])
                    messages.append("")
                else:
                    intents.append(parts[0])
                    messages.append(", ".join(parts[1:]))
            new_df = pd.DataFrame({cols[0]: intents})
            new_df["__synth_message"] = messages
            return new_df

    # otherwise no change
    return df


def train_baseline(data_path: str, desc_path: str, prec_path: str, save_dir: str,
                   tfidf_max_features: int = DEFAULT_TFIDF_MAX, ngram_min: int = 1, ngram_max: int = 2,
                   test_size: float = 0.15, random_state: int = 42,
                   batch_size: int = 32, epochs: int = 20, synth_per_row: int = 4):
    # Load CSVs (robust file reading)
    df = read_csv_fuzzy(data_path)
    df = _maybe_fix_headerless_or_singlecol_df(df)

    df_desc = read_csv_fuzzy(desc_path) if desc_path else None
    df_prec = read_csv_fuzzy(prec_path) if prec_path else None

    intent_col, message_col, response_col = detect_columns(df)

    # If detect_columns produced suspicious values (same col for both, or message missing),
    # prefer safe defaults: first column = intent, '__synth_message' if present.
    first_col = list(df.columns)[0]
    if (message_col == intent_col) or (message_col not in df.columns and "__synth_message" in df.columns):
        if "__synth_message" in df.columns:
            message_col = "__synth_message"
            intent_col = first_col
            print("Adjusted columns: intent -> '%s', message -> '__synth_message' (fallback)." % intent_col)
        elif len(df.columns) > 1:
            # prefer second column as message
            message_col = list(df.columns)[1]
            intent_col = first_col
            print("Adjusted columns: intent -> '%s', message -> '%s' (fallback second column)." % (intent_col, message_col))
        else:
            print("Error: cannot determine message column. Dataframe columns:", df.columns, file=sys.stderr)
            sys.exit(1)

    print(f"Detected columns -> intent: {intent_col}, message: {message_col}")

    # Build a canonical group key for each row (natural message)
    df = df.copy()
    if message_col not in df.columns:
        df[message_col] = ""
    df["_group_key"] = df[message_col].apply(_group_key_from_message)

    # Representative intent per group
    grouped = df.groupby("_group_key")[intent_col].agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]).reset_index()
    group_keys = grouped["_group_key"].values
    group_labels = grouped[intent_col].values

    # check group counts to decide split strategy
    group_label_counts = Counter(group_labels)
    min_group_label_count = min(group_label_counts.values()) if group_label_counts else 0
    singletons = [lbl for lbl, cnt in group_label_counts.items() if cnt < 2]

    if group_label_counts and min_group_label_count >= 2:
        # safe group-aware split
        train_groups, test_groups = train_test_split(
            group_keys,
            test_size=test_size,
            random_state=random_state,
            stratify=group_labels if len(np.unique(group_labels)) > 1 else None
        )
        train_df = df[df["_group_key"].isin(train_groups)].reset_index(drop=True)
        test_df = df[df["_group_key"].isin(test_groups)].reset_index(drop=True)
        print("Using group-aware split.")
    else:
        # fallback to row-level split
        if singletons:
            print("Warning: some labels have <2 groups; group-aware split disabled.")
            print("Example singleton labels (up to 10):", singletons[:10])
        # decide whether we can stratify by row-level label counts
        label_row_counts = Counter(df[intent_col].values)
        min_label_row_count = min(label_row_counts.values()) if label_row_counts else 0
        stratify_arg = df[intent_col] if min_label_row_count >= 2 and len(label_row_counts) > 1 else None
        if stratify_arg is not None:
            print("Falling back to row-level stratified split on intent labels.")
        else:
            print("Falling back to plain random row-level split (stratify not possible).")
        indices = np.arange(len(df))
        tr_idx, te_idx = train_test_split(indices, test_size=test_size, random_state=random_state, stratify=stratify_arg)
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        test_df = df.iloc[te_idx].reset_index(drop=True)
        train_groups = sorted(list(set(train_df["_group_key"].values)))
        test_groups = sorted(list(set(test_df["_group_key"].values)))

    print(f"Group-aware split -> train rows: {train_df.shape[0]}, test rows: {test_df.shape[0]}, groups train: {len(train_groups)}, groups test: {len(test_groups)}")

    # Synthesize utterances only for training rows (augment)
    texts_train, labels_train = _synthesize_for_df(train_df, intent_col, message_col, synth_per_row=synth_per_row)

    # For evaluation, use natural/original messages only (no synthesis)
    texts_test, labels_test = _natural_texts_from_df(test_df, intent_col, message_col)

    # If texts_train empty, give actionable diagnostic and abort
    if not texts_train:
        print("No training texts found after preprocessing. Possible reasons:")
        print("- message column ('%s') is empty for all rows." % message_col)
        print("- dataset rows were not parsed as expected (header issues or single-column parsing).")
        print("Suggestions:")
        print("- Inspect the first 10 rows of your CSV with pandas read_csv to confirm parsing.")
        print("- If your CSV is headerless (first row is data), re-run generate_dataset.py or provide --headerless flag (not implemented) or ensure dataset has header.")
        print("Debug sample (first 10 rows):")
        print(df[[intent_col, message_col]].head(10).to_string(index=False))
        sys.exit(1)

    if not texts_test:
        print("WARNING: No natural test texts found. Test set will be synthesized from test rows.")
        texts_test, labels_test = _synthesize_for_df(test_df, intent_col, message_col, synth_per_row=1)

    # Build TF-IDF on training texts only
    tfidf = build_tfidf(texts_train, max_features=tfidf_max_features, ngram_range=(ngram_min, ngram_max))
    X_train = tfidf.transform(texts_train)
    X_test = tfidf.transform(texts_test)

    le = LabelEncoder()
    y_train = le.fit_transform(labels_train)

    # safe mapping for y_test (use -1 for unseen labels)
    try:
        y_test = le.transform(labels_test)
    except Exception:
        y_test = []
        for lbl in labels_test:
            if lbl in list(le.classes_):
                y_test.append(int(np.where(le.classes_ == lbl)[0][0]))
            else:
                y_test.append(-1)
        y_test = np.array(y_test, dtype=int)

    num_classes = len(le.classes_)
    print(f"Classes found: {num_classes}")

    # Convert to dense arrays for Keras (may use substantial memory)
    print("Converting TF-IDF sparse matrices to dense arrays for Keras training (may use substantial memory).")
    X_train_arr = X_train.toarray()
    X_test_arr = X_test.toarray()

    # Remove unseen test labels (-1) from validation arrays BEFORE passing them to Keras
    val_mask = np.array([lbl >= 0 for lbl in y_test], dtype=bool)
    num_val_total = y_test.shape[0]
    num_val_valid = int(val_mask.sum())
    num_val_invalid = num_val_total - num_val_valid
    if num_val_invalid > 0:
        print(f"Note: {num_val_invalid} test samples have labels unseen by the label-encoder and will be excluded from validation (they remain in the test set used for later reporting).")
    X_test_arr_val = X_test_arr[val_mask]
    y_test_val = y_test[val_mask]

    # class weights
    try:
        class_weights_values = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
        class_weight = {int(c): float(w) for c, w in zip(np.unique(y_train), class_weights_values)}
    except Exception:
        class_weight = None

    # build model
    model = build_keras_model(input_dim=X_train_arr.shape[1], num_classes=num_classes, hidden_units=128, dropout=0.5)
    model.summary()

    ensure_dir(save_dir)
    keras_native_path = os.path.join(save_dir, "keras_model.keras")
    keras_h5_path = os.path.join(save_dir, "keras_model.h5")

    es = callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1)
    chk_native = callbacks.ModelCheckpoint(keras_native_path, monitor="val_loss", save_best_only=True, verbose=1)
    chk_h5 = callbacks.ModelCheckpoint(keras_h5_path, monitor="val_loss", save_best_only=True, verbose=0)

    validation_data = (X_test_arr_val, y_test_val) if y_test_val.size > 0 else None

    history = model.fit(
        X_train_arr, y_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es, chk_native, chk_h5],
        class_weight=class_weight,
        verbose=2
    )

    # load best model (native preferred)
    if os.path.exists(keras_native_path):
        model = keras.models.load_model(keras_native_path)
    elif os.path.exists(keras_h5_path):
        model = keras.models.load_model(keras_h5_path)

    # Evaluate on full test set for reporting: only compute metrics where labels are valid
    y_pred_probs = model.predict(X_test_arr, batch_size=batch_size)
    y_pred = np.argmax(y_pred_probs, axis=1)
    valid_idx = [i for i, lbl in enumerate(y_test) if lbl >= 0]
    if valid_idx:
        valid_y_test = y_test[valid_idx]
        valid_y_pred = np.array(y_pred)[valid_idx]

        acc = accuracy_score(valid_y_test, valid_y_pred)

        unique_labels = np.unique(valid_y_test)
        try:
            target_names = [str(x) for x in le.inverse_transform(unique_labels)]
        except Exception:
            classes = getattr(le, "classes_", None)
            if classes is not None:
                target_names = [str(classes[int(i)]) if int(i) < len(classes) else str(i) for i in unique_labels]
            else:
                target_names = [str(i) for i in unique_labels]

        try:
            report = classification_report(valid_y_test, valid_y_pred, labels=unique_labels, target_names=target_names, zero_division=0)
        except Exception as e:
            print("Warning: classification_report failed with error:", e)
            report = f"Classification report unavailable; accuracy: {acc:.4f}"

        print("=== Keras baseline training complete ===")
        print(f"Train samples: {X_train_arr.shape[0]}, Test samples: {X_test_arr.shape[0]} (valid for metrics: {len(valid_idx)})")
        print(f"Test accuracy (valid subset): {acc:.4f}")
        print("Classification report (test valid subset):")
        print(report)
    else:
        print("Skipping detailed evaluation because no valid test labels were available (all -1).")

    # Save artifacts
    tfidf_path = os.path.join(save_dir, "tfidf.pkl")
    le_path = os.path.join(save_dir, "label_encoder.pkl")
    save_pickle(tfidf, tfidf_path)
    try:
        model.save(keras_native_path)
        model.save(keras_h5_path)
    except Exception as e:
        print("Warning: saving model failed for one of the formats:", e)
    save_pickle(le, le_path)
    print(f"Saved: {tfidf_path}, {keras_native_path}, {keras_h5_path}, {le_path}")

    # metadata
    metadata = {"descriptions": {}, "precautions": {}, "symptom_map": {}}

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
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--synth_per_row", type=int, default=4)
    args = p.parse_args()

    train_baseline(
        args.data_path, args.desc_path, args.prec_path, args.save_dir,
        tfidf_max_features=args.tfidf_max_features, ngram_min=args.ngram_min,
        ngram_max=args.ngram_max, epochs=args.epochs, batch_size=args.batch_size,
        synth_per_row=args.synth_per_row
    )


if __name__ == "__main__":
    main()
