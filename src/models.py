"""Model definitions.

* HybridCNNXGB - a 1D-CNN learns deep features
  from the flow-feature vector; its penultimate "embedding" layer feeds an
  XGBoost classifier. `fusion=True` concatenates the embedding
  with the original scaled features before XGBoost.
* FlowAutoencoder - trained on benign traffic only; high reconstruction error
  = novel / zero-day behaviour.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from xgboost import XGBClassifier

from config import SEED

tf.random.set_seed(SEED)
np.random.seed(SEED)


def build_cnn(n_features: int, n_classes: int) -> keras.Model:
    inp = keras.Input(shape=(n_features,), name="flow")
    x = keras.layers.Reshape((n_features, 1))(inp)
    x = keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4), name="embedding")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(n_classes, activation="softmax", name="softmax")(x)
    model = keras.Model(inp, out, name="flow_cnn")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def fit_cnn(model, Xtr, ytr, Xva, yva, epochs=30, batch_size=None, verbose=2):
    batch_size = batch_size or int(np.clip(len(Xtr) // 100, 32, 512))
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
          keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)]
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=batch_size,
                     callbacks=cb, verbose=verbose)
    return hist.history


def make_xgb(n_classes: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9, colsample_bytree=0.8,
        tree_method="hist", objective="multi:softprob" if n_classes > 2 else "binary:logistic",
        eval_metric="mlogloss" if n_classes > 2 else "logloss", early_stopping_rounds=25,
        n_jobs=-1, random_state=SEED)


class HybridCNNXGB:
    def __init__(self, n_features, n_classes, fusion=False, cnn=None):
        self.n_features, self.n_classes, self.fusion = n_features, n_classes, fusion
        self.cnn = cnn or build_cnn(n_features, n_classes)
        self._embedder = None
        self.xgb = make_xgb(n_classes)
        self.history = None

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = keras.Model(self.cnn.input, self.cnn.get_layer("embedding").output)
        return self._embedder

    def features(self, X):
        E = self.embedder.predict(X, batch_size=4096, verbose=0)
        return np.hstack([E, X]) if self.fusion else E

    def fit(self, Xtr, ytr, Xva, yva, epochs=30, train_cnn=True, verbose=2):
        if train_cnn:
            self.history = fit_cnn(self.cnn, Xtr, ytr, Xva, yva, epochs=epochs, verbose=verbose)
        self.xgb.fit(self.features(Xtr), ytr, eval_set=[(self.features(Xva), yva)], verbose=False)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        p = self.xgb.predict_proba(self.features(X))
        return p if p.ndim == 2 else np.column_stack([1 - p, p])

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def save(self, folder: Path, name: str):
        folder.mkdir(parents=True, exist_ok=True)
        self.cnn.save(folder / f"{name}_cnn.keras")
        self.xgb.save_model(folder / f"{name}_xgb.json")
        (folder / f"{name}_meta.json").write_text(json.dumps(
            {"n_features": self.n_features, "n_classes": self.n_classes, "fusion": self.fusion}))

    @classmethod
    def load(cls, folder: Path, name: str):
        meta = json.loads((folder / f"{name}_meta.json").read_text())
        cnn = keras.models.load_model(folder / f"{name}_cnn.keras")
        obj = cls(meta["n_features"], meta["n_classes"], meta["fusion"], cnn=cnn)
        obj.xgb = XGBClassifier()
        obj.xgb.load_model(folder / f"{name}_xgb.json")
        return obj


class FlowAutoencoder:
    def __init__(self, n_features):
        inp = keras.Input(shape=(n_features,))
        x = keras.layers.Dense(64, activation="relu")(inp)
        x = keras.layers.Dense(24, activation="relu")(x)
        z = keras.layers.Dense(8, activation="relu", name="latent")(x)
        x = keras.layers.Dense(24, activation="relu")(z)
        x = keras.layers.Dense(64, activation="relu")(x)
        out = keras.layers.Dense(n_features, activation="sigmoid")(x)
        self.model = keras.Model(inp, out, name="flow_autoencoder")
        self.model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        self.threshold = None

    def fit(self, X_benign, X_val_benign, epochs=60, percentile=99.0, verbose=2):
        cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]
        bs = int(np.clip(len(X_benign) // 50, 16, 256))
        self.model.fit(X_benign, X_benign, validation_data=(X_val_benign, X_val_benign),
                       epochs=epochs, batch_size=bs, callbacks=cb, verbose=verbose)
        self.threshold = float(np.percentile(self.score(X_val_benign), percentile))
        return self

    def score(self, X):
        X = np.asarray(X, dtype=np.float32)
        R = self.model.predict(X, batch_size=4096, verbose=0)
        return np.mean((X - R) ** 2, axis=1)

    def save(self, folder: Path):
        self.model.save(folder / "autoencoder.keras")
        (folder / "autoencoder_meta.json").write_text(json.dumps({"threshold": self.threshold}))

    @classmethod
    def load(cls, folder: Path):
        obj = cls.__new__(cls)
        obj.model = keras.models.load_model(folder / "autoencoder.keras")
        obj.threshold = json.loads((folder / "autoencoder_meta.json").read_text())["threshold"]
        return obj


def timed(fn, *a, **k):
    t = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t


def dump(obj, path):
    joblib.dump(obj, path)


def load(path):
    return joblib.load(path)
