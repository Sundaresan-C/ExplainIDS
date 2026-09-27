"""Adversarial robustness evaluation + FGSM adversarial training (our extension)."""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from tensorflow import keras

from config import SEED
from models import HybridCNNXGB

EPSILONS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]


def fgsm(cnn: keras.Model, X, y, eps, batch=4096):
    """Fast Gradient Sign Method in the [0,1]-scaled feature space (white-box on the CNN)."""
    if eps == 0:
        return X.copy()
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    out = []
    for i in range(0, len(X), batch):
        xb = tf.convert_to_tensor(X[i:i + batch])
        yb = tf.convert_to_tensor(y[i:i + batch])
        with tf.GradientTape() as tape:
            tape.watch(xb)
            loss = loss_fn(yb, cnn(xb, training=False))
        g = tape.gradient(loss, xb)
        out.append(np.clip(xb.numpy() + eps * np.sign(g.numpy()), 0, 1))
    return np.vstack(out).astype(np.float32)


def gaussian(X, eps, rng):
    return np.clip(X + rng.normal(0, eps, X.shape), 0, 1).astype(np.float32) if eps else X.copy()


def evaluate(model: HybridCNNXGB, X, y):
    rng = np.random.default_rng(SEED)
    rows = []
    for eps in EPSILONS:
        for kind, Xa in (("Gaussian noise", gaussian(X, eps, rng)), ("FGSM", fgsm(model.cnn, X, y, eps))):
            p = model.predict(Xa)
            rows.append({"attack": kind, "epsilon": eps, "accuracy": float(accuracy_score(y, p)),
                         "macro_f1": float(f1_score(y, p, average="macro"))})
    return rows


def adversarial_training(base: HybridCNNXGB, Xtr, ytr, Xva, yva, eps=0.05, epochs=6):
    """Fine-tune a copy of the CNN on clean + FGSM examples, then refit XGBoost."""
    cnn = keras.models.clone_model(base.cnn)
    cnn.set_weights(base.cnn.get_weights())
    cnn.compile(optimizer=keras.optimizers.Adam(5e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    rng = np.random.default_rng(SEED)
    for ep in range(epochs):
        idx = rng.choice(len(Xtr), min(len(Xtr), 60_000), replace=False)
        Xa = fgsm(cnn, Xtr[idx], ytr[idx], eps * rng.uniform(0.3, 1.0))
        Xmix = np.vstack([Xtr[idx], Xa]); ymix = np.concatenate([ytr[idx], ytr[idx]])
        cnn.fit(Xmix, ymix, validation_data=(Xva, yva), epochs=1, batch_size=512, verbose=0)
        print(f"  adversarial epoch {ep + 1}/{epochs}")
    robust = HybridCNNXGB(base.n_features, base.n_classes, fusion=base.fusion, cnn=cnn)
    robust.fit(Xtr, ytr, Xva, yva, train_cnn=False)
    return robust
