#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""TP2 classification experiments (MLP/CNN + hyperparameter impact).

This script is designed to produce reproducible metrics and plots for the TP2 report.
It trains multiple model configurations on Fashion-MNIST and exports:
- training loss curves,
- accuracy/time comparison charts,
- a CSV summary.
"""

from __future__ import annotations

import csv
import argparse
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "Rapports" / "Capture_ecran_TP2_Classification"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0
    return train_images, train_labels, test_images, test_labels


def build_mlp_baseline() -> tf.keras.Model:
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10),
    ])


def build_mlp_deeper() -> tf.keras.Model:
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10),
    ])


def build_cnn() -> tf.keras.Model:
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10),
    ])


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )


def plot_loss(name: str, history: tf.keras.callbacks.History) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(history.epoch, history.history["loss"], marker="o", label="loss train")
    if "val_loss" in history.history:
        plt.plot(history.epoch, history.history["val_loss"], marker="s", label="loss val")
    plt.title(f"Courbe de loss - {name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"loss_{name}.png", dpi=180)
    plt.close()


def train_one(
    name: str,
    build_fn: Callable[[], tf.keras.Model],
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    is_cnn: bool = False,
) -> Dict[str, float]:
    model = build_fn()
    compile_model(model, learning_rate)

    x_train = train_images[..., np.newaxis] if is_cnn else train_images
    x_test = test_images[..., np.newaxis] if is_cnn else test_images

    start = time.time()
    history = model.fit(
        x_train,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=2,
    )
    train_time = time.time() - start

    test_loss, test_acc = model.evaluate(x_test, test_labels, verbose=0)

    plot_loss(name, history)

    return {
        "name": name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "train_time_s": train_time,
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
    }


def save_summary(results: List[Dict[str, float]]) -> None:
    csv_path = OUT_DIR / "tp2_results_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "learning_rate", "batch_size", "epochs", "train_time_s", "test_loss", "test_acc"],
        )
        writer.writeheader()
        writer.writerows(results)

    names = [r["name"] for r in results]
    acc = [r["test_acc"] * 100 for r in results]
    times = [r["train_time_s"] for r in results]

    plt.figure(figsize=(9, 4))
    plt.bar(names, acc)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Accuracy test (%)")
    plt.title("Comparaison des accuracies (TP2)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tp2_accuracy_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.bar(names, times)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Temps entrainement (s)")
    plt.title("Comparaison des temps d'entrainement (TP2)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tp2_time_comparison.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="TP2 classification experiments")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs per experiment")
    parser.add_argument(
        "--train-limit",
        type=int,
        default=25000,
        help="Limit training set size for faster iterations (0 keeps full dataset)",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=5000,
        help="Limit test set size for faster iterations (0 keeps full dataset)",
    )
    args = parser.parse_args()

    train_images, train_labels, test_images, test_labels = load_data()

    if args.train_limit and args.train_limit > 0:
        train_images = train_images[: args.train_limit]
        train_labels = train_labels[: args.train_limit]
    if args.test_limit and args.test_limit > 0:
        test_images = test_images[: args.test_limit]
        test_labels = test_labels[: args.test_limit]

    print("Train images:", train_images.shape)
    print("Train labels:", train_labels.shape)
    print("Test images:", test_images.shape)
    print("Test labels:", test_labels.shape)

    experiments = [
        ("mlp_baseline", build_mlp_baseline, 0.001, 100, False),
        ("mlp_lr_0_1", build_mlp_baseline, 0.1, 100, False),
        ("mlp_batch_1000", build_mlp_baseline, 0.001, 1000, False),
        ("mlp_deeper", build_mlp_deeper, 0.001, 100, False),
        ("cnn_baseline", build_cnn, 0.001, 100, True),
    ]

    results: List[Dict[str, float]] = []
    for name, fn, lr, bs, is_cnn in experiments:
        print(f"\\n=== {name} | lr={lr} | batch={bs} ===")
        result = train_one(
            name=name,
            build_fn=fn,
            train_images=train_images,
            train_labels=train_labels,
            test_images=test_images,
            test_labels=test_labels,
            learning_rate=lr,
            batch_size=bs,
            epochs=args.epochs,
            is_cnn=is_cnn,
        )
        print(f"Test acc: {result['test_acc']:.4f} | time: {result['train_time_s']:.2f}s")
        results.append(result)

    save_summary(results)

    print("\\nSorties generees dans:", OUT_DIR)


if __name__ == "__main__":
    main()
