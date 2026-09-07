import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

from .config import ensure_parent_dir, load_config
from .data import build_image_index, make_dataset, split_dataframe, summarize_classes
from .model import build_binary_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CDT binary classifier.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    image_size = tuple(config["image_size"])
    dataframe = build_image_index(config["dataset_dir"], config["class_mapping"])
    train_df, val_df = split_dataframe(dataframe, config["validation_split"], config["seed"])

    Path("reports").mkdir(exist_ok=True)
    summarize_classes(dataframe).to_csv("reports/class_counts.csv", index=False)
    train_df.to_csv("reports/train_split.csv", index=False)
    val_df.to_csv("reports/validation_split.csv", index=False)

    train_ds = make_dataset(train_df, image_size, config["batch_size"], shuffle=True, seed=config["seed"])
    val_ds = make_dataset(val_df, image_size, config["batch_size"], shuffle=False, seed=config["seed"])

    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_df["label"].to_numpy())
    class_weight = {int(label): float(weight) for label, weight in zip(classes, weights)}

    model = build_binary_classifier(image_size, config["learning_rate"])
    ensure_parent_dir(config["model_output_path"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=5,
            mode="max",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config["model_output_path"],
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["epochs"],
        class_weight=class_weight,
        callbacks=callbacks,
    )

    model.save(config["model_output_path"])
    pd.DataFrame(history.history).to_csv("reports/training_history.csv", index=False)

    print(f"Saved model to {config['model_output_path']}")
    print("Saved reports to reports/")


if __name__ == "__main__":
    main()
