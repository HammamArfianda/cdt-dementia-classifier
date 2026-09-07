import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from .config import load_config
from .data import build_image_index, make_dataset, split_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the CDT binary classifier.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for impaired class.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    image_size = tuple(config["image_size"])

    dataframe = build_image_index(config["dataset_dir"], config["class_mapping"])
    _, val_df = split_dataframe(dataframe, config["validation_split"], config["seed"])
    val_ds = make_dataset(val_df, image_size, config["batch_size"], shuffle=False, seed=config["seed"])

    model = tf.keras.models.load_model(config["model_output_path"])
    probabilities = model.predict(val_ds).ravel()
    predictions = (probabilities >= args.threshold).astype(int)
    y_true = val_df["label"].astype(int).to_numpy()

    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_true,
        predictions,
        target_names=["normal", "impaired"],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv("reports/evaluation_metrics.csv")

    matrix = confusion_matrix(y_true, predictions)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["normal", "impaired"],
        yticklabels=["normal", "impaired"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrix.png", dpi=160)

    print(pd.DataFrame(report).transpose())
    print("Saved evaluation outputs to reports/")


if __name__ == "__main__":
    main()
