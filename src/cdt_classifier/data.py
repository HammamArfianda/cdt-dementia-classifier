from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def build_image_index(dataset_dir: str | Path, class_mapping: dict[str, str]) -> pd.DataFrame:
    """Create a dataframe of image paths and binary labels from class folders."""
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {dataset_path}. "
            "Update dataset_dir in configs/config.yaml."
        )

    rows = []
    for source_label, binary_label in class_mapping.items():
        class_dir = dataset_path / source_label
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Expected class folder does not exist: {class_dir}. "
                "Either rename the folder or update class_mapping in configs/config.yaml."
            )

        for image_path in class_dir.rglob("*"):
            if image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                rows.append(
                    {
                        "filepath": str(image_path),
                        "original_label": source_label,
                        "binary_label": binary_label,
                    }
                )

    if not rows:
        raise ValueError(f"No supported images found in {dataset_path}.")

    dataframe = pd.DataFrame(rows)
    dataframe["label"] = dataframe["binary_label"].map({"normal": 0, "impaired": 1})

    if dataframe["label"].isna().any():
        bad_labels = sorted(dataframe.loc[dataframe["label"].isna(), "binary_label"].unique())
        raise ValueError(f"Only 'normal' and 'impaired' binary labels are supported. Got: {bad_labels}")

    return dataframe


def split_dataframe(
    dataframe: pd.DataFrame,
    validation_split: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stratified train/validation split."""
    train_df, val_df = train_test_split(
        dataframe,
        test_size=validation_split,
        random_state=seed,
        stratify=dataframe["label"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _load_image(filepath: tf.Tensor, label: tf.Tensor, image_size: tuple[int, int]) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(filepath)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize_with_pad(image, image_size[0], image_size[1])
    image = tf.cast(image, tf.float32)
    return image, label


def make_dataset(
    dataframe: pd.DataFrame,
    image_size: tuple[int, int],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    """Build a TensorFlow dataset from a dataframe."""
    paths = dataframe["filepath"].to_numpy()
    labels = dataframe["label"].astype("float32").to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe), seed=seed, reshuffle_each_iteration=True)

    dataset = dataset.map(
        lambda path, label: _load_image(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def summarize_classes(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return class counts for logging and reports."""
    return (
        dataframe.groupby(["original_label", "binary_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["binary_label", "original_label"])
    )
