import argparse
from pathlib import Path

import tensorflow as tf

from .config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one CDT image.")
    parser.add_argument("image_path", help="Path to a CDT image.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for impaired class.")
    return parser.parse_args()


def load_image(image_path: str | Path, image_size: tuple[int, int]) -> tf.Tensor:
    image = tf.io.read_file(str(image_path))
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)
    return tf.expand_dims(image, axis=0)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    image_size = tuple(config["image_size"])

    model = tf.keras.models.load_model(config["model_output_path"])
    image = load_image(args.image_path, image_size)
    probability = float(model.predict(image, verbose=0).ravel()[0])
    label = "impaired" if probability >= args.threshold else "normal"

    print(f"prediction={label}")
    print(f"impaired_probability={probability:.4f}")


if __name__ == "__main__":
    main()
