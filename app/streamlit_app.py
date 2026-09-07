from pathlib import Path
import sys
import tempfile

import streamlit as st
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cdt_classifier.config import load_config
from src.cdt_classifier.predict import load_image


st.set_page_config(page_title="CDT Classifier", page_icon="CDT", layout="centered")

st.title("CDT Classifier")
st.caption("Research demo only. Not a medical diagnosis tool.")

config_path = PROJECT_ROOT / "configs" / "config.yaml"
config = load_config(config_path)
model_path = PROJECT_ROOT / config["model_output_path"]

uploaded_file = st.file_uploader("Upload a clock drawing image", type=["jpg", "jpeg", "png", "tif", "tiff"])

if not model_path.exists():
    st.warning("Train a model first before using the demo app.")
elif uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = Path(temp_file.name)

    st.image(str(temp_path), caption="Uploaded CDT image", use_container_width=True)

    model = tf.keras.models.load_model(model_path)
    image = load_image(temp_path, tuple(config["image_size"]))
    probability = float(model.predict(image, verbose=0).ravel()[0])
    label = "impaired" if probability >= 0.5 else "normal"

    st.metric("Prediction", label)
    st.metric("Impaired probability", f"{probability:.2%}")
