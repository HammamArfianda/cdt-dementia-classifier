# CDT Dementia Classifier

TensorFlow computer vision project for classifying Clock Drawing Test (CDT) images.
Clock Drawing Test (CDT) image classification model using TensorFlow/Keras.

This project starts with binary classification:

- `normal`: no dementia
- `impaired`: light, medium, or heavy impairment

The dataset is private and should not be committed to GitHub. Keep real images outside the repository or under `data/raw/`, which is ignored by Git.

## Project Structure

```text
configs/              Training configuration
data/                 Dataset documentation and example label format
notebooks/            Exploration and experiment notebooks
src/cdt_classifier/   Reusable training, evaluation, and prediction code
app/                  Future Streamlit demo app
reports/              Generated metrics and figures
models/               Trained model files
```

## Expected Dataset Layout

Use one folder per class:

```text
D:/Projects/CDT Project/data/Dataset ISPO/
  Dataset Tidak Ada Demensia/
  Dataset Demensia Ringan/
  Dataset Demensia Sedang/
  Dataset Demensia Berat/
```

The training code maps `Dataset Tidak Ada Demensia` to `normal` and maps the three dementia folders to `impaired`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train

Edit `configs/config.yaml` so `dataset_dir` points to your private dataset folder.

```powershell
python -m src.cdt_classifier.train --config configs/config.yaml
```

## Evaluate

```powershell
python -m src.cdt_classifier.evaluate --config configs/config.yaml
```

## Predict One Image

Save a test drawing outside the tracked project files, or put it under `local_tests/`, which is ignored by Git.

```powershell
python -m src.cdt_classifier.predict "local_tests/my_clock.jpg" --config configs/config.yaml
```

## Demo App

After training a model locally, launch the upload demo:

```powershell
streamlit run app/streamlit_app.py
```

Then open the local URL that Streamlit prints and upload a clock drawing image.

## Medical Disclaimer

This project is for learning and research demonstration only. It is not a medical device and must not be used to diagnose dementia.
