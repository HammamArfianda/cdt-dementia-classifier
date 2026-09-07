# CDT Dementia Classifier

TensorFlow computer vision project for classifying Clock Drawing Test (CDT) images.

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

## Medical Disclaimer

This project is for learning and research demonstration only. It is not a medical device and must not be used to diagnose dementia.
