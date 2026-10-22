# AI-Assisted Diabetic Retinopathy Screening Using Digital Image Processing

An educational project proposal and prototype combining visible classical image processing with binary transfer-learning classification. **This is not a medical diagnostic tool and does not replace professional ophthalmological assessment.**

## What the application does

For an uploaded JPG, JPEG, or PNG image, the Streamlit app displays the original image, the fundus ROI after mild Gaussian denoising and CLAHE enhancement, a classical green-channel vessel-candidate mask, and simple image statistics. If a trained local model is available, it also reports a Normal/Diabetic Retinopathy prediction, the model's class probability, and an approximate Grad-CAM overlay.

The mask and handcrafted statistics are DIP demonstrations; they are not passed into the initial classifier. The classifier receives a cropped, denoised, CLAHE-enhanced RGB image. The app does not add uploads to the training dataset.

## Dataset

Use the labeled data from **APTOS 2019 Blindness Detection**. The current project data folder supports the archive you extracted, which provides pre-separated labeled train, validation, and test splits:

```text
data/
├── train_1.csv
├── valid.csv
├── test.csv
├── train_images/train_images/
├── val_images/val_images/
└── test_images/test_images/
```

Each CSV is expected to have `id_code` and `diagnosis` columns. Grade 0 maps to Normal; grades 1–4 map to Diabetic Retinopathy. The project preserves the archive's supplied partitions and verifies that image IDs do not overlap. It does not use the unlabeled competition test images. If instead using the original combined labeled CSV, put it and its images under `data/aptos/`; the project can create a stratified 70/15/15 split. Since patient identifiers are not included in the commonly distributed labels, an image-level split may not prevent related photographs crossing partitions; report this limitation.

## Setup

Python 3.13 is verified for this project. Create an environment and install packages:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `uv` is installed, the equivalent setup is `uv venv --python 3.13 .venv` followed by `uv pip install --python .venv\Scripts\python.exe -r requirements.txt`.

## Workflow

Run commands from the project root.

1. Create and inspect split metadata from the supplied CSVs and image folders:

   ```powershell
   python -m scripts.prepare_data
   ```

   The saved `data/splits.csv` records each image, binary label, original grade, and partition. Inspect class counts before training. For this archive, the pre-provided 80/10/10 partitions are preserved rather than re-splitting.

2. Train EfficientNetB0 with an ImageNet-pretrained frozen backbone, class weights, early stopping, and validation-loss checkpointing:

   ```powershell
   python -m scripts.train --epochs 20 --batch-size 16
   ```

   Optional fine-tuning of the top backbone layers:

   ```powershell
   python -m scripts.train --epochs 20 --fine-tune
   ```

   Training writes model/history outputs and training/validation curves to `models/`. Training-only horizontal flips and small rotations augment the data. TensorFlow will attempt to obtain the pretrained ImageNet weights if they are not already cached; if network access is unavailable, prepare the weights separately or adapt the configuration to use `weights=None` and document that choice.

3. Evaluate the selected model on the held-out test partition only after model development is complete:

   ```powershell
   python -m scripts.evaluate
   ```

   It writes accuracy, precision, recall, F1, confusion matrix, and test predictions to `models/`. Do not repeatedly tune against the test results; use validation data for model decisions. If a preliminary evaluation is run before training is complete, any subsequent training/model selection requires a new, untouched test set for an unbiased final result.

4. Launch the app:

   ```powershell
   streamlit run app.py
   ```

   Upload a fundus image and click **Run prediction**. Without `models/dr_screening.keras`, the DIP views still work and the app explains how to train a model.

## Project structure

```text
app.py
src/
  preprocessing.py
  data.py
  model.py
  gradcam.py
scripts/
  prepare_data.py
  train.py
  evaluate.py
tests/
data/                 # user-provided dataset; not committed
models/               # generated model/metrics; not committed
```

## Processing details

1. Decode and validate image dimensions/channels.
2. Crop the non-black fundus region when detectable.
3. Apply a mild Gaussian filter.
4. Convert to LAB, apply CLAHE to lightness, and convert back to RGB.
5. Resize to 224 x 224 for the initial model.
6. For the DIP-only vessel output, enhance the green channel, estimate local background with morphological closing, threshold vessel candidates with Otsu, then clean with opening/closing.
7. Calculate channel means, grayscale intensity standard deviation, and vessel-candidate fraction for display.

The vessel mask is a simple classical demonstration and can be inaccurate. File checks cannot guarantee that an uploaded image is a retinal photograph. Model probabilities may be uncalibrated; Grad-CAM is only an approximate explanation and does not confirm a lesion.

## Evaluation and responsible use

Report accuracy, precision, recall, F1-score, confusion matrix, and training/validation accuracy and loss. A target such as 80% accuracy is aspirational only; report actual test performance, even if lower. Because classes are imbalanced, accuracy alone is insufficient. All model selection and threshold decisions must use training/validation data, never the held-out test set.

The software is an academic prototype. It is not clinically validated, must not be used to make health decisions, and does not replace assessment by a qualified eye-care professional.

## Tests

```powershell
pytest
```
