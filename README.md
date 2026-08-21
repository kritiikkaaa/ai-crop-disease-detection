# AI-Powered Crop Disease Detection Using Deep Learning for Sustainable Agriculture

CropVision is a full-stack college project that classifies a tomato-leaf image with a real TensorFlow/Keras CNN, then returns the model's prediction confidence and practical disease guidance through a Flask API and React interface. It follows the methodology in the accompanying research paper; reported evaluation values are generated only after training.

## Problem and objective

Visual leaf diseases can be difficult to identify early. This project provides a browser interface for screening six verified PlantVillage tomato-leaf categories. It is an educational screening tool, not a substitute for laboratory testing, local extension services, or approved pesticide labels.

## Dataset and selected classes

The source is the original [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset), using `raw/color`. The original data remains unmodified in `dataset/PlantVillage-Dataset/`.

| Class | Images |
| --- | ---: |
| Tomato Early blight | 1,000 |
| Tomato Late blight | 1,909 |
| Tomato Septoria leaf spot | 1,771 |
| Tomato Spider mites (two-spotted) | 1,676 |
| Tomato Target Spot | 1,404 |
| Tomato healthy | 1,591 |

The experiment uses 9,351 images split reproducibly with seed 42: 6,545 training, 1,403 validation, and 1,403 test images. Training augmentation is limited to horizontal flip, small rotation, zoom, and translation. Validation, test, and API inputs use no augmentation.

## Model and preprocessing

- RGB image conversion and 224×224 resize
- Pixel normalization from 0–255 to 0–1
- CNN blocks of Conv2D, ReLU, MaxPooling, and Dropout, followed by global average pooling, Dense, and a six-unit Softmax output
- Adam optimizer, 0.001 initial learning rate, categorical cross-entropy, batch size 32, maximum 25 epochs
- Early stopping, learning-rate reduction, and best-model checkpointing

## Project structure

```text
dataset/PlantVillage-Dataset/raw/color/  original RGB dataset (not committed)
backend/
  train.py                              real split, training, and evaluation entry point
  models/crop_disease_model.keras       generated best model (not committed)
  models/labels.json                    generated class order
  evaluation/                           generated metrics, report, plots, confusion matrix
  app/                                  Flask routes and inference services
frontend/                               React + Vite interface
```

## Train and evaluate

```bash
cd backend
source .venv/bin/activate
python train.py ../dataset/PlantVillage-Dataset/raw/color --epochs 25 --batch-size 32
```

This creates the model and real evaluation artifacts in `backend/evaluation/`:

- `dataset_distribution.csv` and `.json`
- `classification_report.txt`
- `metrics.json` (test accuracy, weighted precision, recall, and F1)
- `confusion_matrix.png`
- `training_accuracy.png` and `training_loss.png`

## Actual evaluation results

The completed Colab run achieved the following results on the held-out 1,403-image test set:

| Metric | Result |
| --- | ---: |
| Accuracy | 88.17% |
| Weighted precision | 88.56% |
| Weighted recall | 88.17% |
| Weighted F1-score | 88.05% |

See `backend/evaluation/metrics.json`, the classification report, and confusion matrix for the full generated output. Future training runs also save the exact training and validation history in `training_history.json`.

## Run the application

```bash
cd backend
source .venv/bin/activate
cp .env.example .env
python run.py
```

```bash
cd frontend
npm run dev
```

The React client sends `multipart/form-data` to `POST /predict` with an `image` field. The Flask API validates JPEG, PNG, and WEBP uploads, preprocesses them with OpenCV, runs the saved CNN, and returns the actual highest-probability class, confidence, symptoms, causes, treatment, and prevention guidance. `GET /health` reports whether the trained model is ready.

## Testing

After training, start the backend and submit a real image from the test split through the interface or API. Also verify `/health`, unsupported-file errors, corrupted-image errors, oversized-image errors, and the frontend reset/upload-again flow. The training evaluation script supplies the test metrics and confusion matrix; do not substitute assumed paper values.

## Future improvements

Use field-image datasets, collect local expert labels, add calibration and rejection thresholds, and validate treatment guidance for the deployment region.
