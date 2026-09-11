# AI-Powered Crop Disease Detection Using Deep Learning for Sustainable Agriculture

CropVision is a complete, production-ready full-stack college project that classifies crop-leaf images using a real trained TensorFlow/Keras CNN with 38 plant disease and health classes. The model is deployed via a Flask API with a responsive React interface. It provides real prediction confidence, unknown-image rejection, and practical disease guidance. All evaluation metrics and graphs are generated from actual test data.

## Problem and objective

Visual leaf diseases can be difficult to identify early, and misidentification can lead to ineffective or harmful interventions. This project provides a browser-based screening tool supporting 14 crops and 38 disease/health classes from the PlantVillage dataset. It implements confidence-based rejection to avoid misleading users with uncertain predictions. It is an educational and research screening tool, not a substitute for laboratory testing, professional agricultural expertise, local extension services, or approved pesticide labels.

## Dataset and selected classes

The source is the original [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset), using the complete `raw/color` directory. The dataset contains 54,305 color leaf images across 14 crops and 38 disease/healthy classes.

**Supported Crops:** Apple, Blueberry, Cherry (sour), Corn (maize), Grape, Orange, Peach, Pepper (bell), Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

**Classes per crop:** Healthy + 2–7 disease classes per crop = 38 total classes

The pipeline uses stratified splitting with seed 42: approximately 70% training, 15% validation, 15% test. Test set: 8,143 images. Training includes augmentation (rotation, flip, zoom, translation, brightness); validation and test sets use no augmentation. Preprocessing: 224×224 RGB, pixel normalization (0–1).

## Model and preprocessing

**Architecture:**

- Input: RGB images, 224×224 pixels
- Normalization: pixel values divided by 255.0 (range 0–1)
- Convolutional blocks: Conv2D (32–128 filters) → ReLU → MaxPooling (2×2)
- Regularization: Batch Normalization, Dropout (0.2–0.5)
- Global Average Pooling → Dense (128 units) → Softmax (38 outputs, one per class)

**Training configuration:**

- Optimizer: Adam with initial learning rate 0.001
- Loss: Sparse Categorical Cross-Entropy
- Batch size: 32
- Maximum epochs: 25 (with early stopping)
- Callbacks: Early Stopping (patience 3), ReduceLROnPlateau, ModelCheckpoint

**Unknown-image handling:**

- Confidence-based rejection: predictions rejected if softmax probability < per-class threshold
- Entropy-based rejection: normalized Shannon entropy must be ≤ calibrated threshold
- Calibration: thresholds computed from validation set to balance precision and recall
- Result: Unknown predictions display "Cannot classify" rather than forcing incorrect class

## Project structure

```
backend/
  run.py                                Flask app entry point
  train.py                              Dataset pipeline, model training, evaluation
  evaluate.py                           Test-set evaluation and metric generation
  calibrate_rejection.py                Confidence/entropy threshold calibration
  models/
    best_model.keras                    Trained 38-class CNN (TensorFlow format)
    class_names.json                    Class label mapping
  artifacts/
    model_metadata.json                 Model config, class count, image size
    preprocessing_config.json           Normalization constants
    rejection_calibration.json          Confidence/entropy thresholds
    class_indices.json                  Class name to index mapping
    split_manifest.csv                  Train/val/test split record
    training_history.json               Epoch-by-epoch training metrics
  app/
    __init__.py                         Flask factory and config
    routes/
      predict.py                        POST /predict inference endpoint
      results.py                        GET /results evaluation dashboard data
      health.py                         GET /health status check
    services/
      predictor.py                      Model loading, preprocessing, inference
      disease_info.py                   Disease/symptom/treatment guidance
    data/
      disease_info.json                 Crop disease information database
  results/
    metrics.json                        Test accuracy, precision, recall, F1
    classification_report.json          Per-class metrics (JSON)
    classification_report.csv           Per-class metrics (CSV)
    confusion_matrix.csv                Confusion matrix (CSV)
    confusion_matrix_normalized.csv     Normalized confusion matrix
    graphs/
      training_accuracy.png             Training vs validation accuracy curve
      training_loss.png                 Training vs validation loss curve
      confusion_matrix.png              Graphical confusion matrix heatmap
      class_distribution.png            Test set class distribution
      per_class_metrics.png             Per-class precision, recall, F1

frontend/
  src/
    main.jsx                            React entry point
    App.jsx                             Routing and layout
    pages/
      Home.jsx                          Project overview and instructions
      Predict.jsx                       Image upload and prediction form
      Result.jsx                        Prediction results with disease info
      Dashboard.jsx                     Model evaluation metrics dashboard
      About.jsx                         Project background and disclaimer
    components/
      Layout.jsx                        Header, navigation, footer
      Uploader.jsx                      Image upload widget
    lib/
      api.js                            Axios API client
      history.js                        Local prediction history
  package.json                          Dependencies (React, Vite, Tailwind, Framer Motion)
  vite.config.js                        Vite bundler config
  tailwind.config.js                    CSS framework config

dataset/
  PlantVillage-Dataset/                 Original dataset (54K images, not versioned)
    raw/color/                          Training source images

.gitignore                              Excludes models, data, node_modules
render.yaml                             Deployment config for Render
```

## Train and evaluate

**Prerequisites:**

1. Download the [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
2. Extract to `dataset/PlantVillage-Dataset/`

**Training pipeline:**

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Discover all classes and create the 70/15/15 stratified split
python dataset_pipeline.py ../dataset/PlantVillage-Dataset/raw/color --output artifacts --seed 42

# Train, evaluate, and calibrate rejection as separate reproducible steps
python train.py --artifacts artifacts --models model --epochs 25 --batch-size 32 --seed 42
python evaluate.py --artifacts artifacts --models model --results results --batch-size 32
python calibrate_rejection.py --artifacts artifacts --models model --batch-size 32
```

This pipeline:

1. **Discovers all classes** from directory structure (38 classes found)
2. **Validates images** and removes corrupted files
3. **Creates stratified split** (70/15/15) with seed 42
4. **Trains the CNN** with callbacks and real-time metrics
5. **Evaluates on test set** and generates all metrics
6. **Calibrates rejection thresholds** using validation set
7. **Saves all artifacts:**

- `model/crop_disease_model.keras` — trained weights
- `model/class_names.json` — complete dynamic 38-class mapping
- `artifacts/model_metadata.json` — architecture info
- `artifacts/rejection_calibration.json` — confidence thresholds
- `results/metrics.json` — test evaluation results
- `results/graphs/` — 8 visualization plots
- `results/classification_report.json` — per-class metrics

## Train on Google Colab GPU

For faster training, use a Colab runtime with a T4 GPU. The dataset must be
available inside the same Colab runtime where the manifest is generated because
the manifest stores absolute image paths.

1. In Colab, select **Runtime → Change runtime type → T4 GPU**.
2. Upload the project to GitHub, then run:

```python
!git clone YOUR_REPOSITORY_URL /content/ai-crop-disease-detection
%cd /content/ai-crop-disease-detection/backend
!pip install -r requirements.txt
```

3. Place the dataset at:

```text
/content/ai-crop-disease-detection/dataset/PlantVillage-Dataset/raw/color/
```

You can upload a zip of your existing local `dataset/PlantVillage-Dataset`
directory to Google Drive, mount Drive, and extract it into that location:

```python
from google.colab import drive
drive.mount('/content/drive')
!unzip -q "/content/drive/MyDrive/PlantVillage-Dataset.zip" -d /content/ai-crop-disease-detection/dataset/
```

4. Confirm the GPU and run the complete all-class pipeline:

```python
!nvidia-smi
!python dataset_pipeline.py ../dataset/PlantVillage-Dataset/raw/color --output artifacts --seed 42
!python train.py --artifacts artifacts --models model --epochs 25 --batch-size 32 --seed 42
!python evaluate.py --artifacts artifacts --models model --results results --batch-size 32
!python calibrate_rejection.py --artifacts artifacts --models model --batch-size 32
```

The preparation command prints the dataset root, all detected classes, image
counts, and total image count before training. It discovers every valid class
folder and does not filter by crop or select a fixed number of classes.

5. Download the trained outputs from Colab:

```python
from google.colab import files
!zip -qr /content/crop-disease-results.zip model artifacts results
files.download('/content/crop-disease-results.zip')
```

## Actual evaluation results

**Tested on held-out 8,143 images (15% of full PlantVillage dataset):**

| Metric             |      Value |
| ------------------ | ---------: |
| **Test Accuracy**  | **95.98%** |
| Macro Precision    |     94.16% |
| Macro Recall       |     96.15% |
| Macro F1-score     |     94.87% |
| Weighted Precision |     95.96% |
| Weighted Recall    |     95.98% |
| Weighted F1-score  |     95.91% |

**Per-class performance highlights:**

- Perfect predictions (100% F1): Apple Cedar apple rust, Corn healthy, Cherry Powdery mildew, Corn Common rust
- Lowest performance: Corn Cercospora leaf spot (86.2% F1) — inherently difficult due to visual similarity to other diseases
- Average per-class support: ~214 images (range 41–245)

**Generated visualizations** (in `results/graphs/`):

1. Training accuracy curve (training vs validation)
2. Training loss curve (training vs validation)
3. Confusion matrix heatmap (normalized)
4. Class distribution in test set
5. Per-class precision/recall/F1 comparison
6. ROC curves (per-class)
7. Additional diagnostic plots

Full details: `backend/results/classification_report.json` and `backend/results/metrics.json`

## Run the application

**Backend (Flask API):**

```bash
cd backend
source .venv/bin/activate
python run.py
```

API runs on http://127.0.0.1:5001

**Frontend (React + Vite):**

```bash
cd frontend
npm install --legacy-peer-deps
npm run dev
```

Frontend runs on http://localhost:5173

**API Endpoints:**

| Endpoint                     | Method | Description                   |
| ---------------------------- | ------ | ----------------------------- |
| `/`                          | GET    | Service status                |
| `/predict`                   | POST   | Upload image, get prediction  |
| `/results`                   | GET    | Evaluation metrics and graphs |
| `/results/graphs/<filename>` | GET    | Serve PNG graph               |
| `/health`                    | GET    | Model readiness check         |

**Example prediction request:**

```bash
curl -F "image=@leaf.jpg" http://127.0.0.1:5001/predict
```

**Prediction response (classified):**

```json
{
  "success": true,
  "status": "classified",
  "prediction": "Tomato — Late blight",
  "label": "Tomato___Late_blight",
  "crop": "Tomato",
  "disease": "Tomato — Late blight",
  "confidence": 98.5,
  "normalized_entropy": 0.0031,
  "symptoms": ["Water-soaked spots on leaves and stems...", "..."],
  "causes": ["Phytophthora infestans fungus...", "..."],
  "treatment": ["Apply fungicides...", "..."],
  "prevention": ["Use resistant varieties...", "..."],
  "top_predictions": [
    { "disease": "Tomato — Late blight", "confidence": 98.5 },
    { "disease": "Tomato — Early blight", "confidence": 1.2 },
    { "disease": "Tomato — Septoria leaf spot", "confidence": 0.3 }
  ],
  "message": "Image classified using the trained PlantVillage model."
}
```

**Prediction response (unknown/low-confidence):**

```json
{
  "success": true,
  "status": "unknown",
  "prediction": null,
  "label": null,
  "confidence": 62.3,
  "normalized_entropy": 0.5234,
  "message": "Unknown / low confidence: this image cannot be classified reliably by this model."
}
```

**Frontend features:**

- Image upload with preview
- Real-time prediction from Flask API
- Confidence score display
- Disease information: symptoms, causes, treatment, prevention
- Top-3 predictions
- Results dashboard: accuracy, F1, confusion matrix, class-wise metrics, training curves
- Responsive design (mobile-friendly, dark mode support)
- Local prediction history

## Testing

**Backend API tests:**

```bash
# Health check
curl http://127.0.0.1:5001/

# Get evaluation results
curl http://127.0.0.1:5001/results

# Predict with an image
curl -F "image=@test_leaf.jpg" http://127.0.0.1:5001/predict

# Download a graph
curl http://127.0.0.1:5001/results/graphs/confusion_matrix.png -o cm.png
```

**Frontend tests (via browser):**

1. Visit http://localhost:5173/
2. Go to "Analyze a crop leaf"
3. Upload a crop-leaf image (JPG, PNG, or WEBP)
4. Click "Get disease analysis"
5. Review prediction, disease information, and top predictions
6. Check "View results dashboard" for model evaluation metrics

**Full-pipeline validation:**

- Dataset: 54K images from 14 crops, 38 classes ✅
- Training: Stratified split, augmentation, callbacks ✅
- Evaluation: Real test-set metrics (95.98% accuracy) ✅
- Rejection mechanism: Low-confidence/unknown images properly rejected ✅
- API: Prediction, results, health endpoints working ✅
- Frontend: Upload, predict, results, dashboard pages functional ✅
- End-to-end: Upload image through UI → API processes → results displayed ✅

## Limitations and disclaimers

- **AI-assisted screening only**: This model provides a first-pass disease screening signal. It is not a substitute for professional agricultural diagnosis, expert inspection, or laboratory testing.
- **PlantVillage data bias**: The training dataset contains controlled, well-lit leaf images. Field images with complex backgrounds, partial leaves, poor lighting, or multiple disease types may yield lower accuracy.
- **No treatment validation**: Disease information is provided for educational context. Treatment recommendations should be verified with local agricultural extension services and approved for your region and crop type.
- **Unknown prediction handling**: ~5% of test images trigger the rejection mechanism (low confidence or high entropy), preventing unreliable classifications. This is intentional and beneficial.
- **Out-of-distribution rejection**: Images of unsupported crops, severely damaged leaves, or non-leaf objects should trigger "unknown" status. If not, report via issue.
- **Class imbalance**: Some classes (e.g., Cedar apple rust) have fewer training images. Precision and recall vary by class; see detailed metrics for class-specific performance.
- **No treatment-effectiveness guarantee**: The model does not track treatment outcomes or seasonal variations in disease progression.

## Disclaimer

Users are responsible for verifying all guidance with local agricultural professionals before making treatment or purchasing decisions. Developers and researchers disclaim liability for yield losses, crop damage, pesticide misuse, or other agricultural or business losses resulting from use of this tool.

## Future improvements

- Integrate field-collected images for robustness validation
- Add confidence calibration for deployment regions
- Expand dataset with underrepresented crops and disease variants
- Implement image preprocessing feedback (suggest better framing, lighting)
- Add disease progression timeline and seasonal guidance
- Export prediction history as CSV/PDF reports
- Multi-image batch prediction for farm-scale scouting
- Integration with farm-management systems (APIs, webhooks)
- Model distillation for mobile/edge deployment
- Uncertainty quantification using Bayesian CNN or ensembles
