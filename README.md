# CropVision — AI-Powered Crop Disease Detection

CropVision is a full-stack crop-leaf analysis application. The browser sends an uploaded image to a Flask API; the API performs OpenCV preprocessing and TensorFlow inference against a real, locally trained PlantVillage CNN. There are no random or mock predictions.

## Structure

```text
backend/
  app/                 Flask factory, blueprints, inference services, disease data
  models/              generated model.keras and labels.json (not committed)
  evaluation/          generated charts, reports, and metrics
  train.py             PlantVillage training and evaluation entry point
frontend/              Vite + React + Tailwind interface
render.yaml            Render API blueprint
```

## Prerequisites

- Python 3.10–3.12
- Node.js 20+
- A PlantVillage dataset organized as `dataset/Class_Name/image.jpg`.

Download PlantVillage from its official source or a licensed copy, then preserve its class-directory structure. Labels are discovered directly from those folders during training.

## Train the model

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py /absolute/path/to/PlantVillage --epochs 30 --batch-size 32
```

The command produces `backend/models/model.keras`, `backend/models/labels.json`, and evaluation output in `backend/evaluation/`:

- accuracy and loss plots
- weighted precision, recall, and F1 score
- full classification report
- confusion matrix

The generated `labels.json` is the single source of truth for inference classes; it is never manually maintained.

## Fast start with the bundled pretrained model

For immediate local inference, this workspace can use the downloaded `backend/models/model.keras` pretrained MobileNetV2 classifier. It is a genuine TensorFlow/Keras model trained on the PlantVillage augmented dataset and has a matching 38-class `labels.json`. No dataset download or local training is required for this route. The model expects a clear single-leaf image at inference time. Render downloads the same MIT-licensed model during its build, so it is not committed to Git.

To reproduce or replace it with your own model, use the training instructions above. Model artifacts are ignored by Git because of their size; deploy them through a release artifact or secure model store.

## Run locally

In one terminal:

```bash
cd backend
source .venv/bin/activate
cp .env.example .env
python run.py
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL (normally `http://localhost:5173`). The app requires the trained model artifacts to return a prediction. Until training completes, `/health` reports `model_unavailable` and `/predict` responds with a clear `503`—it never invents a diagnosis.

## API

- `GET /` — service status
- `GET` or `POST /health` — model readiness
- `POST /predict` — multipart form field `image` (`jpeg`, `png`, or `webp`)

Example response:

```json
{
  "disease": "Tomato — Early blight",
  "confidence": 98.91,
  "description": "...",
  "symptoms": [],
  "causes": [],
  "treatment": [],
  "prevention": []
}
```

## Deployment

Deploy the repository to Render using `render.yaml` (change the build command to `pip install -r requirements-production.txt` if using Gunicorn). Supply the generated model and labels as a persistent disk, release artifact, or a secure model store and set `MODEL_PATH` / `LABELS_PATH`. Set `CORS_ORIGINS` to your Vercel domain.

Deploy `frontend/` as a Vercel project. Set `VITE_API_URL` to the deployed API origin. The included rewrite enables React routes on refresh.

## Disease information

`backend/app/data/disease_info.json` contains curated information for selected high-frequency labels. Every other trained PlantVillage label is supported through the automatic label metadata fallback, so new dataset classes cannot cause a failed or fabricated prediction. Expand the JSON with region-specific, reviewed guidance before production use.

## Notes on responsible use

This is a visual screening tool, not a replacement for local extension experts, laboratory testing, or pesticide labels. Use local agronomic guidance for treatment decisions.
