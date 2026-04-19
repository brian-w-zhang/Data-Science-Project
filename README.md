# Natural Disaster Tweets Classification Using Multimodal Data

Binary classification of social media posts as disaster-informative or not, using tweet text and images. Built with DistilBERT (text), ResNet50 (vision), and a late-fusion MLP head.

**Authors:** Sondre Kristiansen, Rasmus Tuokko, Brian Zhang, Sabrina Wang

**Repository:** [github.com/brian-w-zhang/Data-Science-Project](https://github.com/brian-w-zhang/Data-Science-Project)

---

## Table of Contents

- [Project Structure](#project-structure)
- [Datasets](#datasets)
  - [Shared Google Drive](#shared-google-drive)
- [Setup](#setup)
- [Training](#training)
- [Inference & UI](#inference--ui)
- [Reproducing Results Without Retraining](#reproducing-results-without-retraining)
- [Model Architecture](#model-architecture)

---

## Project Structure

```
.
├── app/
│   └── streamlit_app.py          # Streamlit UI
├── data/
│   ├── kaggle_text.py            # Kaggle CSV loader (uses clean_tweet)
│   ├── crisismmd.py             # CrisisMMD loaders, datasets, transforms
│   └── text_cleaner.py           # Shared RegEx preprocessing for train & inference
├── models/
│   ├── text_branch.py            # DistilBERT classifier wrapper
│   ├── vision_branch.py          # ResNet50 classifier wrapper
│   └── fusion_model.py           # Late-fusion network
├── training/
│   ├── utils.py                  # train_one_epoch, evaluate
│   ├── 00_dataloader.ipynb       # Dataset download & path setup
│   ├── 01_text_branch_train.ipynb
│   ├── 02_vision_branch_train.ipynb
│   ├── 03_fusion_layer_train.ipynb
│   └── 04_datasaver.ipynb        # Copy Colab checkpoints to Drive
├── inference/
│   └── service.py                # Inference service for the UI
├── checkpoints/                  # Saved model weights (generated at training time)
│   ├── text_branch/              # HuggingFace model directory
│   ├── vision_brain.pth          # ResNet50 state dict
│   └── fusion_brain.pth          # Fusion model state dict
├── CrisisMMD_v2.0/               # CrisisMMD dataset (downloaded separately)
├── train.csv                     # Kaggle dataset (downloaded separately)
├── requirements.txt
└── README.md
```

---

## Datasets

Two datasets are required. Neither is included in the repository and must be downloaded before training.

### Shared Google Drive

A shared folder contains **CrisisMMD v2.0**, **Kaggle** `train.csv` / `test.csv`, and **saved model checkpoints** (`checkpoints/`) so you can skip long downloads and full retraining:

[Google Drive — project assets](https://drive.google.com/drive/folders/1veiX_VP2qMTxj5m7aodgQWu3qjbmM6_x?usp=sharing)

Download what you need and place files as in [Project Structure](#project-structure): `train.csv` at the repo root, extract `CrisisMMD_v2.0/` beside it, and copy `checkpoints/` (with `text_branch/`, `vision_brain.pth`, `fusion_brain.pth`) for inference or notebook evaluation.

### Kaggle — NLP with Disaster Tweets

Used for text branch training (combined with CrisisMMD text).

Download via the notebook or manually from [Kaggle](https://www.kaggle.com/competitions/nlp-getting-started/data). Place `train.csv` in the project root.

```bash
# Or via gdown (run in 00_dataloader.ipynb):
gdown 19gyTA8S9wXPlxkyU0mZ_raJO62ny-vpo   # train.csv
gdown 1NqChJSt70NvgInNvYEyh2GTsBfpXUNuj   # test.csv
```

### CrisisMMD v2.0

Used for vision branch training, fusion training, and fusion evaluation.

```bash
gdown 1wi_kAVETQHQyGA-thBhP7VY1L2jD5quI   # CrisisMMD_v2.0.tar.gz
tar -xzf CrisisMMD_v2.0.tar.gz
```

The extracted folder `CrisisMMD_v2.0/` must be in the project root. It should contain:
```
CrisisMMD_v2.0/
├── annotations/      # .tsv annotation files (one per disaster event)
└── data_image/       # tweet images
```

---

## Setup

### Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch`, `torchvision`
- `transformers`, `datasets`
- `scikit-learn`, `pandas`, `numpy`
- `Pillow`, `tqdm`
- `streamlit`

### Running on Google Colab (recommended)

The training notebooks are designed for Colab with GPU. Each notebook clones the repository and sets up paths automatically. Mount Google Drive at the start of each notebook to persist checkpoints across sessions.

### Running locally

Clone the repository and ensure dataset paths in each notebook match your local file structure. The default paths assume datasets are in the project root:
- `train.csv` → `/path/to/project/train.csv`
- `CrisisMMD_v2.0/` → `/path/to/project/CrisisMMD_v2.0/`

---

## Training

Run the notebooks **in order**. Each stage depends on checkpoints saved by the previous one.

### 1. Dataset preparation — `00_dataloader.ipynb`

Downloads both datasets and verifies paths. Run this first on a fresh environment.

### 2. Text branch — `01_text_branch_train.ipynb`

Fine-tunes DistilBERT on a balanced combination of Kaggle and CrisisMMD text data (up to 5,000 samples per class per source, stratified 80/20 split). Applies `clean_tweet()` preprocessing to both sources before tokenisation.

Saves to: `checkpoints/text_branch/` (HuggingFace model directory)

Key hyperparameters:
| Parameter | Value |
|---|---|
| Model | `distilbert-base-uncased` |
| Optimizer | AdamW |
| Learning rate | `2e-5` |
| Batch size | 16 |
| Max epochs | 10 |
| Early stopping patience | 3 |
| Max token length | 128 |

### 3. Vision branch — `02_vision_branch_train.ipynb`

Fine-tunes ResNet50 on CrisisMMD images. Only `layer3`, `layer4`, and `fc` are unfrozen. Uses differential learning rates across layers.

Saves to: `checkpoints/vision_brain.pth`

Key hyperparameters:
| Parameter | Value |
|---|---|
| Model | ResNet50 (ImageNet pretrained) |
| Optimizer | Adam |
| Learning rates | `layer3`: `1e-5`, `layer4`: `1e-4`, `fc`: `1e-3` |
| Batch size | 32 |
| Max epochs | 15 |
| Early stopping patience | 3 |
| Scheduler | StepLR (step=3, γ=0.5) |

### 4. Fusion model — `03_fusion_layer_train.ipynb`

Loads frozen text and vision encoders from the checkpoints above. Trains only the fusion MLP head on CrisisMMD multimodal data. Uses weighted cross-entropy to handle class imbalance.

Saves to: `checkpoints/fusion_brain.pth`

Key hyperparameters:
| Parameter | Value |
|---|---|
| Optimizer | Adam (fusion head only) |
| Learning rate | `1e-3` |
| Batch size | 32 |
| Max epochs | 20 |
| Early stopping patience | 4 |
| Dropout | 0.5 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Class weights | `[4.207, 1.312]` for `[safe, disaster]` |

### 5. Save checkpoints — `04_datasaver.ipynb` (Colab only)

If training on Colab, run this to copy checkpoints from the Colab runtime to Google Drive before the session ends.

---

## Inference & UI

### Prerequisites

All three checkpoints must exist:
```
checkpoints/text_branch/
checkpoints/vision_brain.pth
checkpoints/fusion_brain.pth
```

### Launch the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

The interface accepts **at least one** of:
- Tweet text
- An image upload — `jpg`, `jpeg`, or `png`

If only text is sent, inference uses a placeholder image tensor (same as training-time fusion path). If only an image is sent, text is passed through as empty after cleaning.

The app returns the predicted label (`disaster` or `not_disaster`) with a confidence score (class names match `inference/service.py`).

---

## Reproducing Results Without Retraining

If checkpoints are already available (e.g. downloaded from a shared Drive folder):

1. Place checkpoints in the correct locations listed above.
2. Skip notebooks 01–03.
3. Open the evaluation cell at the bottom of whichever notebook's results you want to reproduce, ensure the checkpoint path variables are set correctly, and run from that cell downward.

To reproduce all reported metrics:

| Metric | Notebook | Cell |
|---|---|---|
| Text branch validation results | `01_text_branch_train.ipynb` | "Evaluation on best checkpoint" |
| Vision branch validation results | `02_vision_branch_train.ipynb` | "Evaluation on best checkpoint" |
| Fusion validation results + majority baseline | `03_fusion_layer_train.ipynb` | "Evaluation on best checkpoint" |

---

## Model Architecture

```
Tweet text ──► clean_tweet() ──► DistilBERT ──► [CLS] token [B, 768] ──┐
                                  (frozen)                               ├──► cat [B, 2816] ──► MLP ──► logits [B, 2]
Tweet image ──► transforms ──► ResNet50 backbone ──► features [B, 2048] ┘
                                (frozen, fc=Identity)

MLP head: Linear(2816→512) ──► ReLU ──► Dropout(0.5) ──► Linear(512→2)
```

**Label logic (fusion):** `label = 1` if `text_info == informative` OR `image_info == informative`, else `0`. This OR-labelling ensures that tweets with a strong signal in either modality are flagged as disaster-relevant.

**Text preprocessing:** URLs, @mentions, and non-ASCII characters (emojis, encoding artefacts) are stripped via RegEx before tokenisation, applied consistently during both training and inference.