# IndoML Datathon — Mistake Identification in AI Tutoring Responses

A machine-learning solution for the **IndoML Datathon** challenge focused on evaluating AI tutor responses in mathematics education. The task is to classify whether a tutor correctly identifies a student's mistake in a multi-turn dialogue.

---

## 📋 Task Overview

Given a tutoring conversation between a student and an AI tutor, the goal is to predict the **Mistake Identification** label for each tutor response:

| Label | Meaning |
|---|---|
| `Yes` | The tutor correctly identifies the student's mistake |
| `To some extent` | The tutor partially identifies the mistake |
| `No` | The tutor does not identify the student's mistake |

The dataset contains responses from **8 AI tutors**: `Sonnet`, `Llama318B`, `Llama31405B`, `GPT4`, `Mistral`, `Expert`, `Gemini`, and `Phi3`.

---

## 🗂️ Repository Structure

```
indoml/
├── trainset.json                        # 300 labelled conversations (train split)
├── dev_testset.json                     # 41 unlabelled conversations (dev/validation split)
├── testset.json                         # 150 unlabelled conversations (final test split)
│
├── script.py                            # Baseline: RoBERTa-base with 5-fold CV
├── final_script_mistake_id_optimized.py # Optimised: DeBERTa-v3-large with class weights & early stopping
├── train.py                             # Enhanced: DeBERTa-v3-base/large, oversampling, weighted ensemble (1-fold)
├── t.py                                 # Full 5-fold CV variant of train.py
├── generate_submission.py               # Inference: load saved model → produce submission CSV
│
├── j.c                                  # MPI point-to-point messaging demo (C)
└── l.cpp                                # Floyd-Warshall shortest paths — serial & MPI parallel (C++)
```

---

## 📦 Dataset Format

Each JSON file contains a list of conversation objects:

```json
{
  "conversation_id": "221-362eb11a-...",
  "conversation_history": "<multi-turn dialogue>",
  "tutor_responses": {
    "GPT4": {
      "response": "<tutor reply>",
      "annotation": {
        "Mistake_Identification": "Yes",
        "Providing_Guidance": "Yes"
      }
    },
    ...
  }
}
```

- **`trainset.json`** — 300 conversations, all responses annotated.
- **`dev_testset.json`** — 41 conversations used during development; no annotations.
- **`testset.json`** — 150 conversations for the final competition submission; no annotations.

---

## 🤖 Model Variants

### 1. `script.py` — Baseline (RoBERTa-base)
- Model: `roberta-base`
- 5-fold stratified cross-validation
- Input: `conversation_history [SEP] response`
- Outputs `submission_cv.csv`

### 2. `final_script_mistake_id_optimized.py` — Optimised Single-Model
- Model: `microsoft/deberta-v3-large`
- Stratified 90/10 train–dev split
- Class-weighted cross-entropy loss
- Early stopping (patience = 13 steps)
- Input includes tutor name: `tutor: <name> [SEP] dialogue: <history> [SEP] response: <reply>`
- Saves best checkpoint to `./results_mistake_id_base_finalll`

### 3. `train.py` — Enhanced Training (Weighted Ensemble)
- Model: `microsoft/deberta-v3-base` (or `-large` via `USE_LARGE=1`)
- Minority-class oversampling
- Label smoothing or focal loss (configurable)
- Cosine LR scheduler with warm-up
- Weighted fold ensemble by validation macro-F1
- Outputs `submission_cv_ensemble_weighted.csv`

### 4. `t.py` — Full 5-Fold CV Variant
- Same architecture as `train.py` but runs all 5 folds
- Useful for more robust evaluation and ensemble

### 5. `generate_submission.py` — Inference
- Loads a checkpoint saved by `final_script_mistake_id_optimized.py`
- Runs inference on `testset.json`
- Outputs `submission.csv` ready for competition upload

---

## ⚙️ Configuration (Environment Variables)

`train.py` and `t.py` are configurable without editing code:

| Variable | Default | Description |
|---|---|---|
| `USE_LARGE` | `0` | Use `deberta-v3-large` instead of `-base` |
| `USE_8BIT` | `0` | Enable 8-bit quantisation via `bitsandbytes` |
| `USE_FP16` | `0` | Enable FP16 mixed precision |
| `USE_FOCAL` | `0` | Use focal loss instead of label-smoothed CE |
| `OVERSAMPLE` | `1` | Oversample minority classes |
| `NUM_FOLDS` | `5` | Number of CV folds |
| `EPOCHS` | `12` | Max training epochs |
| `LR` | `2e-5` | Learning rate |
| `BATCH` | `1` | Per-device batch size |
| `ACCUM` | `8` | Gradient accumulation steps (effective batch = 8) |
| `LABEL_SMOOTHING` | `0.1` | Label smoothing factor |
| `EARLY_STOP_PATIENCE` | `5` | Early stopping patience (epochs) |
| `SEED` | `42` | Random seed |

---

## 🚀 Quick Start

### Install dependencies
```bash
pip install torch transformers datasets scikit-learn pandas numpy
# optional: pip install bitsandbytes  # for 8-bit quantisation
```

### Train (optimised single-model)
```bash
python final_script_mistake_id_optimized.py
```

### Train (full CV ensemble)
```bash
python t.py
```

### Generate submission CSV
```bash
python generate_submission.py
```

### Baseline (RoBERTa)
```bash
python script.py
```

---

## 📊 Evaluation Metric

The competition uses **Macro F1-score** across the three classes (`Yes`, `To some extent`, `No`). All training scripts report macro-F1 and accuracy after each epoch/fold.

---

## 🔬 Bonus: Parallel Computing Experiments

Two additional files demonstrate parallel computing concepts (unrelated to the datathon):

- **`j.c`** — MPI point-to-point messaging in C, using the Chandrayaan-3 mission as a themed example (requires `mpicc`).
- **`l.cpp`** — Floyd-Warshall all-pairs shortest path in C++, with both serial and MPI-parallel implementations and execution-time benchmarking (compile with `g++` or `mpicxx -DMPI_ENABLED`).

---

## 📝 Output Files

| File | Generated by | Description |
|---|---|---|
| `submission.csv` | `generate_submission.py` | Final predictions for competition upload |
| `submission_cv.csv` | `script.py` | Baseline CV predictions |
| `submission_cv_ensemble_weighted.csv` | `train.py` / `t.py` | Ensemble CV predictions |

