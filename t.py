import os
import gc
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)

# ==========================================================
# CONFIGURATION
# ==========================================================
MODEL_NAME = "microsoft/deberta-v3-large"  # always large
USE_8BIT = True                            # always use 8-bit quantization
USE_4BIT = False                           # set True if memory still low
EPOCHS = 12
LR = 2e-5
BATCH = 1
ACCUM = 8
FOLDS = 5
SEED = 42
MAX_LEN = 512

# ==========================================================
# SEED FIX
# ==========================================================
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# ==========================================================
# LOAD DATA
# ==========================================================
train_df = pd.read_csv("train.csv")  # expects columns: text,label
texts = train_df["text"].astype(str).tolist()
labels = train_df["label"].astype(int).tolist()
num_labels = len(set(labels))

# ==========================================================
# TOKENIZER
# ==========================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.model_max_length = MAX_LEN

# ==========================================================
# QUANTIZATION CONFIG
# ==========================================================
quant_config = BitsAndBytesConfig(
    load_in_8bit=USE_8BIT,
    load_in_4bit=USE_4BIT,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# ==========================================================
# DATASET
# ==========================================================
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# ==========================================================
# TRAINER
# ==========================================================
class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        cw = self.class_weights.to(logits.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=cw)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ==========================================================
# METRICS
# ==========================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"macro_f1": macro_f1, "accuracy": acc}

# ==========================================================
# K-FOLD CROSS VALIDATION
# ==========================================================
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
fold_f1s, fold_accs = [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
    print(f"\n===== FOLD {fold} =====")

    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]

    train_enc = tokenizer(train_texts, truncation=True, padding=True)
    val_enc = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = TextDataset(train_enc, train_labels)
    val_dataset = TextDataset(val_enc, val_labels)

    cw = torch.tensor(
        compute_class_weight("balanced", classes=np.arange(num_labels), y=train_labels),
        dtype=torch.float
    )

    # ======================================================
    # LOAD MODEL
    # ======================================================
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # ======================================================
    # TRAINING ARGS
    # ======================================================
    args = TrainingArguments(
        output_dir=f"./fold_{fold}_results",
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        gradient_accumulation_steps=ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        logging_dir=f"./logs_fold_{fold}",
        report_to="none",
        fp16=True,
        dataloader_pin_memory=True,
    )

    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=cw,
        tokenizer=tokenizer,
    )

    # ======================================================
    # TRAIN
    # ======================================================
    trainer.train()

    # ======================================================
    # EVAL
    # ======================================================
    metrics = trainer.evaluate()
    print(f"Fold {fold} results:", metrics)
    fold_f1s.append(metrics["eval_macro_f1"])
    fold_accs.append(metrics["eval_accuracy"])

    # Free memory
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

# ==========================================================
# FINAL RESULTS
# ==========================================================
print("\n========== FINAL CROSS-VALIDATION RESULTS ==========")
print(f"Macro F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
print(f"Accuracy: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
