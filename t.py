import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import gc
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from transformers.data.data_collator import default_data_collator

USE_LARGE = os.environ.get("USE_LARGE", "1") == "1"
USE_8BIT = os.environ.get("USE_8BIT", "0") == "1"
USE_FP16 = os.environ.get("USE_FP16", "1") == "1"
USE_FOCAL = os.environ.get("USE_FOCAL", "0") == "0"
OVERSAMPLE = os.environ.get("OVERSAMPLE", "1") == "1"
OUTPUT_DIR_BASE = os.environ.get("OUTPUT_DIR_BASE", "./results_fold")
TRAIN_FILE_PATH = "./trainset.json"
DEV_TEST_FILE_PATH = "./dev_testset.json"
MAX_LENGTH = 512
NUM_FOLDS = int(os.environ.get("NUM_FOLDS", 5))
SEED = int(os.environ.get("SEED", 42))
EPOCHS = int(os.environ.get("EPOCHS", 12))
LR = float(os.environ.get("LR", 1e-5))
BATCH = int(os.environ.get("BATCH", 1))
ACCUM_STEPS = int(os.environ.get("ACCUM", 8))
WEIGHT_DECAY = float(os.environ.get("WD", 0.01))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", 0.2))
EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", 5))
LABEL_SMOOTHING = float(os.environ.get("LABEL_SMOOTHING", 0.15))

BNB_AVAILABLE = False
if USE_8BIT:
    try:
        import bitsandbytes
        BNB_AVAILABLE = True
    except:
        BNB_AVAILABLE = False

torch.manual_seed(SEED)
np.random.seed(SEED)

def preprocess_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for entry in data:
        history = entry.get("conversation_history", "")
        responses = entry.get("tutor_responses", {}) or {}
        for tutor_name, details in responses.items():
            ann = details.get("annotation", {}) or {}
            if "Mistake_Identification" in ann:
                rows.append({
                    "history": history,
                    "tutor_name": str(tutor_name),
                    "response": details.get("response", ""),
                    "mistake_label": ann["Mistake_Identification"]
                })
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["history", "response", "tutor_name", "mistake_label"]).reset_index(drop=True)
    return df

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"macro_f1": f1, "accuracy": acc}

class DatathonDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, is_test=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.texts = (df['history'] + " [SEP] " + df['response']).astype(str).tolist()
        if not is_test:
            self.labels = df['labels'].tolist()
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if not self.is_test:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, use_focal=False, label_smoothing=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.use_focal = use_focal
        self.label_smoothing = label_smoothing
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        outputs = model(**inputs)
        logits = outputs.logits
        device = logits.device

        if labels is not None:
            if self.class_weights is not None:
                cw = self.class_weights.to(device)
            else:
                cw = None

            if self.use_focal:
                ce = torch.nn.functional.cross_entropy(logits, labels, weight=cw, reduction='none')
                pt = torch.exp(-ce.detach())
                loss = ((1 - pt) ** 2 * ce).mean()
            else:
                loss_fct = torch.nn.CrossEntropyLoss(weight=cw, label_smoothing=self.label_smoothing)
                loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        else:
        # Evaluation without labels: just return dummy loss
            loss = torch.tensor(0.0, device=logits.device)

        if return_outputs:
            try:
                outputs.logits = outputs.logits.detach()
            except Exception:
                pass
            return loss, outputs
        return loss

def safe_model_load(checkpoint, num_labels, id2label, label2id, use_8bit=False):
    kwargs = dict(num_labels=num_labels, id2label=id2label, label2id=label2id)
    try:
        if use_8bit and BNB_AVAILABLE:
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint, load_in_8bit=True, device_map="auto", **kwargs)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint, low_cpu_mem_usage=True, **kwargs)
        model.config.use_cache = False
        return model
    except:
        if checkpoint != "microsoft/deberta-v3-base":
            return safe_model_load("microsoft/deberta-v3-base", num_labels, id2label, label2id, use_8bit=False)
        raise

def main():
    print("Starting enhanced CV training (aim: high macro-F1)")
    df = preprocess_data(TRAIN_FILE_PATH)
    if df.empty: raise RuntimeError("No data found in TRAIN_FILE_PATH")
    labels_list = sorted(df['mistake_label'].unique().tolist())
    label2id = {l:i for i,l in enumerate(labels_list)}
    id2label = {i:l for l,i in label2id.items()}
    NUM_LABELS = len(labels_list)
    df['labels'] = df['mistake_label'].map(label2id)
    class_weights = torch.tensor(compute_class_weight("balanced", classes=np.arange(NUM_LABELS), y=df['labels'].to_numpy()), dtype=torch.float)

    with open(DEV_TEST_FILE_PATH, 'r', encoding='utf-8') as f:
        raw_test = json.load(f)
    test_rows = []
    for entry in raw_test:
        history = entry.get("conversation_history", "")
        convo_id = entry.get("conversation_id", None)
        for tutor_name, details in (entry.get("tutor_responses", {}) or {}).items():
            test_rows.append({"conversation_id": convo_id, "tutor": tutor_name, "history": history, "response": details.get("response", "")})
    test_df = pd.DataFrame(test_rows)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large" if USE_LARGE else "microsoft/deberta-v3-base", use_fast=True)
    test_dataset = DatathonDataset(test_df, tokenizer, MAX_LENGTH, is_test=True)

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    df['fold'] = -1
    for fold_idx, (_, val_idx) in enumerate(skf.split(df, df['labels'])):
        df.loc[val_idx, "fold"] = fold_idx

    all_val_f1, all_val_acc, all_test_preds, fold_weights = [], [], [], []

    for fold in range(NUM_FOLDS):
        print(f"\n=== FOLD {fold} ===")
        train_df_fold = df[df['fold'] != fold].reset_index(drop=True)
        val_df_fold = df[df['fold'] == fold].reset_index(drop=True)

        if OVERSAMPLE:
            counts = train_df_fold['labels'].value_counts()
            max_count = counts.max()
            parts = []
            for label, cnt in counts.items():
                subset = train_df_fold[train_df_fold['labels']==label]
                if cnt < max_count:
                    add = subset.sample(max_count - cnt, replace=True, random_state=SEED)
                    subset = pd.concat([subset, add], ignore_index=True)
                parts.append(subset)
            train_df_fold = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

        train_dataset = DatathonDataset(train_df_fold, tokenizer, MAX_LENGTH)
        val_dataset = DatathonDataset(val_df_fold, tokenizer, MAX_LENGTH)

        torch.cuda.empty_cache()
        gc.collect()
        model = safe_model_load("microsoft/deberta-v3-large" if USE_LARGE else "microsoft/deberta-v3-base", NUM_LABELS, id2label, label2id, use_8bit=(USE_8BIT and BNB_AVAILABLE))
        model.config.use_cache = False
        if USE_LARGE and torch.cuda.is_available() and not USE_FP16:
            model.gradient_checkpointing_enable()


        training_args = TrainingArguments(
            output_dir=f"{OUTPUT_DIR_BASE}_{fold}",
            num_train_epochs=EPOCHS,
            learning_rate=LR,
            per_device_train_batch_size=BATCH,
            per_device_eval_batch_size=BATCH,
            gradient_accumulation_steps=ACCUM_STEPS,
            warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,
            logging_dir=f"./logs_fold_{fold}",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            save_total_limit=1,
            report_to="none",
            fp16=USE_FP16,
            dataloader_pin_memory=True,
            lr_scheduler_type="cosine",
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            class_weights=class_weights,
            use_focal=USE_FOCAL,
            label_smoothing=LABEL_SMOOTHING,
            data_collator=default_data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
        )

        trainer.train()
        metrics = trainer.evaluate()
        val_f1 = float(metrics.get("eval_macro_f1", 0.0))
        val_acc = float(metrics.get("eval_accuracy", 0.0))
        print(f"Fold {fold} metrics:", metrics)
        all_val_f1.append(val_f1)
        all_val_acc.append(val_acc)
        fold_weights.append(max(0.0, val_f1))
        preds_out = trainer.predict(test_dataset)
        all_test_preds.append(preds_out.predictions)
        del trainer, model
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)

    weights = np.array(fold_weights, dtype=float)
    if weights.sum() <= 0: weights = np.ones(len(all_test_preds), dtype=float)/len(all_test_preds)
    else: weights = weights/weights.sum()
    stacked = np.stack(all_test_preds, axis=0)
    avg = np.tensordot(weights, stacked, axes=(0,0))
    final_nums = np.argmax(avg, axis=1)
    rev_map = {v:k for k,v in label2id.items()}
    final_texts = [rev_map[int(i)] for i in final_nums]

    submission_df = test_df[['conversation_id', 'tutor']].copy()
    submission_df['prediction'] = final_texts
    submission_df.to_csv("submission_cv_ensemble_weighted.csv", index=False)
    print("\nFinal CV Macro F1: {:.4f} ± {:.4f}".format(np.mean(all_val_f1), np.std(all_val_f1)))
    print("Final CV Accuracy: {:.4f} ± {:.4f}".format(np.mean(all_val_acc), np.std(all_val_acc)))
    print("Saved submission_cv_ensemble_weighted.csv")

if __name__ == "__main__":
    main()
