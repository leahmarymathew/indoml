# run_cv_safe.py
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import gc
import time
import math
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
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

# Config
USE_LARGE = os.environ.get("USE_LARGE", "0") == "1"
USE_8BIT = os.environ.get("USE_8BIT", "0") == "1"
MODEL_CHECKPOINT = "microsoft/deberta-v3-large" if USE_LARGE else "microsoft/deberta-v3-base"
TRAIN_FILE_PATH = "./trainset.json"
DEV_TEST_FILE_PATH = "./dev-testset.json"
MAX_LENGTH = 512
NUM_FOLDS = 5
RANDOM_STATE = 42
SEED = 42

# bitsandbytes optional import
BNB_AVAILABLE = False
if USE_8BIT:
    try:
        import bitsandbytes  # noqa: F401
        BNB_AVAILABLE = True
    except Exception:
        BNB_AVAILABLE = False

def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

def print_cuda_info():
    print("CUDA available:", torch.cuda.is_available())
    print("Torch:", torch.__version__)
    if torch.cuda.is_available():
        try:
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                free, total = torch.cuda.mem_get_info(i)
                print(f"  GPU {i}: {prop.name} total={total/1024**3:.2f}GB free={free/1024**3:.2f}GB")
        except Exception:
            pass
    print("Using large model:", USE_LARGE)
    print("Requested 8-bit:", USE_8BIT, "Available:", BNB_AVAILABLE)
    print()

def preprocess_data(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for entry in data:
        history = entry.get("conversation_history", "")
        tutor_responses = entry.get("tutor_responses", {}) or {}
        for tutor_name, details in tutor_responses.items():
            if not isinstance(details, dict):
                continue
            annotation = details.get("annotation", {}) or {}
            if "Mistake_Identification" in annotation:
                rows.append({
                    "history": history,
                    "tutor_name": str(tutor_name),
                    "response": details.get("response", ""),
                    "mistake_label": annotation["Mistake_Identification"]
                })
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["history", "response", "tutor_name", "mistake_label"]).reset_index(drop=True)
    return df

def load_test_df(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    rows = []
    for entry in raw:
        history = entry.get("conversation_history", "")
        convo_id = entry.get("conversation_id", None)
        tutor_responses = entry.get("tutor_responses", {}) or {}
        for tutor_name, details in tutor_responses.items():
            rows.append({
                "conversation_id": convo_id,
                "tutor": tutor_name,
                "history": history,
                "response": details.get("response", "")
            })
    return pd.DataFrame(rows)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"macro_f1": f1, "accuracy": acc}

class DatathonDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = (df['history'] + " [SEP] " + df['response']).astype(str).tolist()
        self.labels = df['labels'].tolist() if 'labels' in df.columns else None
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = (df['history'] + " [SEP] " + df['response']).astype(str).tolist()
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {k: v.squeeze(0) for k, v in enc.items()}

def safe_model_load(checkpoint, num_labels, id2label, label2id, use_8bit=False):
    kwargs = dict(num_labels=num_labels, id2label=id2label, label2id=label2id)
    try:
        if use_8bit and BNB_AVAILABLE:
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint, load_in_8bit=True, device_map="auto", **kwargs)
        else:
            # low_cpu_mem_usage reduces memory pressure when loading
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint, low_cpu_mem_usage=True, **kwargs)
        model.config.use_cache = False
        return model
    except Exception as e:
        print("Model load failed:", repr(e))
        print("Falling back to FP32 base model if available.")
        if checkpoint != "microsoft/deberta-v3-base":
            return safe_model_load("microsoft/deberta-v3-base", num_labels, id2label, label2id, use_8bit=False)
        raise

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return loss

if __name__ == "__main__":
    print_cuda_info()
    print("Loading datasets...")
    df = preprocess_data(TRAIN_FILE_PATH)
    test_df = load_test_df(DEV_TEST_FILE_PATH)

    if df.empty:
        raise RuntimeError("No training data found after preprocessing. Check trainset.json structure.")

    labels_list = sorted(df['mistake_label'].unique().tolist())
    label2id = {label: i for i, label in enumerate(labels_list)}
    id2label = {i: label for label, i in label2id.items()}
    NUM_LABELS = len(labels_list)
    df['labels'] = df['mistake_label'].map(label2id)

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    df['fold'] = -1
    for fold_num, (_, val_idx) in enumerate(skf.split(df, df['labels'])):
        df.loc[val_idx, 'fold'] = fold_num

    class_weights = compute_class_weight('balanced', classes=np.unique(df['labels']), y=df['labels'].to_numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)
    test_dataset = TestDataset(test_df, tokenizer, MAX_LENGTH)

    all_val_f1 = []
    all_val_acc = []
    all_test_preds = []

    for fold in range(NUM_FOLDS):
        print(f"\n--- Starting fold {fold} ---")
        fold_train_df = df[df['fold'] != fold].reset_index(drop=True)
        fold_val_df = df[df['fold'] == fold].reset_index(drop=True)

        train_dataset = DatathonDataset(fold_train_df, tokenizer, MAX_LENGTH)
        val_dataset = DatathonDataset(fold_val_df, tokenizer, MAX_LENGTH)

        torch.cuda.empty_cache()
        gc.collect()

        model = safe_model_load(MODEL_CHECKPOINT, NUM_LABELS, id2label, label2id, use_8bit=(USE_8BIT and BNB_AVAILABLE))
        # don't enable gradient checkpointing for stability
        model.config.use_cache = False

        training_args = TrainingArguments(
            output_dir=f"./results_fold_{fold}",
            num_train_epochs=8,
            learning_rate=2e-5,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_ratio=0.1,
            weight_decay=0.01,
            bf16=False,
            fp16=False,
            logging_dir=f"./logs_fold_{fold}",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            save_total_limit=1,
            report_to="none",
            dataloader_pin_memory=True,
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        # single-step sanity check
        print("Running single-step sanity check (forward+backward+opt) ...")
        try:
            sample = [train_dataset[i] for i in range(min(2, len(train_dataset)))]
            batch = default_data_collator(sample)
            device = trainer.model.device
            batch = {k: v.to(device) for k, v in batch.items()}
            if trainer.optimizer is None:
                trainer.create_optimizer_and_scheduler(num_training_steps=1)
            trainer.model.train()
            trainer.optimizer.zero_grad(set_to_none=True)
            loss = trainer.compute_loss(trainer.model, dict(batch))
            print(" single-step loss:", float(loss.detach().cpu().numpy()))
            loss.backward()
            trainer.optimizer.step()
            print(" single-step optimizer step OK")
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print("Single-step sanity test failed on fold", fold, "error:", repr(e))
            print("Suggested actions: (1) set USE_8BIT=1 and install bitsandbytes; (2) set USE_LARGE=0 to use base model; (3) request larger GPU slice")
            raise

        # train
        print("Starting training for fold", fold)
        trainer.train()

        # evaluate
        metrics = trainer.evaluate()
        all_val_f1.append(metrics.get("eval_macro_f1", 0.0))
        all_val_acc.append(metrics.get("eval_accuracy", 0.0))
        print(f"Fold {fold} eval results:", metrics)

        # test predictions
        preds_out = trainer.predict(test_dataset)
        all_test_preds.append(preds_out.predictions)

        # cleanup between folds
        del trainer
        del model
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)

    mean_f1 = float(np.mean(all_val_f1))
    std_f1 = float(np.std(all_val_f1))
    mean_acc = float(np.mean(all_val_acc))
    std_acc = float(np.std(all_val_acc))
    print(f"\nFinal CV Macro F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Final CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    avg_preds = np.mean(np.stack(all_test_preds, axis=0), axis=0)
    final_preds_num = np.argmax(avg_preds, axis=1)
    rev_map = {v: k for k, v in label2id.items()}
    final_preds_text = [rev_map[int(i)] for i in final_preds_num]

    submission_df = test_df[['conversation_id', 'tutor']].copy()
    submission_df['prediction'] = final_preds_text
    submission_df.to_csv("submission_cv.csv", index=False)
    print("\nSaved submission_cv.csv. Head:")
    print(submission_df.head())
