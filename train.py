# train.py (stable, copy-paste replacement)
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import gc
import time
import math
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

# -------------------- CONFIG --------------------
# set env variables to change behavior if needed:
# USE_LARGE=1 to try large model (only if you have big GPU); USE_8BIT=1 to use bitsandbytes
USE_LARGE = os.environ.get("USE_LARGE", "0") == "1"
USE_8BIT = os.environ.get("USE_8BIT", "0") == "1"
ENABLE_CHECKPOINTING = False   # set True only after confirming stability
USE_FP16 = False               # set True only if your env/GPU & torch support it reliably

MODEL_CHECKPOINT = "microsoft/deberta-v3-large" if USE_LARGE else "microsoft/deberta-v3-base"
TRAIN_FILE_PATH = "./trainset.json"
DEV_TEST_FILE_PATH = "./dev_testset.json"
MAX_LENGTH = 512
NUM_FOLDS = 5
RANDOM_STATE = 42
SEED = 42
OUTPUT_BASE = "./results_fold"

# optional bitsandbytes import
BNB_AVAILABLE = False
if USE_8BIT:
    try:
        import bitsandbytes  # noqa
        BNB_AVAILABLE = True
    except Exception:
        BNB_AVAILABLE = False

# -------------------- HELPERS --------------------
def set_seed(s=SEED):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

set_seed()

def preprocess_data(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for entry in data:
        history = entry.get("conversation_history", "")
        responses = entry.get("tutor_responses", {}) or {}
        for tutor_name, details in responses.items():
            if not isinstance(details, dict):
                continue
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
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if not self.is_test:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# -------------------- CUSTOM TRAINER (stable) --------------------
class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # keep inputs local copy to avoid modifying external dict
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            cw = self.class_weights.to(logits.device)
        else:
            cw = None

        loss_fct = torch.nn.CrossEntropyLoss(weight=cw)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        # If Trainer asked for outputs (eval/predict), return them but detach tensors to avoid retaining graph
        if return_outputs:
            try:
                # detach tensors inside outputs if possible
                if hasattr(outputs, "logits"):
                    outputs.logits = outputs.logits.detach()
            except Exception:
                pass
            return loss, outputs
        return loss

# -------------------- SAFE MODEL LOAD --------------------
def safe_model_load(checkpoint, num_labels, id2label, label2id, use_8bit=False):
    kwargs = dict(num_labels=num_labels, id2label=id2label, label2id=label2id)
    try:
        if use_8bit and BNB_AVAILABLE:
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint, load_in_8bit=True, device_map="auto", **kwargs)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint, low_cpu_mem_usage=True, **kwargs)
        model.config.use_cache = False
        # enable checkpointing only if explicitly allowed
        if ENABLE_CHECKPOINTING:
            try:
                model.gradient_checkpointing_enable()
                model.config.use_cache = False
            except Exception:
                pass
        return model
    except Exception as e:
        print("Model load failed:", repr(e))
        if checkpoint != "microsoft/deberta-v3-base":
            print("Falling back to base model.")
            return safe_model_load("microsoft/deberta-v3-base", num_labels, id2label, label2id, use_8bit=False)
        raise

# -------------------- MAIN --------------------
def main():
    print("Starting stable CV training...")
    print("Model checkpoint:", MODEL_CHECKPOINT, "USE_LARGE:", USE_LARGE, "USE_8BIT:", USE_8BIT, "ENABLE_CHECKPOINTING:", ENABLE_CHECKPOINTING, "USE_FP16:", USE_FP16)
    df = preprocess_data(TRAIN_FILE_PATH)
    if df.empty:
        raise RuntimeError("No training rows found. Check TRAIN_FILE_PATH")

    labels_list = sorted(df['mistake_label'].unique().tolist())
    label2id = {label: i for i, label in enumerate(labels_list)}
    id2label = {i: label for label, i in label2id.items()}
    NUM_LABELS = len(labels_list)
    df['labels'] = df['mistake_label'].map(label2id)

    class_weights = torch.tensor(compute_class_weight("balanced", classes=np.arange(NUM_LABELS), y=df['labels'].to_numpy()), dtype=torch.float)

    # prepare test data
    with open(DEV_TEST_FILE_PATH, 'r', encoding='utf-8') as f:
        raw_test = json.load(f)
    test_rows = []
    for entry in raw_test:
        history = entry.get("conversation_history", "")
        convo_id = entry.get("conversation_id", None)
        for tutor_name, details in (entry.get("tutor_responses", {}) or {}).items():
            test_rows.append({"conversation_id": convo_id, "tutor": tutor_name, "history": history, "response": details.get("response", "")})
    test_df = pd.DataFrame(test_rows)
    test_dataset = DatathonDataset(test_df, AutoTokenizer.from_pretrained(MODEL_CHECKPOINT), MAX_LENGTH, is_test=True)

    # split folds
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    df['fold'] = -1
    for fold_idx, (_, val_idx) in enumerate(skf.split(df, df['labels'])):
        df.loc[val_idx, "fold"] = fold_idx

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)
    all_val_f1, all_val_acc, all_test_preds = [], [], []

    for fold in range(NUM_FOLDS):
        print(f"\n===== FOLD {fold} =====")
        train_df_fold = df[df['fold'] != fold].reset_index(drop=True)
        val_df_fold = df[df['fold'] == fold].reset_index(drop=True)

        train_dataset = DatathonDataset(train_df_fold, tokenizer, MAX_LENGTH)
        val_dataset = DatathonDataset(val_df_fold, tokenizer, MAX_LENGTH)

        # free memory
        torch.cuda.empty_cache()
        gc.collect()

        model = safe_model_load(MODEL_CHECKPOINT, NUM_LABELS, id2label, label2id, use_8bit=(USE_8BIT and BNB_AVAILABLE))

        training_args = TrainingArguments(
            output_dir=f"{OUTPUT_BASE}_{fold}",
            num_train_epochs=8,
            learning_rate=2e-5,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            weight_decay=0.01,
            bf16=False,
            fp16=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            logging_dir=f"./logs_fold_{fold}",
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
            class_weights=class_weights,
            data_collator=default_data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        # single-step sanity test
        print("Running single-step sanity test (forward+backward+opt) ...")
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
            print("Single-step sanity test FAILED. Error:", repr(e))
            print("Suggestions: set USE_8BIT=1 and install bitsandbytes, set USE_LARGE=0 (use base), or request larger GPU slice.")
            raise

        # train
        print("Starting training for fold", fold)
        trainer.train()

        # evaluate
        metrics = trainer.evaluate()
        all_val_f1.append(metrics.get("eval_macro_f1", 0.0))
        all_val_acc.append(metrics.get("eval_accuracy", 0.0))
        print(f"Fold {fold} metrics:", metrics)

        # test predictions
        preds_out = trainer.predict(test_dataset)
        all_test_preds.append(preds_out.predictions)

        # cleanup
        del trainer
        del model
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)

    # ensemble & save
    avg_preds = np.mean(np.stack(all_test_preds, axis=0), axis=0)
    final_preds_num = np.argmax(avg_preds, axis=1)
    rev = {v: k for k, v in label2id.items()}
    final_preds_text = [rev[int(i)] for i in final_preds_num]
    submission_df = test_df[['conversation_id', 'tutor']].copy()
    submission_df['prediction'] = final_preds_text
    submission_df.to_csv("submission_cv_ensemble.csv", index=False)

    print(f"\nFinal CV Macro F1: {np.mean(all_val_f1):.4f} ± {np.std(all_val_f1):.4f}")
    print(f"Final CV Accuracy: {np.mean(all_val_acc):.4f} ± {np.std(all_val_acc):.4f}")
    print("Saved submission_cv_ensemble.csv")

if __name__ == "__main__":
    main()
