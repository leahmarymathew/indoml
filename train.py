# train_safe.py
"""
Robust training script for Mistake Identification.
- Defaults chosen to minimize CUDA allocator / optimizer / AMP issues on MIG slices.
- Optional 8-bit quantization via bitsandbytes can be enabled with env var USE_8BIT=1 (and bitsandbytes installed).
- To use the large model set env var USE_LARGE=1 (only do this if you have a large contiguous GPU allocation).
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"   # disable torch.compile / dynamo
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import gc
import time
import math
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
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

# --- Optional bitsandbytes import (only used if present and requested) ---
USE_8BIT = os.environ.get("USE_8BIT", "0") == "1"
try:
    if USE_8BIT:
        import bitsandbytes as bnb  # noqa: F401
        BNB_AVAILABLE = True
    else:
        BNB_AVAILABLE = False
except Exception:
    BNB_AVAILABLE = False

# --- CONFIGURATION ---
TRAIN_FILE_PATH = "./trainset.json"
# default to base (safer). Set env var USE_LARGE=1 to attempt large (only if you know your GPU can handle it)
MODEL_CHECKPOINT = "microsoft/deberta-v3-base" if os.environ.get("USE_LARGE", "0") != "1" else "microsoft/deberta-v3-large"
OUTPUT_DIR = "./experiment_results_safe"
SEED = 42

# --- HELPERS: Preprocess & Metrics ---
def preprocess_data(filepath: str):
    """
    Expected JSON structure per your earlier messages:
    [
      {
        "conversation_history": "...",
        "tutor_responses": {
            "TutorName": {
                "response": "...",
                "annotation": {
                    "Mistake_Identification": "<label>"
                }
            },
            ...
        }
      },
      ...
    ]
    This function extracts rows of (history, tutor_name, response, mistake_label).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for entry in data:
        history = entry.get("conversation_history", "")
        tutor_responses = entry.get("tutor_responses", {}) or {}
        for tutor_name, details in tutor_responses.items():
            if not isinstance(details, dict):
                continue
            response = details.get("response", "")
            annotation = details.get("annotation", {}) or {}
            if "Mistake_Identification" in annotation:
                label = annotation["Mistake_Identification"]
                rows.append({
                    "history": history,
                    "tutor_name": str(tutor_name),
                    "response": response,
                    "mistake_label": label,
                })
    df = pd.DataFrame(rows)
    # Minimal cleaning
    df = df.dropna(subset=["history", "response", "tutor_name", "mistake_label"])
    df = df.reset_index(drop=True)
    return df

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    micro_f1 = f1_score(labels, preds, average="micro")
    return {"accuracy": acc, "macro_f1": macro_f1, "micro_f1": micro_f1}


# --- Set deterministic seed ---
def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# --- Utility: print device info & free memory visible to PyTorch ---
def print_cuda_info():
    print("CUDA available:", torch.cuda.is_available())
    print("Torch version:", torch.__version__)
    if torch.cuda.is_available():
        try:
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                free, total = torch.cuda.mem_get_info(i)
                print(f"  GPU {i}: {prop.name}  total={total/1024**3:.2f} GB  free={free/1024**3:.2f} GB")
        except Exception as e:
            print("  (could not get per-device mem info)", repr(e))
    print("bitsandbytes available:", BNB_AVAILABLE)
    print("Using 8-bit requested via env USE_8BIT=1:", USE_8BIT)
    print("Selected model checkpoint:", MODEL_CHECKPOINT)
    print("")

# --- MAIN ---
def main():
    print("\n=== Robust Training Script (safe defaults) ===\n")
    print_cuda_info()

    print("Loading and preprocessing data...")
    df = preprocess_data(TRAIN_FILE_PATH)
    if df.empty:
        raise RuntimeError(f"No training rows found in {TRAIN_FILE_PATH}. Check JSON structure.")
    print(f"Total rows extracted: {len(df)}")

    # train/dev split
    train_df, dev_df = train_test_split(df, test_size=0.1, random_state=SEED, stratify=df["mistake_label"])
    print(f"Split -> train: {len(train_df)}, dev: {len(dev_df)}")

    # Compose input_text
    train_df["input_text"] = (
        "tutor: " + train_df["tutor_name"] + " [SEP] dialogue: " + train_df["history"] + " [SEP] response: " + train_df["response"]
    )
    dev_df["input_text"] = (
        "tutor: " + dev_df["tutor_name"] + " [SEP] dialogue: " + dev_df["history"] + " [SEP] response: " + dev_df["response"]
    )

    labels_list = sorted(train_df["mistake_label"].unique().tolist())
    label2id = {label: i for i, label in enumerate(labels_list)}
    id2label = {i: label for label, i in label2id.items()}
    NUM_LABELS = len(labels_list)
    print(f"Detected labels ({NUM_LABELS}): {labels_list}")

    # class weights
    class_weights = compute_class_weight("balanced", classes=np.unique(train_df["mistake_label"]), y=train_df["mistake_label"].to_numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print("Computed class weights:", class_weights.tolist())

    train_df["labels"] = train_df["mistake_label"].map(label2id)
    dev_df["labels"] = dev_df["mistake_label"].map(label2id)

    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)

    def tokenize_function(examples):
        return tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=512)

    print("Tokenizing datasets (this may take a little)...")
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_dev_dataset = dev_dataset.map(tokenize_function, batched=True)

    # keep only necessary columns for Trainer
    keep_cols = [c for c in tokenized_train_dataset.column_names if c in ("input_ids", "attention_mask", "labels")]
    tokenized_train_dataset = tokenized_train_dataset.remove_columns([c for c in tokenized_train_dataset.column_names if c not in keep_cols])
    tokenized_dev_dataset = tokenized_dev_dataset.remove_columns([c for c in tokenized_dev_dataset.column_names if c not in keep_cols])

    print("Tokenization complete. Sample columns:", tokenized_train_dataset.column_names)

    # small collator sanity test (fixes the earlier error you saw)
    sample_list = [tokenized_train_dataset[i] for i in range(min(2, len(tokenized_train_dataset)))]
    collated = default_data_collator(sample_list)
    print("Default data collator sanity check OK. Collated keys:", list(collated.keys()))

    torch.cuda.empty_cache()
    gc.collect()

    # Decide how to load model:
    model_kwargs = dict(num_labels=NUM_LABELS, id2label=id2label, label2id=label2id)
    use_8bit = BNB_AVAILABLE and USE_8BIT

    if use_8bit:
        print("bitsandbytes detected and USE_8BIT=1 -> loading model in 8-bit with device_map='auto' (safe for memory).")
        # note: transformers >= 4.31 supports BitsAndBytesConfig; older may accept load_in_8bit directly
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_CHECKPOINT,
                load_in_8bit=True,
                device_map="auto",
                **model_kwargs
            )
        except Exception as e:
            print("8-bit loading failed, falling back to normal fp32 load. Error:", repr(e))
            use_8bit = False

    if not use_8bit:
        print("Loading model in FP32 (no 8-bit). If you have trouble with memory consider setting USE_8BIT=1 and installing bitsandbytes.")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, **model_kwargs)

    # For stability: do NOT enable gradient checkpointing by default (it caused backward/graph issues previously)
    # model.gradient_checkpointing_enable()  # <-- intentionally disabled
    model.config.use_cache = False

    # --- Custom Trainer (safe compute_loss: return only loss) ---
    
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    # --- Training arguments (conservative defaults) ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=8,
        learning_rate=2e-5,
        per_device_train_batch_size=1,     # minimal per-step memory footprint
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,    # accumulate to achieve larger effective batch
        warmup_ratio=0.06,
        weight_decay=0.01,
        bf16=False,    # disabled by default for stability
        fp16=False,
        logging_dir="./logs_safe",
        eval_strategy="epoch",
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
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_dev_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )

    print("\nTrainer ready. Performing single-step sanity test (forward + backward + optimizer.step).")

    # Single-step sanity test (safe)
    try:
        # get a collated batch (list -> collator -> move to device)
        sample = [tokenized_train_dataset[i] for i in range(min(2, len(tokenized_train_dataset)))]
        batch = default_data_collator(sample)
        device = trainer.model.device
        batch = {k: v.to(device) for k, v in batch.items()}

        # ensure optimizer present
        if trainer.optimizer is None:
            trainer.create_optimizer_and_scheduler(num_training_steps=1)

        trainer.model.train()
        trainer.optimizer.zero_grad(set_to_none=True)
        loss = trainer.compute_loss(trainer.model, dict(batch))  # scalar loss
        print("  single-step loss computed:", float(loss.detach().cpu().numpy()))
        loss.backward()
        trainer.optimizer.step()
        print("  single-step optimizer step OK")
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print("\n!!! Single-step sanity test FAILED. Error below:")
        print(repr(e))
        print("\nSuggested actions (pick one):")
        print(" 1) If you are on a small MIG slice, set USE_8BIT=1 and install bitsandbytes (pip install bitsandbytes).")
        print(" 2) Switch to the base model (default) or request a larger GPU slice / full GPU from admin.")
        print(" 3) If you set USE_LARGE=1 earlier, unset it and retry with the default base model.")
        raise

    # Start training
    print("\nStarting full training now...")
    train_start = time.time()
    trainer.train()
    train_end = time.time()
    print(f"Training finished in {math.floor(train_end - train_start)} seconds.")

    # Final evaluation
    print("\n--- FINAL EVALUATION RESULTS ---")
    final_evaluation = trainer.evaluate()
    print(final_evaluation)

    # Save final model
    trainer.save_model(OUTPUT_DIR)
    print(f"\nModel saved to: {OUTPUT_DIR}")
    print("Done.")

if __name__ == "__main__":
    main()
