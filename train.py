# run_experiment_safe.py  (paste/replace your run_experiment.py with this)
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"    # disable torch._compile / dynamo for stability
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import gc
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

# --- 1. CONFIGURATION ---
TRAIN_FILE_PATH = "./trainset.json"
MODEL_CHECKPOINT = "microsoft/deberta-v3-large"  # switch to base if still failing
OUTPUT_DIR = "./final_model_for_submission"

# --- 2. HELPER FUNCTION ---
def preprocess_data(filepath: str) -> pd.DataFrame:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    processed_rows = []
    for entry in data:
        for tutor_name, details in entry.get('tutor_responses', {}).items():
            if 'annotation' in details and 'Mistake_Identification' in details['annotation']:
                processed_rows.append({
                    'history': entry.get('conversation_history', ''),
                    'tutor_name': tutor_name,
                    'response': details.get('response', ''),
                    'mistake_label': details['annotation']['Mistake_Identification']
                })
    return pd.DataFrame(processed_rows)

# --- 3. MAIN EXECUTION SCRIPT ---
if __name__ == "__main__":
    print("🚀 Starting FINAL training process (safe mode)...")

    # clear caches
    torch.cuda.empty_cache()
    gc.collect()

    train_df = preprocess_data(TRAIN_FILE_PATH)
    print(f"✅ Full training data loaded. Training set size: {len(train_df)}")

    train_df['input_text'] = (
        "tutor: " + train_df['tutor_name'] +
        " [SEP] dialogue: " + train_df['history'] +
        " [SEP] response: " + train_df['response']
    )
    labels_list = sorted(train_df['mistake_label'].unique().tolist())
    label2id = {label: i for i, label in enumerate(labels_list)}
    id2label = {i: label for label, i in label2id.items()}
    NUM_LABELS = len(labels_list)

    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_df['mistake_label']),
        y=train_df['mistake_label'].to_numpy()
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    train_df['labels'] = train_df['mistake_label'].map(label2id)
    train_dataset = Dataset.from_pandas(train_df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    def tokenize_function(examples):
        return tokenizer(examples['input_text'], padding="max_length", truncation=True, max_length=512)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_train_dataset = tokenized_train_dataset.remove_columns([c for c in tokenized_train_dataset.column_names if c not in ("input_ids", "attention_mask", "labels")])
    print("✅ Tokenization complete.")

    # --- CustomTrainer with safe compute_loss (only returns loss) ---
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return loss  # ALWAYS return just the scalar loss (no outputs)

    # --- Model loading (standard) ---
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    )
    # Do NOT call model.gradient_checkpointing_enable() here for stability
    model.config.use_cache = False

    # --- Training arguments (conservative / stable) ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=8,
        learning_rate=2e-5,
        per_device_train_batch_size=1,    # minimal per-step memory
        gradient_accumulation_steps=16,   # accumulate to simulate larger batch
        warmup_ratio=0.1,
        bf16=False,                       # disable bf16/fp16 to avoid AMP issues
        fp16=False,
        weight_decay=0.01,
        logging_dir="./logs_final",
        save_strategy="epoch",
        report_to="none",
        dataloader_pin_memory=True,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    print("✅ Model and Trainer are ready.")

    # --- SINGLE-STEP SANITY CHECK (forward + backward + optimizer step) ---
    print("\n🔎 Running single-step sanity test (this checks forward+backward+opt)...")
    try:
        # build a tiny batch
        d = tokenized_train_dataset[:2]
        batch = default_data_collator(d)
        device = trainer.model.device
        batch = {k: v.to(device) for k, v in batch.items()}

        opt = trainer.create_optimizer_and_scheduler(num_training_steps=1)[0] if getattr(trainer, "optimizer", None) is None else trainer.optimizer
        # ensure optimizer exists on trainer
        if trainer.optimizer is None:
            trainer.optimizer = opt

        trainer.model.train()
        trainer.optimizer.zero_grad()
        loss = trainer.compute_loss(trainer.model, batch)  # scalar loss
        print("  -> single-step loss computed:", float(loss.detach().cpu().numpy()))
        loss.backward()
        trainer.optimizer.step()
        print("  -> single-step optimizer step OK")
        # clear caches
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print("❗ Single-step sanity test FAILED. Error below:")
        print(repr(e))
        print("\nActionable suggestions:")
        print(" 1) Try MODEL_CHECKPOINT = 'microsoft/deberta-v3-base'")
        print(" 2) If you can, enable bitsandbytes 8-bit loading (commented example in file)")
        print(" 3) Request a larger MIG slice / full GPU from admin")
        raise

    # --- Start full training ---
    print("\n🏋️ Starting full training now...")
    trainer.train()
    print("\n🎉 Training finished!")

    trainer.save_model(OUTPUT_DIR)
    print(f"\n✨ Final model saved to {OUTPUT_DIR}")

# -------------------------
# Optional: bitsandbytes 8-bit loading (uncomment and install bitsandbytes & accelerate)
# from transformers import BitsAndBytesConfig
# bnb_config = BitsAndBytesConfig(load_in_8bit=True)
# model = AutoModelForSequenceClassification.from_pretrained(
#     MODEL_CHECKPOINT,
#     quantization_config=bnb_config,
#     device_map="auto",
#     num_labels=NUM_LABELS,
# )
# If you use bitsandbytes, set training_args.optim to "adamw_torch" or use accelerate config.
