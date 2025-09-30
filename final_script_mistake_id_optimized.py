# final_script_mistake_id_optimized.py

import pandas as pd
import json
import torch
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# --- 1. CONFIGURATION ---
TRAIN_FILE_PATH = "./trainset.json" # Assumes your trainset.json is in the current directory
MODEL_CHECKPOINT = "microsoft/deberta-v3-large"
OUTPUT_DIR = "./results_mistake_id_base_finalll"

# --- 2. HELPER FUNCTIONS ---

def preprocess_data(filepath: str) -> pd.DataFrame:
    """Loads and flattens the JSON data for the Mistake Identification task."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: The file was not found at {filepath}")
        exit()

    processed_rows = []
    for entry in data:
        for tutor_name, details in entry['tutor_responses'].items():
            # Ensure the annotation exists to avoid errors
            if 'annotation' in details and 'Mistake_Identification' in details['annotation']:
                processed_rows.append({
                    'history': entry['conversation_history'],
                    'tutor_name': tutor_name,
                    'response': details['response'],
                    'mistake_label': details['annotation']['Mistake_Identification']
                })
    return pd.DataFrame(processed_rows)

def compute_metrics(eval_pred):
    """Calculates Accuracy, Macro F1, Macro Precision, and Macro Recall."""
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    
    # Use zero_division=0 to handle cases where a class is not predicted
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    macro_precision = precision_score(labels, predictions, average="macro", zero_division=0)
    macro_recall = recall_score(labels, predictions, average="macro", zero_division=0)
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
    }

# --- 3. MAIN EXECUTION SCRIPT ---
if __name__ == "__main__":
    print("🚀 Starting optimized training process for Mistake Identification...")
    
    full_train_df = preprocess_data(TRAIN_FILE_PATH)
    train_df, dev_df = train_test_split(
        full_train_df, test_size=0.1, random_state=42, stratify=full_train_df['mistake_label']
    )
    print(f"✅ Data loaded. Training set size: {len(train_df)}, Dev set size: {len(dev_df)}")

    # Add the tutor_name feature to the input text
    train_df['input_text'] = "tutor: " + train_df['tutor_name'] + " [SEP] dialogue: " + train_df['history'] + " [SEP] response: " + train_df['response']
    dev_df['input_text'] = "tutor: " + dev_df['tutor_name'] + " [SEP] dialogue: " + dev_df['history'] + " [SEP] response: " + dev_df['response']

    labels_list = sorted(train_df['mistake_label'].unique().tolist()) # Sort for consistency
    label2id = {label: i for i, label in enumerate(labels_list)}
    id2label = {i: label for label, i in label2id.items()}
    NUM_LABELS = len(labels_list)
    
    print("Calculating class weights for Mistake Identification...")
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(train_df['mistake_label']), 
        y=train_df['mistake_label'].to_numpy()
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).cuda()
    print(f"✅ Class Weights: {class_weights}")

    train_df['labels'] = train_df['mistake_label'].map(label2id)
    dev_df['labels'] = dev_df['mistake_label'].map(label2id)

    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    def tokenize_function(examples):
        return tokenizer(examples['input_text'], padding="max_length", truncation=True, max_length=512)
    
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_dev_dataset = dev_dataset.map(tokenize_function, batched=True)
    print("✅ Tokenization complete.")

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # The loss function is expecting the weights on the same device as the logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=NUM_LABELS, id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=20, # High epoch count, but Early Stopping will prevent overfitting
        learning_rate=2e-5,
        per_device_train_batch_size=4, # Increased slightly, adjust if you get memory errors
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4, # Effective batch size = 4 * 4 = 16
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir="./logs_mistake_id_final",
        # FIXED: Changed from 'evaluation_strategy' to 'eval_strategy'
        eval_strategy="steps",
        eval_steps=20,
        save_steps=20,
        load_best_model_at_end=True, # This is key
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none"
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_dev_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=13)]
    )
    print("✅ Model and Trainer are ready.")

    print("\nStarting training for Mistake Identification... 🏋")
    trainer.train()
    print("\n🎉 Training finished!")

    print("\nEvaluating the best model on the dev set...")
    final_evaluation = trainer.evaluate()
    
    print("\n--- FINAL EVALUATION RESULTS (Mistake Identification) ---")
    for key, value in final_evaluation.items():
        print(f"{key.replace('_', ' ').title()}: {value:.4f}")
    print("---------------------------------------------------------")
    print(f"\n✨ Best model saved to {OUTPUT_DIR}")