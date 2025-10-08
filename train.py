import pandas as pd
import json
import torch
import gc
import numpy as np
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

# -------------------- CONFIG --------------------
MODEL_CHECKPOINT = "microsoft/deberta-v3-large"
TRAIN_FILE_PATH = "./trainset.json"
DEV_TEST_FILE_PATH = "./dev-testset.json"
MAX_LENGTH = 512
NUM_FOLDS = 5
RANDOM_STATE = 42

# -------------------- DATA PREPROCESS --------------------
def preprocess_data(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    processed_rows = []
    for entry in data:
        for tutor_name, details in entry['tutor_responses'].items():
            if 'annotation' in details and 'Mistake_Identification' in details['annotation']:
                processed_rows.append({
                    'history': entry['conversation_history'],
                    'tutor_name': tutor_name,
                    'response': details['response'],
                    'mistake_label': details['annotation']['Mistake_Identification']
                })
    return pd.DataFrame(processed_rows)

# -------------------- METRICS --------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"macro_f1": f1, "accuracy": acc}

# -------------------- CUSTOM DATASET --------------------
class DatathonDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, is_test=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.texts = (df['history'] + " [SEP] " + df['response']).tolist()
        if not is_test:
            self.labels = df['labels'].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze() for key, val in encoding.items()}
        if not self.is_test:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

# -------------------- FOCAL LOSS --------------------
class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Focal Loss
        ce_loss = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device), reduction='none')
        ce = ce_loss(logits, labels)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** 2 * ce).mean()

        return (loss, outputs) if return_outputs else loss

# -------------------- MAIN --------------------
if __name__ == "__main__":
    print(f"🚀 Starting {NUM_FOLDS}-fold cross-validation...")

    df = preprocess_data(TRAIN_FILE_PATH)

    labels_list = sorted(df['mistake_label'].unique())
    label2id = {label: i for i, label in enumerate(labels_list)}
    id2label = {i: label for label, i in label2id.items()}
    NUM_LABELS = len(labels_list)

    df['labels'] = df['mistake_label'].map(label2id)

    # Class weights
    class_weights = torch.tensor(
        compute_class_weight('balanced', classes=np.arange(NUM_LABELS), y=df['labels'].to_numpy()),
        dtype=torch.float
    )

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['labels'])):
        df.loc[val_idx, 'fold'] = fold

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # Prepare test set
    with open(DEV_TEST_FILE_PATH, 'r', encoding='utf-8') as f:
        raw_test_data = json.load(f)
    test_rows = []
    for entry in raw_test_data:
        for tutor_name, details in entry['tutor_responses'].items():
            test_rows.append({
                'conversation_id': entry['conversation_id'],
                'tutor': tutor_name,
                'history': entry['conversation_history'],
                'response': details['response'],
            })
    test_df = pd.DataFrame(test_rows)
    test_dataset = DatathonDataset(test_df, tokenizer, MAX_LENGTH, is_test=True)

    all_val_f1, all_val_acc, all_test_preds = [], [], []

    for fold in range(NUM_FOLDS):
        print(f"\n===== FOLD {fold} =====")
        train_df_fold = df[df['fold'] != fold]
        val_df_fold = df[df['fold'] == fold]

        train_dataset = DatathonDataset(train_df_fold, tokenizer, MAX_LENGTH)
        val_dataset = DatathonDataset(val_df_fold, tokenizer, MAX_LENGTH)

        torch.cuda.empty_cache()
        gc.collect()

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_CHECKPOINT,
            num_labels=NUM_LABELS,
            id2label=id2label,
            label2id=label2id
        )
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

        training_args = TrainingArguments(
            output_dir=f'./results_fold_{fold}',
            num_train_epochs=20,
            learning_rate=2e-5,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            weight_decay=0.01,
            bf16=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            logging_dir=f"./logs_fold_{fold}",
            save_total_limit=1,
            report_to="none",
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            class_weights=class_weights,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        trainer.train()

        eval_results = trainer.evaluate()
        all_val_f1.append(eval_results['eval_macro_f1'])
        all_val_acc.append(eval_results['eval_accuracy'])

        test_preds = trainer.predict(test_dataset)
        all_test_preds.append(test_preds.predictions)

    # -------------------- ENSEMBLE --------------------
    avg_test_preds = np.mean(all_test_preds, axis=0)
    final_test_preds_num = np.argmax(avg_test_preds, axis=1)
    reverse_label_map = {v: k for k, v in label2id.items()}
    final_test_preds_text = [reverse_label_map[i] for i in final_test_preds_num]

    submission_df = test_df[['conversation_id', 'tutor']].copy()
    submission_df['prediction'] = final_test_preds_text
    submission_df.to_csv('submission_cv_ensemble.csv', index=False)

    print(f"\n--> Final CV Macro F1: {np.mean(all_val_f1):.4f} ± {np.std(all_val_f1):.4f}")
    print(f"--> Final CV Accuracy: {np.mean(all_val_acc):.4f} ± {np.std(all_val_acc):.4f}")
    print("--> 'submission_cv_ensemble.csv' created successfully!")
