import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

MODEL_NAME = 'roberta-base' 
TRAIN_FILE_PATH = 'trainset.json'
DEV_TEST_FILE_PATH = 'dev-testset.json'
MAX_LENGTH = 512
NUM_FOLDS = 5

raw_train_data = pd.read_json(TRAIN_FILE_PATH)
rows = []
for _, row in raw_train_data.iterrows():
    for tutor, details in row['tutor_responses'].items():
        rows.append({
            'history': row['conversation_history'],
            'response': details['response'],
            'mistake_label': details.get('annotation', {}).get('Mistake_Identification')
        })
df = pd.DataFrame(rows).dropna(subset=['mistake_label'])

raw_test_data = pd.read_json(DEV_TEST_FILE_PATH)
test_rows = []
for _, row in raw_test_data.iterrows():
    for tutor_name, resp_details in row['tutor_responses'].items():
        test_rows.append({
            'conversation_id': row['conversation_id'],
            'tutor': tutor_name,
            'history': row['conversation_history'],
            'response': resp_details['response']
        })
test_df = pd.DataFrame(test_rows)

label_map = {'No': 0, 'To some extent': 1, 'Yes': 2}
df['mistake_label_num'] = df['mistake_label'].map(label_map)

skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
df['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['mistake_label_num'])):
    df.loc[val_idx, 'fold'] = fold

class DatathonDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = (df['history'] + " [SEP] " + df['response']).tolist()
        self.labels = df['mistake_label_num'].tolist()
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
        
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = (df['history'] + " [SEP] " + df['response']).tolist()
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {key: val.squeeze() for key, val in encoding.items()}

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'macro_f1': f1,
        'accuracy': acc,
    }

all_val_f1_scores = []
all_val_acc_scores = []
all_test_predictions = []
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
test_dataset = TestDataset(test_df, tokenizer, MAX_LENGTH)

for fold in range(NUM_FOLDS):
    print(f"\n===== FOLD {fold} of {NUM_FOLDS-1} =====")
    
    train_df = df[df['fold'] != fold]
    val_df = df[df['fold'] == fold]
    train_dataset = DatathonDataset(train_df, tokenizer, MAX_LENGTH)
    val_dataset = DatathonDataset(val_df, tokenizer, MAX_LENGTH)
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_map))
    
    training_args = TrainingArguments(
        output_dir=f'./results_fold_{fold}',
        num_train_epochs=3,
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        greater_is_better=True,
        report_to='none',
        save_total_limit=1,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    
    eval_results = trainer.evaluate()
    all_val_f1_scores.append(eval_results['eval_macro_f1'])
    all_val_acc_scores.append(eval_results['eval_accuracy'])
    
    test_predictions = trainer.predict(test_dataset)
    all_test_predictions.append(test_predictions.predictions)

mean_f1 = np.mean(all_val_f1_scores)
std_f1 = np.std(all_val_f1_scores)
mean_acc = np.mean(all_val_acc_scores)
std_acc = np.std(all_val_acc_scores)

print(f"\n--> Final CV Macro F1 Score: {mean_f1:.4f} +/- {std_f1:.4f}")
print(f"--> Final CV Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")

avg_preds = np.mean(all_test_predictions, axis=0)
final_preds_num = np.argmax(avg_preds, axis=1)

reverse_label_map = {0: 'No', 1: 'To some extent', 2: 'Yes'}
final_preds_text = [reverse_label_map[i] for i in final_preds_num]

submission_df = test_df[['conversation_id', 'tutor']].copy()
submission_df['prediction'] = final_preds_text
submission_df.to_csv('submission_cv.csv', index=False)

print("\n--> 'submission_cv.csv' created successfully!")
print(submission_df.head())
