# generate_submission.py

import pandas as pd
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# --- 1. CONFIGURATION ---
# Path to your saved model (the 'output_dir' from your training script)
SAVED_MODEL_PATH = "./results_mistake_id_base_finalll" 
# Path to the new, unlabeled test file from the organizers
TEST_FILE_PATH = "./testset.json" 
SUBMISSION_FILE_PATH = "./submission.csv"

if __name__ == "__main__":
    print(f"🚀 Loading best model from: {SAVED_MODEL_PATH}")
    
    model = AutoModelForSequenceClassification.from_pretrained(SAVED_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(SAVED_MODEL_PATH)

    print("✅ Model and tokenizer loaded successfully.")

    try:
        with open(TEST_FILE_PATH, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: The test file was not found at {TEST_FILE_PATH}")
        exit()

    processed_rows = []
    for entry in test_data:
        for tutor_name, details in entry['tutor_responses'].items():
            processed_rows.append({
                'conversation_id': entry['conversation_id'],
                'tutor': tutor_name, # Renamed for submission file
                'history': entry['conversation_history'],
                'tutor_name': tutor_name,
                'response': details['response'],
            })
    test_df = pd.DataFrame(processed_rows)

    test_df['input_text'] = "tutor: " + test_df['tutor_name'] + " [SEP] dialogue: " + test_df['history'] + " [SEP] response: " + test_df['response']
    
    test_dataset = Dataset.from_pandas(test_df)
    
    def tokenize_function(examples):
        return tokenizer(examples['input_text'], padding="max_length", truncation=True, max_length=512)
        
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
    print("✅ Test data processed and tokenized.")

    inference_args = TrainingArguments(
        output_dir="./inference_results",
        per_device_eval_batch_size=8,
        report_to="none"
    )

    trainer = Trainer(model=model, args=inference_args)
    
    print("🧠 Generating predictions on the test set...")
    predictions = trainer.predict(tokenized_test_dataset)
    predicted_class_ids = predictions.predictions.argmax(axis=-1)
    
    predicted_labels = [model.config.id2label[i] for i in predicted_class_ids]

    submission_df = test_df[['conversation_id', 'tutor']].copy()
    submission_df['prediction'] = predicted_labels


    submission_df.to_csv(SUBMISSION_FILE_PATH, index=False)
    
    print(f"\n Submission file '{SUBMISSION_FILE_PATH}' created successfully!")
    print(submission_df.head())