"""
Starter script: Fine-tuning a Genomic Transformer on Bacterial DNA Sequences
and publishing the model to Hugging Face Hub.
"""

import os
import torch
import numpy as np
from datasets import Dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

def main():
    # 1. Sample Data Preparation (Bacterial DNA Sequences)
    # Replace this sample data with your genomic dataset (FASTA/CSV)
    raw_data = [
        {"sequence": "ATGCTAGCTAGCTAGCTAGCTAGCTAGCTA", "label": 1},
        {"sequence": "GCGCGCGCATATATAGCGATCGATCGATCG", "label": 1},
        {"sequence": "AAAAAAAATTTTTTTTTCCCCCCGGGGGGG", "label": 0},
        {"sequence": "TCGATCGATCGACTAGCTAGCTAGCTAATC", "label": 0},
        {"sequence": "ATGCGATCGATCGATCGATCGATCGAATCG", "label": 1},
        {"sequence": "GGGGGGCCCCCCAAAAAAATTTTTTTTTTT", "label": 0},
    ]

    print("Loading dataset...")
    dataset = Dataset.from_list(raw_data)
    dataset_dict = dataset.train_test_split(test_size=0.3, seed=42)

    # 2. Select Pre-trained Genomic Model & Tokenizer
    MODEL_NAME = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    def tokenize_function(examples):
        return tokenizer(
            examples["sequence"],
            truncation=True,
            max_length=512,
            padding=False
        )

    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

    # 3. Initialize Model & Evaluation Metric
    id2label = {0: "Non-Promoter / Regular", 1: "Promoter"}
    label2id = {"Non-Promoter / Regular": 0, "Promoter": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True
    )

    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    # 4. Training Arguments & Trainer Setup
    hf_username = os.getenv("HF_USERNAME", "your-username")
    repo_id = f"{hf_username}/bacterial-promoter-classifier"

    training_args = TrainingArguments(
        output_dir="./bacterial_promoter_model",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,  # Set to True when ready to publish to HF
        hub_model_id=repo_id,
        logging_steps=10,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Trainer initialized successfully.")

    # Uncomment below to train & push when ready:
    # trainer.train()
    # trainer.push_to_hub(commit_message="Initial release of bacterial promoter classifier")

if __name__ == "__main__":
    main()
